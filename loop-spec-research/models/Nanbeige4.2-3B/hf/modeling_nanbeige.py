# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_nanbeige import NanbeigeConfig

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:  # older transformers
    try:
        from transformers.modeling_utils import AttentionInterface as _AttentionInterface

        ALL_ATTENTION_FUNCTIONS = _AttentionInterface
    except ImportError:
        ALL_ATTENTION_FUNCTIONS = None


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "NanbeigeConfig"

DepthAttentionCacheEntry = Tuple[int, torch.Tensor, torch.Tensor]

_SDPA_MASK_SUPPORTS_IS_TRAINING = (
    "is_training" in inspect.signature(AttentionMaskConverter._ignore_causal_mask_sdpa).parameters
)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False

    limit = math.isqrt(value)
    for factor in range(3, limit + 1, 2):
        if value % factor == 0:
            return False
    return True


def _next_prime_after(value: float) -> int:
    candidate = int(value) + 1
    if candidate <= 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1

    while not _is_prime(candidate):
        candidate += 2
    return candidate


def _ngram_embedding_vocab_sizes(m: float, num_tables: int, force_prime: bool) -> List[int]:
    if not force_prime:
        return [int(m + index * 2 + 1) for index in range(num_tables)]

    vocab_sizes = []
    previous = m
    for _ in range(num_tables):
        previous = _next_prime_after(previous)
        vocab_sizes.append(previous)
    return vocab_sizes


def _ngram_hash_base(vocab_size: int, force_prime: bool) -> int:
    if not force_prime:
        return vocab_size
    return _next_prime_after(vocab_size)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_tensor: torch.Tensor,
    past_key_values_length: int,
    is_training: bool,
) -> bool:
    kwargs = {
        "attention_mask": attention_mask,
        "inputs_embeds": input_tensor,
        "past_key_values_length": past_key_values_length,
    }
    if _SDPA_MASK_SUPPORTS_IS_TRAINING:
        kwargs["is_training"] = is_training
    return AttentionMaskConverter._ignore_causal_mask_sdpa(**kwargs)


def _get_loop_cache_layer_idx(
    layer_idx: Optional[int],
    loop_idx: int,
    num_hidden_layers: int,
    cache_layer_idx: Optional[int] = None,
) -> int:
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop-aware caching is enabled.")
    if cache_layer_idx is not None:
        return cache_layer_idx
    return layer_idx + loop_idx * num_hidden_layers


def _is_vllm_backend(config: NanbeigeConfig) -> bool:
    return getattr(config, "_attn_implementation", None) == "vllm"


def _get_num_physical_layers(config: NanbeigeConfig) -> int:
    """Physical decoder depth (checkpoint keys). Safe after KV layer expand."""
    physical = getattr(config, "num_physical_layers", None)
    if physical is not None:
        return physical
    return config.num_hidden_layers


def _prepare_config_for_vllm(config: NanbeigeConfig) -> int:
    """Pin physical depth and expand ``num_hidden_layers`` for vLLM KV slots.
    vLLM probes ``AutoModel.from_config`` on the same config object before the
    real build; pinning ``num_physical_layers`` keeps ModuleList length stable.
    """
    physical = getattr(config, "num_physical_layers", None)
    if physical is None:
        physical = config.num_hidden_layers
        config.num_physical_layers = physical
    num_loops = max(getattr(config, "num_loops", 1), 1)
    if num_loops > 1:
        config.num_hidden_layers = physical * num_loops
    return physical


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Fallback only; vLLM registers its own "vllm" attention interface.
    num_kv_groups = getattr(module, "num_key_value_groups", 1)
    key_states = repeat_kv(key, num_kv_groups)
    value_states = repeat_kv(value, num_kv_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights


class NanbeigeLoopAttention(nn.Module):
    """Parameter-free per-loop attention dispatch with a unique ``layer_idx``.
    Projections remain on the parent ``NanbeigeAttention`` so each physical
    layer keeps a single set of weights across loops.
    """

    def __init__(
        self,
        config: NanbeigeConfig,
        layer_idx: int,
        scaling: float,
        num_key_value_groups: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.scaling = scaling
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if ALL_ATTENTION_FUNCTIONS is None:
            raise ImportError(
                "ALL_ATTENTION_FUNCTIONS is required for _attn_implementation='vllm'. "
                "Please upgrade transformers."
            )
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attn_impl, _eager_attention_forward
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]
        return attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )


def _apply_loop_shared_kv(
    loop_share_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
    layer_idx: Optional[int],
    mhc_loop_idx: Optional[int],
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if loop_share_kv_cache is None or mhc_loop_idx is None:
        return key_states, value_states
    if layer_idx is None:
        raise ValueError("layer_idx must be set when loop_share_kv is enabled.")
    if mhc_loop_idx == 0:
        loop_share_kv_cache[layer_idx] = (key_states, value_states)
        return key_states, value_states
    if layer_idx not in loop_share_kv_cache:
        raise RuntimeError(f"loop_share_kv missing first-pass KV for layer {layer_idx}.")
    return loop_share_kv_cache[layer_idx]


def _reduce_query_to_kv_groups(query: torch.Tensor, num_kv_groups: int) -> torch.Tensor:
    num_query_heads = query.shape[1]
    if num_query_heads == num_kv_groups:
        return query
    if num_query_heads % num_kv_groups != 0:
        raise ValueError(
            f"query heads ({num_query_heads}) must be divisible by KV groups ({num_kv_groups})."
        )
    return query.reshape(
        query.shape[0],
        num_kv_groups,
        num_query_heads // num_kv_groups,
        query.shape[2],
        query.shape[3],
    ).mean(dim=2)


def _depth_attention_mix_value(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    source_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    source_kv = list(source_kv)
    if not source_kv:
        return current_value

    num_kv_groups = current_key.shape[1]
    query_for_kv = _reduce_query_to_kv_groups(query, num_kv_groups)
    if query_for_kv.shape != current_key.shape:
        raise ValueError(
            f"query/K shape mismatch after GQA grouping: {query_for_kv.shape} vs "
            f"{current_key.shape}."
        )

    keys = [key for key, _ in source_kv] + [current_key]
    values = [value for _, value in source_kv] + [current_value]
    for key in keys:
        if key.shape != current_key.shape:
            raise ValueError(f"source key shape {key.shape} does not match {current_key.shape}.")
    for value in values:
        if value.shape != current_value.shape:
            raise ValueError(
                f"source value shape {value.shape} does not match {current_value.shape}."
            )

    key_stack = torch.stack(keys, dim=0)
    value_stack = torch.stack(values, dim=0)
    logits = (query_for_kv.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    depth_probs = torch.softmax(logits * softmax_scale, dim=0).to(value_stack.dtype)
    return (depth_probs.unsqueeze(-1) * value_stack).sum(dim=0).to(current_value.dtype)


def _apply_depth_attention(
    config: NanbeigeConfig,
    layer_idx: Optional[int],
    depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if depth_attention_kv_cache is None:
        return key_states, value_states
    if layer_idx is None:
        raise ValueError("layer_idx must be set when enable_depth_attention=True.")

    source_kv = [(key, value) for _, key, value in depth_attention_kv_cache]
    value_states = _depth_attention_mix_value(
        query_states,
        key_states,
        value_states,
        source_kv,
        softmax_scale=softmax_scale,
    )
    if layer_idx % config.depth_attention_stride == 0:
        depth_attention_kv_cache.append((layer_idx, key_states, value_states))
    return key_states, value_states


def _apply_depth_attention_then_update_cache(
    config: NanbeigeConfig,
    layer_idx: Optional[int],
    depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]],
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    past_key_value: Optional[Cache],
    loop_idx: int,
    loop_cache_layer_idx: Optional[int],
    cache_kwargs: Dict[str, Any],
    skip_cache_update: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states, value_states = _apply_depth_attention(
        config,
        layer_idx,
        depth_attention_kv_cache,
        query_states,
        key_states,
        value_states,
        softmax_scale=softmax_scale,
    )
    if past_key_value is not None and not skip_cache_update:
        cache_layer_idx = _get_loop_cache_layer_idx(
            layer_idx, loop_idx, _get_num_physical_layers(config), loop_cache_layer_idx
        )
        key_states, value_states = past_key_value.update(
            key_states, value_states, cache_layer_idx, cache_kwargs
        )
    return key_states, value_states


def _get_double_loop_split_layer_order(
    num_hidden_layers: int, loop_middle_layers: Optional[int] = None
) -> List[int]:
    return [
        layer_idx
        for layer_idx, _ in _get_double_loop_split_layer_order_with_mhc_loop_indices(
            num_hidden_layers, loop_middle_layers
        )
    ]


def _get_double_loop_split_layer_order_with_mhc_loop_indices(
    num_hidden_layers: int, loop_middle_layers: Optional[int] = None
) -> List[Tuple[int, Optional[int]]]:
    if num_hidden_layers <= 0:
        raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
    if loop_middle_layers is None:
        if num_hidden_layers % 2 != 0:
            raise ValueError(
                "enable_double_loop_split requires num_hidden_layers to be divisible by 2 "
                "when loop_middle_layers is not set."
            )
        loop_middle_layers = num_hidden_layers // 2
    if loop_middle_layers <= 0:
        raise ValueError("loop_middle_layers must be greater than 0.")
    if num_hidden_layers % loop_middle_layers != 0:
        raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")

    first_unlooped_layers = (num_hidden_layers - loop_middle_layers) // 2
    middle_start = first_unlooped_layers
    middle_end = middle_start + loop_middle_layers
    middle_repeats = (num_hidden_layers + loop_middle_layers) // loop_middle_layers
    return (
        [(idx, None) for idx in range(0, middle_start)]
        + [
            (idx, repeat_idx)
            for repeat_idx in range(middle_repeats)
            for idx in range(middle_start, middle_end)
        ]
        + [(idx, None) for idx in range(middle_end, num_hidden_layers)]
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class NanbeigeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        NanbeigeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(NanbeigeRMSNorm)


class SinkhornKnopp(torch.autograd.Function):
    @staticmethod
    def _normalize(matrix: torch.Tensor, iterations: int, eps: float = 1e-6) -> torch.Tensor:
        for _ in range(iterations):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp(min=eps)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp(min=eps)
        return matrix

    @staticmethod
    def forward(ctx, logits: torch.Tensor, iterations: int):
        base = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        result = SinkhornKnopp._normalize(base, iterations)
        ctx.save_for_backward(base)
        ctx.iterations = iterations
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (base,) = ctx.saved_tensors
        with torch.enable_grad():
            base_input = base.detach().requires_grad_(True)
            current = SinkhornKnopp._normalize(base_input, ctx.iterations)
            (grad_base,) = torch.autograd.grad(
                outputs=current,
                inputs=base_input,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
            )
        return grad_base * base, None


class NanbeigeNgramLayerFusion(nn.Module):
    def __init__(self, config: NanbeigeConfig):
        super().__init__()
        self.fusion_size = config.ngram_layer_downproject_size or config.hidden_size
        if config.ngram_layer_downproject_size is None:
            self.hidden_down_proj = None
            self.output_proj = None
        else:
            self.hidden_down_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)
            self.output_proj = nn.Linear(self.fusion_size, config.hidden_size, bias=False)
        self.hidden_norm = NanbeigeRMSNorm(self.fusion_size, eps=config.rms_norm_eps)
        self.ngram_norm = NanbeigeRMSNorm(self.fusion_size, eps=config.rms_norm_eps)
        self.key_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, self.fusion_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, ngram_embeddings: torch.Tensor) -> torch.Tensor:
        key = self.key_proj(ngram_embeddings)
        normed_key = self.ngram_norm(key)
        hidden_for_gate = hidden_states
        if self.hidden_down_proj is not None:
            hidden_for_gate = self.hidden_down_proj(hidden_states)
        normed_hidden = self.hidden_norm(hidden_for_gate)
        gate = (normed_hidden * normed_key).sum(dim=-1, keepdim=True) / math.sqrt(self.fusion_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid()
        fused = gate * self.value_proj(ngram_embeddings)
        if self.output_proj is not None:
            fused = self.output_proj(fused)
        return hidden_states + fused


class NanbeigeHyperConnectionModule(nn.Module):
    def __init__(
        self,
        config: NanbeigeConfig,
        layer_idx: int,
        module_name: str,
        num_residual_streams: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.module_name = module_name
        self.enable_mhc = config.enable_mhc
        self.enable_h_res_identity = config.enable_h_res_identity
        self.mhc_identity_nohresparam = getattr(config, "mhc_identity_nohresparam", False)
        self.num_residual_streams = (
            config.num_residual_streams if num_residual_streams is None else num_residual_streams
        )
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.norm_eps = 1e-6

        in_dim = self.num_residual_streams * self.hidden_size
        init_alpha = config.mhc_init_gating_factor
        self.alpha_pre = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_post = nn.Parameter(torch.full((1,), init_alpha))
        self.alpha_res = nn.Parameter(torch.full((1,), init_alpha))

        if self.enable_mhc:
            out_dim = (
                2 * self.num_residual_streams
                if self.mhc_identity_nohresparam
                else self.num_residual_streams * self.num_residual_streams
                + 2 * self.num_residual_streams
            )
            self.mapping_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            out_dim = self.num_residual_streams * self.num_residual_streams + 2 * self.num_residual_streams
            self.mapping_proj = nn.Linear(in_dim, out_dim, bias=True)
            self.bias = None

        self._build_static_mappings()
        self._init_dynamic_zero()
        self._disable_h_res_identity_unused_params()

    def _disable_h_res_identity_unused_params(self):
        if not self.enable_h_res_identity:
            return

        self.alpha_res.requires_grad_(False)

    def _build_static_mappings(self):
        n = self.num_residual_streams
        stream_index = self.layer_idx % n
        h_pre_static = torch.zeros(n)
        h_pre_static[stream_index] = 1.0
        h_post_static = torch.ones(n)
        h_res_static = torch.eye(n)
        self.register_buffer("h_pre_static", h_pre_static)
        self.register_buffer("h_post_static", h_post_static)
        self.register_buffer("h_res_static", h_res_static)

    def _init_dynamic_zero(self):
        nn.init.zeros_(self.mapping_proj.weight)
        n = self.num_residual_streams
        if self.enable_mhc:
            with torch.no_grad():
                pre_init = self.bias.new_full((n,), -20.0)
                pre_init[self.layer_idx % n] = 20.0
                self.bias[:n] = pre_init
                self.bias[n : 2 * n].zero_()
                if not self.mhc_identity_nohresparam:
                    h_res_init = self.bias.new_full((n, n), -20.0)
                    h_res_init[torch.arange(n), torch.arange(n)] = 20.0
                    self.bias[2 * n :] = h_res_init.reshape(-1)
        else:
            nn.init.zeros_(self.mapping_proj.bias)

    @staticmethod
    def input_expand(hidden_states: torch.Tensor, num_residual_streams: int) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        expanded = hidden_states.unsqueeze(2).expand(batch_size, seq_len, num_residual_streams, hidden_size)
        return expanded.contiguous().view(batch_size, seq_len, num_residual_streams * hidden_size)

    @staticmethod
    def output_contract(hidden_states: torch.Tensor, num_residual_streams: int) -> torch.Tensor:
        batch_size, seq_len, n_hidden_size = hidden_states.shape
        if n_hidden_size % num_residual_streams != 0:
            raise RuntimeError(
                f"HC output_contract shape mismatch: hidden={n_hidden_size}, streams={num_residual_streams}"
            )
        hidden_size = n_hidden_size // num_residual_streams
        streams = hidden_states.view(batch_size, seq_len, num_residual_streams, hidden_size)
        return streams.mean(dim=2)

    @staticmethod
    def convert_stream_count(
        hidden_states: torch.Tensor, hidden_size: int, target_num_residual_streams: int
    ) -> torch.Tensor:
        batch_size, seq_len, n_hidden_size = hidden_states.shape
        if n_hidden_size == hidden_size:
            return NanbeigeHyperConnectionModule.input_expand(hidden_states, target_num_residual_streams)
        if n_hidden_size % hidden_size != 0:
            raise RuntimeError(
                f"HC convert_stream_count shape mismatch: hidden={n_hidden_size}, base_hidden={hidden_size}"
            )
        current_num_residual_streams = n_hidden_size // hidden_size
        if current_num_residual_streams == target_num_residual_streams:
            return hidden_states

        streams = hidden_states.view(batch_size, seq_len, current_num_residual_streams, hidden_size)
        if target_num_residual_streams % current_num_residual_streams == 0:
            repeat = target_num_residual_streams // current_num_residual_streams
            streams = streams.repeat_interleave(repeat, dim=2)
            return streams.contiguous().view(
                batch_size, seq_len, target_num_residual_streams * hidden_size
            )
        if current_num_residual_streams % target_num_residual_streams == 0:
            group = current_num_residual_streams // target_num_residual_streams
            streams = streams.view(batch_size, seq_len, target_num_residual_streams, group, hidden_size)
            return streams.mean(dim=3).contiguous().view(
                batch_size, seq_len, target_num_residual_streams * hidden_size
            )

        contracted = NanbeigeHyperConnectionModule.output_contract(
            hidden_states, current_num_residual_streams
        )
        return NanbeigeHyperConnectionModule.input_expand(contracted, target_num_residual_streams)

    def _compute_mappings(self, hidden_states: torch.Tensor):
        n = self.num_residual_streams
        h_res_identity = self.h_res_static.view(1, 1, n, n).to(dtype=hidden_states.dtype)
        if self.enable_mhc:
            if self.enable_h_res_identity:
                proj_weight = (
                    self.mapping_proj.weight
                    if self.mhc_identity_nohresparam
                    else self.mapping_proj.weight[: 2 * n, :]
                )
                proj = F.linear(hidden_states, proj_weight)
                n_channels = hidden_states.shape[-1]
                r = hidden_states.norm(dim=-1, keepdim=True) / math.sqrt(n_channels)
                r = 1.0 / (r + self.norm_eps)
                bias = self.bias.to(dtype=hidden_states.dtype)
                alpha_pre = self.alpha_pre.to(dtype=hidden_states.dtype)
                alpha_post = self.alpha_post.to(dtype=hidden_states.dtype)
                h_pre_logits = r * proj[..., :n] * alpha_pre + bias[:n].view(1, 1, n)
                h_post_logits = r * proj[..., n : 2 * n] * alpha_post + bias[n : 2 * n].view(1, 1, n)
                h_pre = h_pre_logits.sigmoid()
                h_post = h_post_logits.sigmoid() * 2.0
            else:
                proj = self.mapping_proj(hidden_states)
                n_channels = hidden_states.shape[-1]
                r = hidden_states.norm(dim=-1, keepdim=True) / math.sqrt(n_channels)
                r = 1.0 / (r + self.norm_eps)
                alpha = torch.cat(
                    [self.alpha_pre.expand(n), self.alpha_post.expand(n), self.alpha_res.expand(n * n)], dim=0
                ).to(dtype=hidden_states.dtype)
                h = r * proj * alpha + self.bias.to(dtype=hidden_states.dtype).view(1, 1, -1)
                h_pre = h[..., :n].sigmoid()
                h_post = h[..., n : 2 * n].sigmoid() * 2.0
            if self.enable_h_res_identity:
                h_res = h_res_identity.expand(hidden_states.shape[0], hidden_states.shape[1], n, n)
            else:
                h_res_logits = h[..., 2 * n :].view(hidden_states.shape[0], hidden_states.shape[1], n, n)
                h_res = SinkhornKnopp.apply(h_res_logits, self.sinkhorn_iterations)
        else:
            normalized = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            logits = torch.tanh(self.mapping_proj(normalized))
            h_pre_logits = logits[..., :n]
            h_post_logits = logits[..., n : 2 * n]
            h_pre = h_pre_logits * self.alpha_pre + self.h_pre_static.view(1, 1, n).to(dtype=hidden_states.dtype)
            h_post = h_post_logits * self.alpha_post + self.h_post_static.view(1, 1, n).to(dtype=hidden_states.dtype)
            if self.enable_h_res_identity:
                h_res = h_res_identity.expand(hidden_states.shape[0], hidden_states.shape[1], n, n)
            else:
                h_res_logits = logits[..., 2 * n :].view(logits.shape[0], logits.shape[1], n, n)
                h_res = h_res_logits * self.alpha_res + h_res_identity
        return h_pre, h_post, h_res

    def forward(self, hidden_states: torch.Tensor):
        h_pre, h_post, h_res = self._compute_mappings(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        streams = hidden_states.view(batch_size, seq_len, self.num_residual_streams, self.hidden_size)
        aggregated = (streams * h_pre.unsqueeze(-1)).sum(dim=2)
        return aggregated, h_res, h_post

    def fuse_residual(self, h_res: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, output: torch.Tensor):
        batch_size, seq_len, _ = residual.shape
        if self.enable_h_res_identity:
            mixed_residual = residual
        else:
            residual_streams = residual.view(batch_size, seq_len, self.num_residual_streams, self.hidden_size)
            mixed_residual = torch.matmul(h_res, residual_streams).view(
                batch_size, seq_len, self.num_residual_streams * self.hidden_size
            )
        expanded_output = (h_post.unsqueeze(-1) * output.unsqueeze(2)).contiguous().view(
            batch_size, seq_len, self.num_residual_streams * self.hidden_size
        )
        return mixed_residual + expanded_output


class NgramCache(DynamicCache):
    """
    Extended DynamicCache for storing N-gram context alongside KV cache.
    """
    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        if config is not None and config.emb_neighbor_num is not None:
            self.max_context_len = config.emb_neighbor_num - 1
        else:
            self.max_context_len = 0

    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        """
        Update N-gram context with window management.
        Args:
            new_tokens: New tokens to append, shape (batch_size, seq_len)
        """
        if self.max_context_len == 0:
            return

        if self.ngram_context is None:
            self.ngram_context = new_tokens.clone()
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)

        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len:]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        """Reorder cache for beam search."""
        super().reorder_cache(beam_idx)

        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(0, beam_idx.to(self.ngram_context.device))

        return self


class NanbeigeNgramEmbedding(nn.Module):
    """
    Computes embeddings enriched with N-gram features without maintaining internal state.
    """
    def __init__(self, config, base_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings

        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.tp = config.emb_tp_num
        self.ngram_mod_force_prime = getattr(config, "ngram_mod_force_prime", False)
        self.ngram_fused_mode = getattr(config, "ngram_fused_mode", "average")
        self.ngram_hash_base = _ngram_hash_base(
            config.vocab_size, self.ngram_mod_force_prime
        )

        self._init_ngram_embeddings()
        self._vocab_mods_cache = None

        self.use_compressed_tokenizer = getattr(config, 'ngram_compressed_tokenizer', False)

    def _init_ngram_embeddings(self) -> None:
        """Initialize N-gram embedding and projection layers."""
        num_embedders = self.k * (self.n - 1)
        ngram_hidden_size = (
            self.config.ngram_embedding_hidden_size
            if self.config.ngram_embedding_hidden_size is not None
            else self.config.hidden_size
        )
        emb_dim = ngram_hidden_size // num_embedders

        embedders = []
        post_projs = []
        self._ngram_vocab_dims = _ngram_embedding_vocab_sizes(
            self.m, num_embedders, self.ngram_mod_force_prime
        )

        for vocab_size in self._ngram_vocab_dims:
            padded_vocab_size = ((vocab_size + self.tp - 1) // self.tp) * self.tp
            emb = nn.Embedding(padded_vocab_size, emb_dim, padding_idx=self.config.pad_token_id)
            proj = (
                nn.Linear(emb_dim, self.config.hidden_size, bias=False)
                if self.ngram_fused_mode == "average"
                else None
            )
            embedders.append(emb)
            if proj is not None:
                post_projs.append(proj)

        self.embedders = nn.ModuleList(embedders)
        if self.ngram_fused_mode == "concat":
            self.concat_proj = nn.Linear(emb_dim * num_embedders, self.config.hidden_size, bias=False)
            self.post_projs = nn.ModuleList()
        else:
            self.post_projs = nn.ModuleList(post_projs)

    def _shift_right_ignore_eos(
        self,
        tensor: torch.Tensor,
        n: int,
        eos_token_id: int = 2,
        eos_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shift tensor right by n positions, resetting at EOS tokens."""
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        if eos_mask is None and eos_token_id is None:
            eos_mask = torch.zeros_like(tensor, dtype=torch.bool)
        elif eos_mask is None:
            eos_mask = (tensor == eos_token_id)
        else:
            eos_mask = eos_mask.to(device=tensor.device, dtype=torch.bool)

        for i in range(batch_size):
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            prev_idx = 0

            for eos_idx in eos_positions:
                end_idx = eos_idx.item() + 1
                if end_idx - prev_idx > n:
                    result[i, prev_idx+n:end_idx] = tensor[i, prev_idx:end_idx-n]
                prev_idx = end_idx

            if prev_idx < seq_len and seq_len - prev_idx > n:
                result[i, prev_idx+n:seq_len] = tensor[i, prev_idx:seq_len-n]

        return result

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        """Precompute modular arithmetic values for vocabulary."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = self._ngram_vocab_dims[index]

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.ngram_hash_base) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: Dict[int, torch.Tensor],
        vocab_mods: List[int],
        ngram: int
    ) -> torch.Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash."""
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def _compress_input_ids(self, input_ids: torch.Tensor, lookup_table: torch.Tensor) -> torch.Tensor:
        """Compress input IDs using lookup table.
        Args:
            input_ids: Input token IDs tensor
            lookup_table: Lookup table for compression
        Returns:
            Compressed token IDs tensor
        """
        pos_mask = input_ids >= 0
        out = input_ids.clone()
        valid_ids = input_ids[pos_mask]
        out[pos_mask] = lookup_table[valid_ids]
        return out

    def compute_ngram_embeddings(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None,
        lookup_table: Optional[torch.Tensor] = None,
        average: bool = True,
    ) -> torch.Tensor:
        seq_len = input_ids.size(-1)

        if ngram_context is not None:
            context = torch.cat([ngram_context[..., -(self.n - 1):], input_ids], dim=-1)
        else:
            context = input_ids

        device = self.word_embeddings.weight.device
        if self.use_compressed_tokenizer and lookup_table is not None:
            compressed_context = self._compress_input_ids(context, lookup_table)
        else:
            compressed_context = context

        vocab_mods = self._precompute_vocab_mods()
        shifted_ids = {}
        eos_mask = None if self.config.eos_token_id is None else context == self.config.eos_token_id
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                compressed_context, i - 1, eos_token_id=self.config.eos_token_id, eos_mask=eos_mask
            )

        if self.ngram_fused_mode == "average":
            x = torch.zeros(
                input_ids.shape[0],
                seq_len,
                self.config.hidden_size,
                device=device,
                dtype=self.word_embeddings.weight.dtype,
            )
        else:
            x = None
        ngram_embedding_parts = []
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = self._ngram_vocab_dims[index]
                ngram_ids = self._get_ngram_ids(
                    compressed_context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                embedder_device = self.embedders[index].weight.device
                x_ngram = self.embedders[index](new_ids.to(embedder_device))
                if self.ngram_fused_mode == "concat":
                    ngram_embedding_parts.append(x_ngram.to(device))
                    continue
                proj_device = self.post_projs[index].weight.device
                x_proj = self.post_projs[index](x_ngram.to(proj_device))
                x = x + x_proj.to(x.device)

        if self.ngram_fused_mode == "concat":
            concat_device = self.concat_proj.weight.device
            x_concat = torch.cat(ngram_embedding_parts, dim=-1).to(concat_device)
            return self.concat_proj(x_concat).to(device)

        if average:
            x = x / (self.k * (self.n - 1))
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        ngram_context: Optional[torch.Tensor] = None,
        lookup_table: Optional[torch.Tensor] = None,
        return_ngram_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Stateless forward pass.
        Args:
            input_ids: Current input token IDs of shape (batch_size, seq_len)
            ngram_context: Optional historical context of shape (batch_size, context_len)
            lookup_table: Optional lookup table for compressed tokenizer
        Returns:
            Embedding tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.word_embeddings(input_ids.to(self.word_embeddings.weight.device)).clone()
        ngram_embeddings = None
        if return_ngram_embeddings or not self.config.skip_ngram_for_input:
            ngram_embeddings = self.compute_ngram_embeddings(
                input_ids,
                ngram_context=ngram_context,
                lookup_table=lookup_table,
                average=self.ngram_fused_mode == "average",
            )
            if not self.config.skip_ngram_for_input:
                if self.ngram_fused_mode == "concat":
                    x = x + ngram_embeddings
                else:
                    x = (x + ngram_embeddings * (self.k * (self.n - 1))) / (1 + self.k * (self.n - 1))
        if return_ngram_embeddings:
            return x, ngram_embeddings
        return x


class NanbeigeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NanbeigeLinearScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    """NanbeigeRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class NanbeigeDynamicNTKScalingRotaryEmbedding(NanbeigeRotaryEmbedding):
    """NanbeigeRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


class NanbeigeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NanbeigeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: NanbeigeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        if config.qk_layernorm:
            self.q_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = NanbeigeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None

        self._init_rope()
        # Only for vLLM: one dispatch module per loop, shared projection weights above.
        if _is_vllm_backend(config):
            self._init_loop_attns(layer_idx)

    def _init_loop_attns(self, layer_idx: Optional[int]) -> None:
        physical = _get_num_physical_layers(self.config)
        num_loops = max(getattr(self.config, "num_loops", 1), 1)
        base_idx = 0 if layer_idx is None else layer_idx
        self.loop_attns = nn.ModuleList(
            [
                NanbeigeLoopAttention(
                    self.config,
                    layer_idx=loop_idx * physical + base_idx,
                    scaling=self.scaling,
                    num_key_value_groups=self.num_key_value_groups,
                )
                for loop_idx in range(num_loops)
            ]
        )

    def _init_rope(self):
        rope_scaling = self.config.rope_scaling
        scaling_type = None
        if isinstance(rope_scaling, dict):
            scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if rope_scaling is None or scaling_type in (None, "default"):
            self.rotary_emb = NanbeigeRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_factor = rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = NanbeigeLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = NanbeigeDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        loop_idx = kwargs.pop("loop_idx", 0)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if self.q_layernorm is not None:
            query_states = self.q_layernorm(query_states)
        if self.k_layernorm is not None:
            key_states = self.k_layernorm(key_states)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # vLLM path: dispatch through loop_attns[loop_idx]; output is [tokens, hidden].
        if _is_vllm_backend(self.config):
            attn_output, attn_weights = self.loop_attns[loop_idx](
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                **kwargs,
            )
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None if not output_attentions else attn_weights, past_key_value

        use_loop_shared_kv = (
            loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        )
        skip_cache_update = use_loop_shared_kv and loop_share_kv_repeat_idx > 0
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, _get_num_physical_layers(self.config), loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NanbeigeFlashAttention2(NanbeigeAttention):
    """
    Nanbeige flash attention module. This module inherits from `NanbeigeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        loop_idx = kwargs.pop("loop_idx", 0)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.q_layernorm is not None:
            query_states = self.q_layernorm(query_states)
        if self.k_layernorm is not None:
            key_states = self.k_layernorm(key_states)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_loop_shared_kv = (
            loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        )
        skip_cache_update = use_loop_shared_kv and loop_share_kv_repeat_idx > 0
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, _get_num_physical_layers(self.config), loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (NanbeigeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in NanbeigeFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class NanbeigeSdpaAttention(NanbeigeAttention):
    """
    Nanbeige attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `NanbeigeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from NanbeigeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        loop_idx = kwargs.pop("loop_idx", 0)
        loop_cache_layer_idx = kwargs.pop("loop_cache_layer_idx", None)
        loop_share_kv_cache = kwargs.pop("loop_share_kv_cache", None)
        loop_share_kv_repeat_idx = kwargs.pop("loop_share_kv_repeat_idx", None)
        depth_attention_kv_cache = kwargs.pop("depth_attention_kv_cache", None)
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "NanbeigeModel is using NanbeigeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                loop_idx=loop_idx,
                loop_cache_layer_idx=loop_cache_layer_idx,
                loop_share_kv_cache=loop_share_kv_cache,
                loop_share_kv_repeat_idx=loop_share_kv_repeat_idx,
                depth_attention_kv_cache=depth_attention_kv_cache,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.q_layernorm is not None:
            query_states = self.q_layernorm(query_states)
        if self.k_layernorm is not None:
            key_states = self.k_layernorm(key_states)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        use_loop_shared_kv = (
            loop_share_kv_cache is not None and loop_share_kv_repeat_idx is not None
        )
        skip_cache_update = use_loop_shared_kv and loop_share_kv_repeat_idx > 0
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if depth_attention_kv_cache is None:
            if past_key_value is not None and not skip_cache_update:
                cache_layer_idx = _get_loop_cache_layer_idx(
                    self.layer_idx, loop_idx, _get_num_physical_layers(self.config), loop_cache_layer_idx
                )
                key_states, value_states = past_key_value.update(
                    key_states, value_states, cache_layer_idx, cache_kwargs
                )
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
        else:
            key_states, value_states = _apply_loop_shared_kv(
                loop_share_kv_cache,
                self.layer_idx,
                loop_share_kv_repeat_idx,
                key_states,
                value_states,
            )
            key_states, value_states = _apply_depth_attention_then_update_cache(
                self.config,
                self.layer_idx,
                depth_attention_kv_cache,
                query_states,
                key_states,
                value_states,
                past_key_value,
                loop_idx,
                loop_cache_layer_idx,
                cache_kwargs,
                skip_cache_update=skip_cache_update,
                softmax_scale=self.head_dim**-0.5,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


NANBEIGE_ATTENTION_CLASSES = {
    "eager": NanbeigeAttention,
    "flash_attention_2": NanbeigeFlashAttention2,
    "sdpa": NanbeigeSdpaAttention,
    # vLLM Transformers backend sets _attn_implementation="vllm".
    "vllm": NanbeigeAttention,
}


class NanbeigeDecoderLayer(nn.Module):
    def __init__(self, config: NanbeigeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.enable_hyper_connection = config.enable_hyper_connection
        self.layer_idx = layer_idx
        self._mhc_loop_middle_layer = (
            self._is_mhc_loop_middle_layer()
            if (
                getattr(config, "enable_double_loop_split", False)
                or getattr(config, "mhc_diff_for_loop", False)
                or getattr(config, "mhc_double_stream_position_for_loop", None) is not None
            )
            else False
        )
        self.num_residual_streams = self._get_layer_num_residual_streams()

        self.self_attn = NANBEIGE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = NanbeigeMLP(config)
        self.input_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.enable_hyper_connection:
            self.self_attn_hc = NanbeigeHyperConnectionModule(
                config,
                layer_idx=layer_idx,
                module_name="self_attention",
                num_residual_streams=self.num_residual_streams,
            )
            self.mlp_hc = NanbeigeHyperConnectionModule(
                config,
                layer_idx=layer_idx,
                module_name="mlp",
                num_residual_streams=self.num_residual_streams,
            )
            if getattr(config, "mhc_diff_for_loop", False) and self._mhc_loop_middle_layer:
                self.self_attn_mhc_loop_hcs = nn.ModuleList(
                    [
                        NanbeigeHyperConnectionModule(
                            config,
                            layer_idx=layer_idx,
                            module_name=f"self_attention_loop_{loop_idx}",
                            num_residual_streams=self.num_residual_streams,
                        )
                        for loop_idx in range(1, self._get_mhc_loop_count())
                    ]
                )
                self.mlp_mhc_loop_hcs = nn.ModuleList(
                    [
                        NanbeigeHyperConnectionModule(
                            config,
                            layer_idx=layer_idx,
                            module_name=f"mlp_loop_{loop_idx}",
                            num_residual_streams=self.num_residual_streams,
                        )
                        for loop_idx in range(1, self._get_mhc_loop_count())
                    ]
                )
            else:
                self.self_attn_mhc_loop_hcs = None
                self.mlp_mhc_loop_hcs = None
        else:
            self.self_attn_hc = None
            self.mlp_hc = None
            self.self_attn_mhc_loop_hcs = None
            self.mlp_mhc_loop_hcs = None

    def _get_mhc_loop_count(self) -> int:
        physical = _get_num_physical_layers(self.config)
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            if physical <= 0 or physical % 2 != 0:
                raise ValueError("mhc_diff_for_loop requires loop_middle_layers or even num_hidden_layers.")
            loop_middle_layers = physical // 2
        return (physical + loop_middle_layers) // loop_middle_layers

    def _get_mhc_loop_middle_bounds(self) -> Tuple[int, int]:
        physical = _get_num_physical_layers(self.config)
        loop_middle_layers = self.config.loop_middle_layers
        if loop_middle_layers is None:
            if physical <= 0 or physical % 2 != 0:
                raise ValueError("mhc_diff_for_loop requires loop_middle_layers or even num_hidden_layers.")
            loop_middle_layers = physical // 2
        first_unlooped_layers = (physical - loop_middle_layers) // 2
        return first_unlooped_layers, first_unlooped_layers + loop_middle_layers

    def _is_mhc_loop_middle_layer(self) -> bool:
        middle_start, middle_end = self._get_mhc_loop_middle_bounds()
        return middle_start <= self.layer_idx < middle_end

    def _get_layer_num_residual_streams(self) -> int:
        num_residual_streams = self.config.num_residual_streams
        double_stream_position = getattr(self.config, "mhc_double_stream_position_for_loop", None)
        if double_stream_position is None:
            return num_residual_streams
        is_middle_layer = self._mhc_loop_middle_layer
        if (double_stream_position == "mid" and is_middle_layer) or (
            double_stream_position == "edge" and not is_middle_layer
        ):
            return num_residual_streams * 2
        return num_residual_streams

    def get_num_residual_streams(self) -> int:
        return self.num_residual_streams

    def _get_self_attn_hc(
        self, mhc_loop_idx: Optional[int] = None
    ) -> Optional[NanbeigeHyperConnectionModule]:
        if (
            mhc_loop_idx is not None
            and self.self_attn_mhc_loop_hcs is not None
            and mhc_loop_idx > 0
        ):
            return self.self_attn_mhc_loop_hcs[mhc_loop_idx - 1]
        return self.self_attn_hc

    def _get_mlp_hc(
        self, mhc_loop_idx: Optional[int] = None
    ) -> Optional[NanbeigeHyperConnectionModule]:
        if mhc_loop_idx is not None and self.mlp_mhc_loop_hcs is not None and mhc_loop_idx > 0:
            return self.mlp_mhc_loop_hcs[mhc_loop_idx - 1]
        return self.mlp_hc

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        loop_idx: int = 0,
        loop_cache_layer_idx: Optional[int] = None,
        mhc_loop_idx: Optional[int] = None,
        loop_share_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        depth_attention_kv_cache: Optional[List[DepthAttentionCacheEntry]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        if self.enable_hyper_connection:
            self_attn_hc = self._get_self_attn_hc(mhc_loop_idx)
            hidden_states, h_res, h_post = self_attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            loop_idx=loop_idx,
            loop_cache_layer_idx=loop_cache_layer_idx,
            loop_share_kv_cache=loop_share_kv_cache,
            loop_share_kv_repeat_idx=mhc_loop_idx,
            depth_attention_kv_cache=depth_attention_kv_cache,
            **kwargs,
        )
        if self.enable_hyper_connection:
            hidden_states = self_attn_hc.fuse_residual(h_res, residual, h_post, hidden_states)
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if self.enable_hyper_connection:
            mlp_hc = self._get_mlp_hc(mhc_loop_idx)
            hidden_states, h_res, h_post = mlp_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_hyper_connection:
            hidden_states = mlp_hc.fuse_residual(h_res, residual, h_post, hidden_states)
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


NANBEIGE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`NanbeigeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    NANBEIGE_START_DOCSTRING,
)
class NanbeigePreTrainedModel(PreTrainedModel):
    config_class = NanbeigeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanbeigeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    # Required by vLLM Transformers modeling backend.
    _supports_attention_backend = True

    def _supports_default_dynamic_cache(self) -> bool:
        return self.config.num_loops == 1 and super()._supports_default_dynamic_cache()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


NANBEIGE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    NANBEIGE_START_DOCSTRING,
)
class NanbeigeModel(NanbeigePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`NanbeigeDecoderLayer`]
    Args:
        config: NanbeigeConfig
    """

    def __init__(self, config: NanbeigeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Initialize N-gram embeddings if configured
        if config.emb_neighbor_num is not None and config.emb_split_num is not None and config.ngram_vocab_size_ratio is not None:
            self.ngram_embeddings = NanbeigeNgramEmbedding(config, self.embed_tokens)

            # Register lookup_table buffer for compressed tokenizer
            # This will be loaded from checkpoint, initialized as identity mapping
            use_compressed_tokenizer = getattr(config, 'ngram_compressed_tokenizer', False)
            if use_compressed_tokenizer:
                lookup_table = torch.arange(config.vocab_size, dtype=torch.long)
                self.register_buffer('lookup_table', lookup_table, persistent=True)
            else:
                self.lookup_table = None
        else:
            self.ngram_embeddings = None
            self.lookup_table = None

        # Physical ModuleList always matches checkpoint (layers.0..L-1). Under vLLM,
        # expand config.num_hidden_layers only so KV/Attention slots = L * num_loops.
        if _is_vllm_backend(config):
            physical = _prepare_config_for_vllm(config)
        else:
            physical = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [NanbeigeDecoderLayer(config, layer_idx) for layer_idx in range(physical)]
        )
        self.ngram_layer_fusion = nn.ModuleDict(
            {
                str(layer_idx): NanbeigeNgramLayerFusion(config)
                for layer_idx in (
                    range(physical)
                    if getattr(config, "ngram_insert_all_layers", False)
                    else getattr(config, "insert_ngram_layer_idx", [])
                )
            }
        )
        self.norm = NanbeigeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_num_loops(self) -> int:
        if getattr(self.config, "enable_double_loop_split", False):
            return 1
        loop_weights = getattr(self.config, "loop_loss_weights", [])
        if loop_weights is not None and len(loop_weights) > 0:
            return len(loop_weights) + 1
        return getattr(self.config, "num_loops", 1)

    def _get_layer_order(self) -> List[int]:
        if getattr(self.config, "enable_double_loop_split", False):
            return _get_double_loop_split_layer_order(
                _get_num_physical_layers(self.config),
                getattr(self.config, "loop_middle_layers", None),
            )
        return list(range(len(self.layers)))

    def _get_layer_execution_order(self) -> List[Tuple[int, Optional[int]]]:
        if getattr(self.config, "enable_double_loop_split", False):
            return _get_double_loop_split_layer_order_with_mhc_loop_indices(
                _get_num_physical_layers(self.config),
                getattr(self.config, "loop_middle_layers", None),
            )
        return [(layer_idx, None) for layer_idx in range(len(self.layers))]

    def _get_layer_num_residual_streams(self, layer_idx: int) -> int:
        layer = self.layers[layer_idx]
        if hasattr(layer, "get_num_residual_streams"):
            return layer.get_num_residual_streams()
        return self.config.num_residual_streams

    def _convert_hyper_connection_streams(
        self, hidden_states: torch.Tensor, target_layer_idx: int
    ) -> torch.Tensor:
        return NanbeigeHyperConnectionModule.convert_stream_count(
            hidden_states,
            self.config.hidden_size,
            self._get_layer_num_residual_streams(target_layer_idx),
        )

    def _contract_hyper_connection_streams(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n_hidden_size = hidden_states.shape[-1]
        if n_hidden_size == self.config.hidden_size:
            return hidden_states
        if n_hidden_size % self.config.hidden_size != 0:
            raise RuntimeError(
                f"HC output_contract shape mismatch: hidden={n_hidden_size}, "
                f"base_hidden={self.config.hidden_size}"
            )
        return NanbeigeHyperConnectionModule.output_contract(
            hidden_states, n_hidden_size // self.config.hidden_size
        )

    def _get_cache_seq_length(self, past_key_values: Optional[Cache]) -> int:
        if past_key_values is None:
            return 0
        max_seq_length = 0
        for loop_idx in range(self._get_num_loops()):
            layer_idx = loop_idx * _get_num_physical_layers(self.config)
            max_seq_length = max(max_seq_length, past_key_values.get_seq_length(layer_idx))
        return max_seq_length

    @add_start_docstrings_to_model_forward(NANBEIGE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        num_loops = self._get_num_loops()
        double_loop_split = getattr(self.config, "enable_double_loop_split", False)

        # Handle cache initialization and conversion before embeddings
        return_legacy_cache = False
        if use_cache and past_key_values is None and self.ngram_embeddings is None and double_loop_split:
            past_key_values = DynamicCache()
        elif use_cache and self.ngram_embeddings is None and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            '''
            if self.ngram_embeddings is not None:
                past_key_values = NgramCache.from_legacy_cache(past_key_values)
            else:
            '''
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        elif use_cache and self.ngram_embeddings is not None and past_key_values is not None and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = NgramCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None and self.ngram_embeddings is not None:
            past_key_values = NgramCache(config=self.config)
        elif use_cache and past_key_values is None and (num_loops > 1 or double_loop_split):
            past_key_values = DynamicCache()
        if use_cache and isinstance(past_key_values, StaticCache) and (num_loops > 1 or double_loop_split):
            raise ValueError("StaticCache is not supported when loop-aware caching is enabled. Please use the default dynamic cache.")
        if use_cache and getattr(self.config, "enable_depth_attention", False):
            if getattr(self.config, "loop_share_kv", False):
                raise ValueError(
                    "enable_depth_attention with loop_share_kv does not support use_cache=True/generation."
                )
            if isinstance(past_key_values, StaticCache):
                raise ValueError(
                    "StaticCache is not supported with enable_depth_attention. Please use the default dynamic cache."
                )

        ngram_context = None
        if self.ngram_embeddings is not None and isinstance(past_key_values, NgramCache):
            ngram_context = past_key_values.ngram_context

        ngram_layer_embeddings = None
        if inputs_embeds is None:
            # Use N-gram embeddings if available and configured
            if self.ngram_embeddings is not None:
                if len(self.ngram_layer_fusion) > 0:
                    inputs_embeds, ngram_layer_embeddings = self.ngram_embeddings(
                        input_ids,
                        ngram_context=ngram_context,
                        lookup_table=self.lookup_table,
                        return_ngram_embeddings=True,
                    )
                else:
                    inputs_embeds = self.ngram_embeddings(
                        input_ids,
                        ngram_context=ngram_context,
                        lookup_table=self.lookup_table,
                    )
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        if (
            self.ngram_embeddings is not None
            and len(self.ngram_layer_fusion) > 0
            and ngram_layer_embeddings is None
        ):
            raise RuntimeError("N-gram layer fusion requires input_ids and ngram embeddings.")

        # Update N-gram context after computing embeddings
        if use_cache and isinstance(past_key_values, NgramCache) and input_ids is not None:
            past_key_values.update_ngram_context(input_ids)

        if cache_position is None:
            past_seen_tokens = self._get_cache_seq_length(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        last_loop_all_hidden_states = None
        last_loop_all_self_attns = None
        last_loop_next_decoder_cache = None
        layer_order = self._get_layer_execution_order()
        layer_lookup = list(self.layers)
        loop_share_kv_cache = {} if getattr(self.config, "loop_share_kv", False) else None
        if loop_share_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("loop_share_kv does not support gradient checkpointing during training.")
        depth_attention_kv_cache = [] if getattr(self.config, "enable_depth_attention", False) else None
        if depth_attention_kv_cache is not None and self.gradient_checkpointing and self.training:
            raise ValueError("enable_depth_attention does not support gradient checkpointing during training.")

        for loop_idx in range(num_loops):
            current_loop_all_hidden_states = () if output_hidden_states else None
            current_loop_all_self_attns = () if output_attentions else None
            current_loop_next_decoder_cache = None

            if self.config.enable_hyper_connection and len(self.layers) > 0:
                hidden_states = self._convert_hyper_connection_streams(hidden_states, 0)

            for execution_idx, (layer_idx, mhc_loop_idx) in enumerate(layer_order):
                decoder_layer = layer_lookup[layer_idx]
                if self.config.enable_hyper_connection:
                    hidden_states = self._convert_hyper_connection_streams(
                        hidden_states, layer_idx
                    )
                if output_hidden_states:
                    if self.config.enable_hyper_connection:
                        current_loop_all_hidden_states += (
                            self._contract_hyper_connection_streams(hidden_states),
                        )
                    else:
                        current_loop_all_hidden_states += (hidden_states,)

                fusion_key = str(layer_idx)
                fusion = self.ngram_layer_fusion[fusion_key] if fusion_key in self.ngram_layer_fusion else None
                if fusion is not None:
                    if ngram_layer_embeddings is None:
                        raise RuntimeError("N-gram layer fusion requires input_ids and ngram embeddings.")
                    if self.config.enable_hyper_connection:
                        contracted = self._contract_hyper_connection_streams(hidden_states)
                        contracted = fusion(contracted, ngram_layer_embeddings)
                        hidden_states = self._convert_hyper_connection_streams(
                            contracted, layer_idx
                        )
                    else:
                        hidden_states = fusion(hidden_states, ngram_layer_embeddings)

                if self.gradient_checkpointing and self.training:
                    cache_layer_idx = (
                        (layer_idx if loop_share_kv_cache is not None else execution_idx)
                        if double_loop_split
                        else None
                    )
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        loop_idx,
                        cache_layer_idx,
                        mhc_loop_idx,
                        loop_share_kv_cache,
                        depth_attention_kv_cache,
                    )
                else:
                    cache_layer_idx = (
                        (layer_idx if loop_share_kv_cache is not None else execution_idx)
                        if double_loop_split
                        else None
                    )
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        loop_idx=loop_idx,
                        loop_cache_layer_idx=cache_layer_idx,
                        mhc_loop_idx=mhc_loop_idx,
                        loop_share_kv_cache=loop_share_kv_cache,
                        depth_attention_kv_cache=depth_attention_kv_cache,
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    current_loop_next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    current_loop_all_self_attns += (layer_outputs[1],)

            if self.config.enable_hyper_connection:
                hidden_states = self._contract_hyper_connection_streams(hidden_states)
            if not getattr(self.config, "skip_loop_final_norm", False):
                hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                current_loop_all_hidden_states += (hidden_states,)

            last_loop_all_hidden_states = current_loop_all_hidden_states
            last_loop_all_self_attns = current_loop_all_self_attns
            last_loop_next_decoder_cache = current_loop_next_decoder_cache

        if getattr(self.config, "skip_loop_final_norm", False):
            hidden_states = self.norm(hidden_states)
            if output_hidden_states and last_loop_all_hidden_states is not None:
                last_loop_all_hidden_states = last_loop_all_hidden_states[:-1] + (hidden_states,)

        next_cache = last_loop_next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, last_loop_all_hidden_states, last_loop_all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=last_loop_all_hidden_states,
            attentions=last_loop_all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = self._get_cache_seq_length(past_key_values)
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if _ignore_causal_mask_sdpa(
                attention_mask,
                input_tensor=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask *= torch.arange(target_length, device=device) > torch.arange(
                    sequence_length, device=device
                ).reshape(-1, 1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class NanbeigeForCausalLM(NanbeigePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = NanbeigeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(NANBEIGE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, NanbeigeForCausalLM
        >>> model = NanbeigeForCausalLM.from_pretrained("meta-llama/Nanbeige-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Nanbeige-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(self, *args, **kwargs):
        """Override to ensure NgramCache is used when ngram embeddings are configured."""
        generation_config = kwargs.get("generation_config", args[1] if len(args) > 1 else None)
        use_cache = kwargs.get(
            "use_cache",
            getattr(generation_config, "use_cache", self.config.use_cache),
        )
        if not use_cache:
            return super().generate(*args, **kwargs)

        if getattr(self.config, "enable_depth_attention", False):
            if getattr(self.config, "loop_share_kv", False):
                raise ValueError("enable_depth_attention with loop_share_kv does not support generation.")
            cache_implementation = kwargs.get(
                "cache_implementation",
                getattr(generation_config, "cache_implementation", None),
            )
            if cache_implementation is not None:
                raise ValueError(
                    "enable_depth_attention generation only supports the default DynamicCache; "
                    "cache_implementation is not supported."
                )
            kwargs["use_cache"] = True
        if self.config.emb_neighbor_num is not None and self.config.emb_split_num is not None and self.config.ngram_vocab_size_ratio is not None:
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = NgramCache(config=self.config)
        elif self.config.num_loops > 1 or getattr(self.config, "enable_double_loop_split", False):
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = DynamicCache()
        elif getattr(self.config, "enable_depth_attention", False):
            if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
                kwargs["past_key_values"] = DynamicCache()
        return super().generate(*args, **kwargs)

    def _get_cache_seq_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, Cache):
            return past_key_values[0][0].shape[-2] if len(past_key_values) > 0 else 0
        if getattr(self.config, "enable_double_loop_split", False):
            return past_key_values.get_seq_length(0)
        max_seq_length = 0
        loop_weights = getattr(self.config, "loop_loss_weights", [])
        num_loops = len(loop_weights) + 1 if loop_weights is not None and len(loop_weights) > 0 else self.config.num_loops
        for loop_idx in range(num_loops):
            layer_idx = loop_idx * _get_num_physical_layers(self.config)
            max_seq_length = max(max_seq_length, past_key_values.get_seq_length(layer_idx))
        return max_seq_length

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else self._get_cache_seq_length(past_key_values)
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past