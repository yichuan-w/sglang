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
"""Nanbeige model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class NanbeigeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NanbeigeModel`]. It is used to instantiate a Nanbeige model
    according to the specified arguments, defining the model architecture. 
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Nanbeige model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`NanbeigeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        head_dim (`int`, *optional*):
            Dimension of each attention head. If unset, defaults to `hidden_size // num_attention_heads`.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to use LayerNorm on query and key states before applying attention.
        emb_neighbor_num (`int`, *optional*):
            Maximum N-gram length for N-gram embeddings. This parameter determines the context window size for N-gram computation. Higher values capture
            longer-range lexical patterns but increase memory usage. If None, N-gram embeddings are disabled.
        emb_split_num (`int`, *optional*):
            Number of hash functions (or splits) to use for N-gram embeddings. Multiple hash functions help improve the quality of N-gram representations.
            Required if emb_neighbor_num is set.
        ngram_vocab_size_ratio (`float`, *optional*):
            Ratio multiplier for N-gram vocabulary size relative to the base vocabulary size. The N-gram vocabulary
            size is calculated as `vocab_size * ngram_vocab_size_ratio`. Required if emb_neighbor_num is set.
        ngram_mod_force_prime (`bool`, *optional*, defaults to `False`):
            Whether to use consecutive prime numbers greater than the N-gram base vocabulary size as hash modulo
            dimensions for N-gram subtables, and the first prime greater than vocab_size as the N-gram hash base.
        ngram_embedding_hidden_size (`int`, *optional*):
            Total hidden size used to split N-gram embedding table dimensions. If None, uses `hidden_size`.
        ngram_fused_mode (`str`, *optional*, defaults to `"average"`):
            How N-gram embeddings are fused into token embeddings. `"average"` preserves the existing per-table
            projector and averaging behavior. `"concat"` concatenates raw N-gram table embeddings, projects once to
            `hidden_size`, and adds the projected N-gram embedding to the token embedding.
        emb_tp_num (`int`, *optional*):
            Tensor parallel padding multiplier for N-gram embeddings. The N-gram embedding vocabulary size is padded
            to the nearest multiple of emb_tp_num. This ensures compatibility with tensor parallel training in Megatron.
            Required if emb_neighbor_num is set.
        ngram_compressed_tokenizer (`bool`, *optional*, defaults to `False`):
            Whether to use compressed tokenizer for N-gram computation. When enabled, the model's tokenizer is used to
            construct a compressed tokenizer similar to the one in engram_demo_v1.py, which normalizes and deduplicates
            tokens before computing N-gram hashes.
        skip_ngram_for_input (`bool`, *optional*, defaults to `False`):
            Whether to skip adding N-gram embeddings to the input embedding.
        insert_ngram_layer_idx (`List[int]`, *optional*):
            0-based decoder layer indices where averaged N-gram embeddings are fused before attention.
        ngram_insert_all_layers (`bool`, *optional*, defaults to `False`):
            Whether to fuse averaged N-gram embeddings before attention in every decoder layer.
        ngram_layer_downproject_size (`int`, *optional*):
            Optional hidden size for N-gram layer fusion projections. If None, fusion uses `hidden_size`.
        num_loops (`int`, *optional*, defaults to 1):
            Number of times the complete decoder-layer stack is executed with shared parameters. Increasing this value
            increases the model's effective depth and FLOPs without adding a separate set of decoder-layer weights.
            This value is ignored when `loop_loss_weights` is non-empty or `enable_double_loop_split=True`.
        loop_loss_weights (`List[float]`, *optional*):
            Weights associated with intermediate loop outputs during multi-loop training. When this list is non-empty,
            the model executes `len(loop_loss_weights) + 1` loops instead of using `num_loops`. For the standard loop
            layout, the weights must sum to at most 1.0. Defaults to an empty list.
        skip_loop_final_norm (`bool`, *optional*, defaults to `False`):
            Whether to skip the final RMS normalization between loops. If `True`, normalization is applied only after
            the last loop; if `False`, every loop output is normalized before it is passed to the next loop.
        enable_double_loop_split (`bool`, *optional*, defaults to `False`):
            Whether to enable LoopSplit. LoopSplit keeps the outer decoder layers unlooped and repeatedly executes a
            contiguous middle block, providing different effective depths for different parts of the network. When
            enabled, the execution order is controlled by `loop_middle_layers` rather than `num_loops`.
        loop_middle_layers (`int`, *optional*):
            Number of contiguous middle decoder layers repeatedly executed by LoopSplit. It must be a positive factor
            of `num_hidden_layers`. If omitted while LoopSplit is enabled, it defaults to half of
            `num_hidden_layers`, which therefore must be even.
        loop_share_kv (`bool`, *optional*, defaults to `False`):
            Whether repeated executions of a LoopSplit middle layer reuse the key and value states produced by that
            layer's first execution. Requires `enable_double_loop_split=True`.
        mhc_diff_for_loop (`bool`, *optional*, defaults to `False`):
            Whether each repeated execution of a LoopSplit middle layer uses separate mHC connection modules instead
            of sharing one set across repetitions. Requires both `enable_double_loop_split=True` and `enable_mhc=True`.
        mhc_double_stream_position_for_loop (`str`, *optional*):
            Selects where LoopSplit doubles the configured number of residual streams. Accepted values are `"mid"`,
            which doubles streams in the looped middle block, and `"edge"`, which doubles streams in the unlooped outer
            blocks. Requires `enable_double_loop_split=True`.
        enable_hyper_connection (`bool`, *optional*, defaults to `False`):
            Whether to replace the standard single residual path with multiple residual streams connected around each
            attention and MLP sublayer by learned hyper-connection modules.
        enable_mhc (`bool`, *optional*, defaults to `False`):
            Whether to use manifold-constrained hyper-connections (mHC), which constrain the learned residual-stream
            mixing matrices with Sinkhorn normalization. Requires `enable_hyper_connection=True`.
        enable_h_res_identity (`bool`, *optional*, defaults to `False`):
            Whether the residual-stream mixing matrix includes an explicit identity component. Requires
            `enable_hyper_connection=True`.
        mhc_identity_nohresparam (`bool`, *optional*, defaults to `False`):
            Whether the identity component is used without a separately learned residual mixing parameter. Requires
            both `enable_mhc=True` and `enable_h_res_identity=True`.
        num_residual_streams (`int`, *optional*, defaults to 4):
            Base number of residual streams used by hyper-connections. It must be at least 2 when
            `enable_hyper_connection=True`; LoopSplit may double it in the region selected by
            `mhc_double_stream_position_for_loop`.
        mhc_sinkhorn_iterations (`int`, *optional*, defaults to 20):
            Number of Sinkhorn normalization iterations used to constrain mHC residual-stream mixing matrices. It must
            be at least 1 when `enable_mhc=True`.
        mhc_init_gating_factor (`float`, *optional*, defaults to 0.01):
            Initial scale of the learned mHC gating terms that perturb the identity-like initialization of the
            hyper-connection matrices.
        enable_depth_attention (`bool`, *optional*, defaults to `False`):
            Whether to enable depth attention. At each decoder layer, the current query selects and mixes value states
            from cached anchor depths before normal token-level self-attention is applied.
        depth_attention_stride (`int`, *optional*):
            Layer interval at which key/value states are added as depth-attention anchors. It must be positive and
            defaults to `num_hidden_layers // 2` when depth attention is enabled.
        depth_attention_recent_window (`int`, *optional*, defaults to 0):
            Reserved size of a recent-depth window in addition to anchor depths. The current anchor-only implementation
            requires this value to be 0.
        depth_attention_static_anchor_once (`bool`, *optional*, defaults to `True`):
            Whether depth-attention anchors are collected once and reused across repeated LoopSplit executions. This
            must be `True` when depth attention and LoopSplit are enabled together.
    ```python
    >>> from transformers import NanbeigeModel, NanbeigeConfig
    >>> # Initializing a Nanbeige style configuration
    >>> configuration = NanbeigeConfig()
    >>> # Initializing a model from the Nanbeige style configuration
    >>> model = NanbeigeModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nanbeige"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Required for vLLM Transformers backend tensor parallel sharding.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        qk_layernorm=False,
        emb_neighbor_num=None,
        emb_split_num=None,
        ngram_vocab_size_ratio=None,
        ngram_mod_force_prime=False,
        ngram_embedding_hidden_size=None,
        ngram_fused_mode="average",
        emb_tp_num=None,
        ngram_compressed_tokenizer=False,
        skip_ngram_for_input=False,
        insert_ngram_layer_idx=None,
        ngram_insert_all_layers=False,
        ngram_layer_downproject_size=None,
        num_loops=1,
        loop_loss_weights=None,
        skip_loop_final_norm=False,
        enable_double_loop_split=False,
        loop_middle_layers=None,
        loop_share_kv=False,
        mhc_diff_for_loop=False,
        mhc_double_stream_position_for_loop=None,
        enable_hyper_connection=False,
        enable_mhc=False,
        enable_h_res_identity=False,
        mhc_identity_nohresparam=False,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=0.01,
        enable_depth_attention=False,
        depth_attention_stride=None,
        depth_attention_recent_window=0,
        depth_attention_static_anchor_once=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.qk_layernorm = qk_layernorm
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio
        self.ngram_mod_force_prime = ngram_mod_force_prime
        self.ngram_embedding_hidden_size = ngram_embedding_hidden_size
        self.ngram_fused_mode = ngram_fused_mode
        self.emb_tp_num = emb_tp_num
        self.ngram_compressed_tokenizer = ngram_compressed_tokenizer
        self.skip_ngram_for_input = skip_ngram_for_input
        self.insert_ngram_layer_idx = insert_ngram_layer_idx if insert_ngram_layer_idx is not None else []
        self.ngram_insert_all_layers = ngram_insert_all_layers
        self.ngram_layer_downproject_size = ngram_layer_downproject_size
        self.num_loops = num_loops
        self.loop_loss_weights = loop_loss_weights if loop_loss_weights is not None else []
        self.skip_loop_final_norm = skip_loop_final_norm
        self.enable_double_loop_split = enable_double_loop_split
        self.loop_middle_layers = loop_middle_layers
        self.loop_share_kv = loop_share_kv
        self.mhc_diff_for_loop = mhc_diff_for_loop
        self.mhc_double_stream_position_for_loop = mhc_double_stream_position_for_loop
        self.enable_hyper_connection = enable_hyper_connection
        self.enable_mhc = enable_mhc
        self.enable_h_res_identity = enable_h_res_identity
        self.mhc_identity_nohresparam = mhc_identity_nohresparam
        self.num_residual_streams = num_residual_streams
        self.mhc_sinkhorn_iterations = mhc_sinkhorn_iterations
        self.mhc_init_gating_factor = mhc_init_gating_factor
        self.enable_depth_attention = enable_depth_attention
        self.depth_attention_stride = depth_attention_stride
        self.depth_attention_recent_window = depth_attention_recent_window
        self.depth_attention_static_anchor_once = depth_attention_static_anchor_once
        self._hyper_connection_validation()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    def _hyper_connection_validation(self):
        if self.mhc_diff_for_loop and not self.enable_double_loop_split:
            raise ValueError("mhc_diff_for_loop requires enable_double_loop_split=True.")
        if self.mhc_diff_for_loop and not self.enable_mhc:
            raise ValueError("mhc_diff_for_loop requires enable_mhc=True.")
        if self.loop_share_kv and not self.enable_double_loop_split:
            raise ValueError("loop_share_kv requires enable_double_loop_split=True.")
        if self.enable_depth_attention:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_depth_attention requires num_hidden_layers to be greater than 0.")
            if self.depth_attention_stride is None:
                self.depth_attention_stride = self.num_hidden_layers // 2
            if self.depth_attention_stride <= 0:
                raise ValueError("depth_attention_stride must be greater than 0.")
            if self.depth_attention_recent_window < 0:
                raise ValueError("depth_attention_recent_window must be >= 0.")
            if self.depth_attention_recent_window != 0:
                raise ValueError("anchor-only Depth-Attention requires depth_attention_recent_window == 0.")
            if self.enable_double_loop_split and not self.depth_attention_static_anchor_once:
                raise ValueError(
                    "enable_depth_attention with double-loop split requires "
                    "depth_attention_static_anchor_once=True."
                )
        if self.mhc_double_stream_position_for_loop is not None:
            if self.mhc_double_stream_position_for_loop not in ("mid", "edge"):
                raise ValueError("mhc_double_stream_position_for_loop must be one of: mid, edge.")
            if not self.enable_double_loop_split:
                raise ValueError(
                    "mhc_double_stream_position_for_loop requires enable_double_loop_split=True."
                )
        if (self.enable_mhc or self.enable_h_res_identity) and not self.enable_hyper_connection:
            raise ValueError(
                "enable_mhc/enable_h_res_identity require enable_hyper_connection=True."
            )
        if self.mhc_identity_nohresparam:
            if not self.enable_h_res_identity:
                raise ValueError("mhc_identity_nohresparam requires enable_h_res_identity=True.")
            if not self.enable_mhc:
                raise ValueError("mhc_identity_nohresparam requires enable_mhc=True.")
        if self.enable_hyper_connection and self.num_residual_streams < 2:
            raise ValueError("num_residual_streams must be >= 2 when enable_hyper_connection=True.")
        if self.enable_mhc and self.mhc_sinkhorn_iterations < 1:
            raise ValueError("mhc_sinkhorn_iterations must be >= 1 when enable_mhc=True.")
        if self.ngram_insert_all_layers and self.insert_ngram_layer_idx:
            raise ValueError("ngram_insert_all_layers cannot be used with insert_ngram_layer_idx.")
        if self.ngram_layer_downproject_size is not None and self.ngram_layer_downproject_size <= 0:
            raise ValueError("ngram_layer_downproject_size must be greater than 0 when set.")
        if self.ngram_embedding_hidden_size is not None and self.ngram_embedding_hidden_size <= 0:
            raise ValueError("ngram_embedding_hidden_size must be greater than 0 when set.")
        if self.ngram_fused_mode not in ("average", "concat"):
            raise ValueError("ngram_fused_mode must be one of: average, concat.")
        if (
            not self.enable_double_loop_split
            and self.loop_loss_weights is not None
            and sum(self.loop_loss_weights) > 1.0
        ):
            raise ValueError("sum(loop_loss_weights) must be <= 1.0.")
        if self.enable_double_loop_split and self.loop_middle_layers is None:
            if self.num_hidden_layers <= 0:
                raise ValueError("enable_double_loop_split requires num_hidden_layers to be greater than 0.")
            if self.num_hidden_layers % 2 != 0:
                raise ValueError(
                    "enable_double_loop_split requires num_hidden_layers to be divisible by 2 "
                    "when loop_middle_layers is not set."
                )
            self.loop_middle_layers = self.num_hidden_layers // 2
        if self.loop_middle_layers is not None:
            if self.num_hidden_layers <= 0:
                raise ValueError("loop_middle_layers requires num_hidden_layers to be greater than 0.")
            if self.loop_middle_layers <= 0:
                raise ValueError("loop_middle_layers must be greater than 0.")
            if self.num_hidden_layers % self.loop_middle_layers != 0:
                raise ValueError("loop_middle_layers must be a factor of num_hidden_layers.")
