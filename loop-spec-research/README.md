# Loop Transformer Serving Acceleration via Speculative Decoding

Research workspace: using self-speculative decoding to speed up serving of looped / recurrent-depth transformers in SGLang.

```
loop-spec-research/
├── README.md                     # this index
├── related_work/                 # loop-transformer spec-decode papers
│   ├── 2609.17184.pdf            # LoopSpec
│   └── 2609.23033v1.pdf          # WaveFront Decoding (WFD)
└── models/                       # target open-source loop models
    ├── Nanbeige4.2-3B/{hf/, paper/}
    ├── Huginn-0125/{hf/, paper/}
    └── Ouro-1.4B/{hf/, paper/}
```

Each `hf/` folder holds the HF README (model card), config, custom modeling code, and safetensors index. Weights are **not** downloaded.

---

## 1. Related work: spec decode for looped transformers

### LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers
- arXiv: [2609.17184](https://arxiv.org/abs/2609.17184) (2026-09-15) → `related_work/2609.17184.pdf`
- Authors: SangLyul Cho, Langqing Cui, Sehoon Kim, Dongsu Han, Insu Han
- Main ideas:
  - Training-free self-spec decoding: **draft tokens come from early recurrent states**, so no separate draft model is needed.
  - **Pipelining:** drafting future tokens overlaps with verifying the current token.
  - **Selective second proposal** from a deeper recurrent depth raises draft accuracy. Decoding stays lossless under both greedy and sampling.
  - The paper derives the **optimal proposal depths in closed form**, and the prediction matches measurement.
  - Reported speedup: up to **6.83×** across several looped transformers on reasoning and coding benchmarks.
- **Code:** https://github.com/kaist-flexml-lab/loopspec, cloned to `related_work/loopspec-code/` (commit 41aa710). Project page: https://langq1225.github.io/loopspec/
  - Built as a plugin on pip `sglang==0.5.14` (package `sglang_recurrent/`, ~7.4k lines of Python). It uses its own scheduler process, branch-wise KV management, and CUDA graphs.
  - **Handles one request at a time (batch size 1):** `_run_scheduler_loop` runs requests sequentially, and there is no continuous batching.
  - Supported models: Ouro-1.4B/2.6B (plus -Thinking), and Raven-Llama-3.2/OLMo-2/TinyLlama (`smcleish/*`, R=32). **Nanbeige and huginn-0125 are not supported.**

### WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models
- arXiv: [2609.23033](https://arxiv.org/abs/2609.23033) (2026-09-19) → `related_work/2609.23033v1.pdf`
- Authors: Hyeongju Ha, Jae-Joon Kim
- Main ideas:
  - Training-free. Weight sharing lets **tokens at different positions and different recurrence depths run in one batched recurrent-block call**.
  - Mixed-depth states form a **diagonal wavefront**: new positions are drafted at shallow depth while earlier positions advance to full-depth verification. Drafting and verification share the same calls instead of running in separate phases.
  - Rejected drafts are corrected with the full-depth predictions.
  - **Code:** no public code. The paper gives no link, and a GitHub search turned up nothing (as of 2026-09-24). Its evaluation is limited to greedy decoding, a single GPU, and Ouro-2.6B and huginn-0125.
  - Results on Spec-Bench (6 categories): **2.42×** on Ouro-2.6B and **3.54×** on Huginn-3.5B. Adding **cross-recurrence KV sharing** raises Huginn to **4.81×**.

---

## 2. Target models

| Model | HF repo | Monthly downloads* | Likes | Params (safetensors) | Loop structure | License | Paper |
|---|---|---|---|---|---|---|---|
| Nanbeige4.2-3B | [Nanbeige/Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) | 40,311 | 801 | ~4.17B (3B non-emb) | 22 layers × `num_loops=2` | Apache-2.0 | [2607.22083](https://arxiv.org/abs/2607.22083) |
| Huginn-0125 | [tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) | 24,795 | 312 | ~3.9B (fp32 ckpt) | prelude 2 + recurrent 4 × `num_steps` (default mean 32) + coda 2 | Apache-2.0 | [2502.05171](https://arxiv.org/abs/2502.05171) |
| Ouro-1.4B | [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | 24,514 | 161 | ~1.43B | 24 layers × `total_ut_steps=4` + early exit | Apache-2.0 | [2510.25741](https://arxiv.org/abs/2510.25741) |

\*HF API `downloads` field (last 30 days), pulled 2026-09-24.

### Nanbeige4.2-3B
- Paper: *Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model* (Nanbeige Lab, 2026-07-24)
  - `paper/2607.22083.pdf`: arXiv version
  - `paper/Nanbeige42_report.pdf`: technical report from the HF repo. It is a different file from the arXiv PDF.
- Architecture `NanbeigeForCausalLM`: hidden 3072, 48 heads / 8 KV heads (GQA), head_dim 128, FFN 10752, vocab 166144, max_pos 262144, rope_theta 7e7.
- Looping: the same 22-layer stack runs twice (`num_loops=2`), giving 44 logical layers. `skip_loop_final_norm=false` means an RMSNorm is applied after every loop.
- Pretrained from scratch on 28T tokens. Aimed at general agents, tool calling, and code agents (SWE-bench, Terminal-Bench, and similar).
- The HF `modeling_nanbeige.py` also contains LoopSplit, mHC with depth attention, and concatenated n-gram embeddings, which are meant for Nanbeige4.5.
- **SGLang already supports it:** `python/sglang/srt/models/nanbeige.py`. Each loop has its own KV cache, with logical layer id = `loop_idx * 22 + i`.

### Huginn-0125
- Paper: *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Geiping et al., 2025-02-07) → `paper/2502.05171.pdf`
- Architecture `RavenForCausalLM`, which the paper calls RecurrentGPT:
  - prelude (2 layers) → recurrent block (4 layers, `SandwichBlock`) iterated `num_steps` times → coda (2 layers).
  - n_embd 5280, 55 heads (MHA), head_dim 96, vocab 65536, context 4096.
- `num_steps` is set at inference time. It was trained with mean recurrence 32; gains continue up to about 64. The model supports **adaptive compute** (`generate_with_adaptive_compute`) and KV-cache sharing across recurrences.
- 800B training tokens. This is the main testbed for studying recurrence count against compute. **Not supported in SGLang.**

### Ouro-1.4B
- Paper: *Scaling Latent Reasoning via Looped Language Models* (Zhu et al., ByteDance Seed, incl. Y. Bengio, 2025-10-29) → `paper/2510.25741.pdf`
- Architecture `OuroForCausalLM`: 24 layers, hidden 2048, 16 heads (MHA), head_dim 128, FFN 5632, vocab 49152, max_pos 65536.
- The whole stack loops `total_ut_steps=4` times. An entropy-regularized objective learns when to stop. `early_exit_threshold` controls adaptive exit, which vLLM does not support: it always runs the full step count.
- Trained on 7.7T tokens. The 1.4B and 2.6B models match SOTA LLMs of up to 12B. The larger Ouro-2.6B is the model WFD evaluates on.
- **Not supported in SGLang.** vLLM has support.

---

## 3. Notes for the research direction
- Both related-work papers use **training-free self-speculation from intermediate recurrence outputs**. They differ in scheduling:
  - LoopSpec: pipelined draft/verify with a closed-form optimal draft depth.
  - WFD: a diagonal wavefront that co-batches drafting and verification.
- Serving-system gaps a real SGLang deployment would hit:
  - batching mixed-depth states,
  - per-loop KV cache layout and cross-recurrence KV sharing,
  - CUDA graphs with dynamic recurrence,
  - how this fits into the existing spec-decode framework (`python/sglang/srt/speculative/`).
- Loop counts differ widely across the models: Nanbeige has 2, Ouro has 4, Huginn has 32 or more. The spec-decode gain likely grows with loop count. Nanbeige's 2 loops leave the least room, but it has the highest downloads and already runs in SGLang.

---

## 4. Experiments

### [`experiments/2026-09-24_sglang_spec_baseline/`](experiments/2026-09-24_sglang_spec_baseline/REPORT.md) — upstream SGLang spec decode on loop models (baseline)
- **Start with `REPORT.md`** (conclusions). `NOTES.md` is the full timeline, including every pitfall.
- Nanbeige4.2-3B + the official DSpark draft:
  - Greedy at conc 1 is **2.35–2.85×**, and the speedup shrinks as concurrency grows.
  - gsm8k accuracy is unchanged.
- IQuest-40B-Loop: only NGRAM works, reaching **1.4–1.9×** at conc 1.
- **Self-speculation feasibility on Nanbeige:** the loop-1 exit (norm → lm_head) agrees with the full model's top-1 only **3–4%** of the time.
  - Draft that exits after the first loop is useless on this model, since `loop_loss_weights=[]` means the intermediate loop was never trained to emit tokens.
  - This is consistent with section 3 above: Nanbeige leaves the least room.
  - The self-spec directions in LoopSpec/WFD are a better fit for Ouro (entropy-regularized early exit) and Huginn (32+ recurrences).
- Found an SGLang bug: IQuest's KV pool is allocated at twice the needed size (320 layers instead of 160).
