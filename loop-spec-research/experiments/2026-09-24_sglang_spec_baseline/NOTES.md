# Loop 模型 × Spec Decode 实验记录

目录: `/ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/`（脚本在 `scripts/`，server 日志 `logs/`，结果 `results/all.jsonl`）
> 注：实验时目录是 `/ssd1/k3/yichuan/venvs/specbench/`，结束后整体搬到这里并改好了脚本里的路径。
代码: `/ssd1/k3/yichuan/sglang` @ `a164c6d9f1`（main），venv `/ssd1/k3/yichuan/venvs/sglang-main`
硬件: B300 SXM6 275GB；**只用 GPU 6、7**（0–5 是别人的）

## 时间线

### 2026-09-24 01:30–01:46 环境搭建
- `uv venv -p 3.12` + `uv pip install -e python`（cu130 index），得到 torch 2.13.0+cu130、flashinfer 0.6.18、sglang-kernel 0.4.7。
- 坑 1：Rust 扩展需要 `cargo`，它在 `~/.cargo/bin` 但不在 PATH → 加 PATH 后重装。
- 坑 2：shell 里 `CUDA_HOME=/usr/local/cuda` 不存在 → DeepGEMM JIT 断言失败。改为 `CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13`。
- 坑 3：flashinfer JIT 报 "CUDA compiler and CUDA toolkit headers are incompatible"（pip 装的 nvcc 13.4 vs runtime 头 13.0）。先装了 `flashinfer-jit-cache==0.6.18+cu130` + `flashinfer-cubin==0.6.18`（Dockerfile 同款）绕开。
- `examples/runtime/engine/offline_batch_inference.py --model Qwen/Qwen3-0.6B` 跑通。

### 02:20–02:35 第一轮（误用了 GPU 0–3，后来发现只有 6/7 是空的，已停掉并迁移）
Nanbeige4.2-3B（22 层 × `num_loops=2`），gsm8k 前 N 题，chat 接口，max_tokens 1024。

| 配置 | conc | temp | throughput tok/s | per-req tok/s | 加速 |
|---|---|---|---|---|---|
| baseline | 1 | 0 | 288.9 | 288.8 | 1.00× |
| DSpark (block 7) | 1 | 0 | 701.9 | 718.3 | **2.43×** |
| NGRAM | 1 | 0 | 275.2 | 276.9 | 0.95× |
| baseline | 16 | 0 | 3710 | 262 | 1.00× |
| DSpark | 16 | 0 | 5469 | 424 | 1.47× |
| baseline | 64 | 0 | 11212 | 201 | 1.00× |
| DSpark | 64 | 0 | 13061 | 236 | 1.16× |
| baseline | 1 | 0.6 | 278.5 | 278.4 | 1.00× |
| DSpark | 1 | 0.6 | 429.6 | 429.5 | 1.54× |

- DSpark 平均 accept len（server 日志）≈ 4.24（bs=1 greedy），NGRAM ≈ 1.79。
- NGRAM 的 conc 16/64 数据**作废**：NGRAM server 在 02:23:59 崩了（JIT 编译 `speculative_sampling` 时又是 CUDA 头版本不匹配）。
- EAGLE3（社区 draft `NullSense/Nanbeige4.2-3B-EAGLE3`）起不来：它是 vLLM speculators 格式，config 没有 `model_type`。即使换成 sglang 格式，
  target 侧 `NanbeigeForCausalLM` 也没实现 `set_eagle3_layers_to_capture`（`attention_backend_setup.py:55` 会无条件调用）。

### 02:40 修环境
- 把 pip 里的 CUDA 工具链对齐到 13.0：`nvidia-cuda-nvcc/crt/cccl/nvvm==13.0.*`（原来被依赖拉到 13.4 / cccl 13.3）。
- 新 bench 脚本 `bench2.py` 走 native `/generate`，从 `meta_info.spec_verify_ct` 直接算 accept length（= 输出 token / verify 次数，含 bonus token），并对 gsm8k 算准确率。

### 02:45 起 第二轮：只用 GPU 6/7
- GPU 6：Nanbeige baseline（port 30200）；GPU 7：DSpark block 7（30201）→ 跑完后换成 NGRAM（30202）。
- `sweep.sh`：数据集 {gsm8k, humaneval, sharegpt} × temp {0, 0.6} × conc {1,4,16,64}；conc<8 用 32 条，否则 conc×4 条；max_new_tokens 1024。
- 小插曲：下载 IQuest-40B-Loop 时发现 `/`（含 `~/.cache`）只剩 21GB → 立刻停掉，删掉 280MB 残留，改为 `HF_HOME=/ssd1/k3/yichuan/hf`，
  模型放 `/ssd1/k3/yichuan/hf/IQuest-Coder-V1-40B-Loop-Instruct`（75GB）。
- 小插曲 2：`pkill -f <pattern>` 会匹配到自己的 shell 命令行把自己杀掉，之后改用 pid。

## Spec decode 是怎么接进 loop 模型的（代码走读）

核心思路：**把循环"展开"成 layers×loops 个逻辑层**，spec decode 基础设施基本不需要知道模型在循环。

1. **KV cache**：每个物理层的 `self_attn.attn` 是 `nn.ModuleList`，第 k 轮用 `layer_id = k*L + i` 的 RadixAttention
   （`nanbeige.py:155-168`，`iquest_loopcoder.py:189-206`）。`model_config.py:1392-1397` 把 `num_attention_layers` 设成 L×loops，
   KV pool 就按 L×loops 层分配。所有轮共享同一个 `out_cache_loc`（同一个 token slot），只是写在不同 layer 的 buffer 里。
2. **Verify（TARGET_VERIFY）**：target 的 verify forward 就是一次普通 forward，所有轮拿同一个 `forward_batch`，
   树/causal mask 对每一轮一视同仁。attention backend 没有任何 loop 特判。
3. **拒绝的 token**：v2 worker 不按层 free，而是下一轮覆盖同一 slot；EAGLE topk>1 压缩用 `move_kv_cache` 搬所有层
   （`memory_pool.py:3085-3110`），所以每一轮的 KV 始终一致。
4. **CUDA graph**：`layer_setup.py:111-113` 把 ModuleList 展开并按 layer_id 排序，graph runner 按全局 layer_id 索引，不依赖 num_hidden_layers。
5. **DFlash/DSpark 需要 target 中间层 hidden**：层号在**展开空间**里编号，`logical_id = loop_idx*L + i`（`nanbeige.py:331-340`）。
   - `spec_aux_hidden_state.py:161-163`：target 层数 = `num_hidden_layers * num_loops`。
   - Nanbeige4.2-3B-DSpark 的 `target_layer_ids = [1,10,21,32,40]`、`num_target_layers = 44` —— 正好是 22×2 展开后的编号，
     也就是 draft 同时吃第 1 轮（1,10,21）和第 2 轮（32→第 2 轮第 10 层，40→第 2 轮第 18 层）的特征。
   - Nanbeige 没有 `set_dspark_layers_to_capture`，DSpark 回落到 `set_dflash_layers_to_capture`（+1 偏移，"层输出"语义）。
   - 拿到的 aux hidden 经 logits processor 打包进 `logits_output.hidden_states`，DSpark/DFlash worker 按 target 的 cache loc 写进 draft 的 KV。
6. **各算法对 target 的要求**：

| 算法 | target 需要提供 | Nanbeige | IQuest LoopCoder |
|---|---|---|---|
| NGRAM | 无 | ✅ | ✅（能起） |
| STANDALONE（小 draft 模型） | 无 | ✅（但没有同 tokenizer 的小模型） | ✅（能起，同样没合适 draft） |
| EAGLE / NEXTN | `get_embed_and_head` | ✅ 接口有（没有现成 draft） | ❌ `eagle_worker_v2.py:428` AttributeError |
| EAGLE3 | + `set_eagle3_layers_to_capture` | ❌ `attention_backend_setup.py:55` AttributeError（`nanbeige.py:301` 注释却说支持） | ❌ |
| DFLASH / DSPARK | 捕获 setter + `get_input_embeddings` | ✅（官方 DSpark draft） | ❌ `attention_backend_setup.py:62` ValueError |
| MTP（NextN / frozen-KV MTP / multi-layer EAGLE） | 模型自带 MTP 头 + `get_embed_and_head` | ❌ 模型没有 MTP 头 | ❌ |

### 走读中发现的潜在 bug
- **IQuest KV pool 多分配一倍（已在代码确认）**：`IQuestLoopCoderForCausalLM` 顶层没有 `start_layer/end_layer` 属性，
  `_resolve_pp_layer_range` 回落到 `model_num_layers = max(80, 160) = 160`，`layer_setup.py:157-159` 又乘 `loop_num` → **320 层**，一半浪费。
  Nanbeige 有这两个属性（`nanbeige.py:474-479`）所以没事。（待实测验证）
- `spec_aux_hidden_state.py:162` 只读 `num_loops`，IQuest 用的是 `loop_num` —— 将来给 IQuest 加 DFlash 时层号会算错。
- IQuest 的 sliding window（loop>0 窗口 64）在 FlashInfer 后端上看起来**没生效**（wrapper 用 window=-1 plan）；FA3 会生效；Triton 可能直接崩。
  也就是说默认后端下 IQuest 输出可能和 HF 不一致（推断，未验证）。
- DSpark 给的 `target_layer_ids` 不做范围检查；若写 L×loops−1，+1 后永远抓不到。

## 时间线（续）

### 02:58 IQuest-40B-Loop 起在 GPU 6（TP1，`--mem-fraction-static 0.85 --trust-remote-code`，默认 attention backend）
- **实测确认 KV pool bug**：日志 `KV Cache is allocated ... #tokens: 125056, K size: 76.37 GB`。
  K 每层每 token = 8 kv_heads × 128 × 2B = 2 KiB → 76.37 GiB / (125056 × 2 KiB) = **320 层**，应该是 80×2 = 160 层。
  也就是 IQuest 在 sglang 里 KV 容量只有应有的一半（同样显存下本应约 250k token）。
- 生成 sanity check 正常。开始 `sweep.sh 30300 iq_base`。

### 03:05 发现 NGRAM 结果被"污染"
- sweep 里每个配置都复用同一批 prompt；NGRAM 的 n-gram corpus 跨请求共享、会记住之前生成的文本，
  所以同一 prompt 第二次跑时 accept len 虚高（gsm8k greedy：conc=1 首跑 1.85，conc=4 复跑 3.68）。
- 修法：`bench2.py` 每次跑之前 `POST /flush_cache`（`scheduler.flush_cache` → `NGRAMWorker.clear_cache_pool` → `ngram_corpus.reset()`，代码已确认）。
- 污染的数据挪到 `results/nb_ngram_contaminated.jsonl`，NGRAM 重跑。
  （baseline / DSpark 不受影响：DSpark 是神经 draft 无记忆；radix prefix cache 只影响 prefill，三者一致。之后所有 run 都带 flush。）

### 03:20–03:50 DSpark 在 temp=0.6 下 accept len 随并发"跳变"的调查
**现象**：sweep 里 temp=0.6、conc=64 的 DSpark accept len ≈ 4.1–4.8，几乎等于 greedy；而 conc≤16 只有 ≈ 2.7。
- 可复现（`nb_dspark7_recheck`）；server 日志自己的 accept len 也一样：running-req ≥ ~32 时 4.0–4.4 / accept rate 0.43–0.48，降到 ≤16 后 2.6–2.9 / 0.22–0.26。
- 细扫并发（`meta_probe.py`，gsm8k，temp 0.6）：

| conc | 1 | 8 | 16 | 24 | 32 | 40 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|
| accept len | 2.69 | 2.76 | 2.71 | 3.54 | 4.03 | 4.15 | 4.24 | 4.08 | 4.16 |

  greedy 对照：conc 1 → 3.73，16 → 4.22，64 → 4.24。
- **是不是偷偷退化成 greedy（有损）？不是：**
  1. 同 64 个 prompt 跑两次 temp 0.6：0/64 输出相同，也 0/64 等于 greedy 输出（`determinism_probe.py`）。
  2. 采样 token 在 target 下的平均 logprob（`logprob_probe.py`，256 题 × 512 token）：
     no-spec baseline conc8 −0.0881 / conc64 −0.0860；DSpark conc8 −0.0852 / conc64 −0.0851（SE≈0.002）——分布一致。
  → 两边都是无损的，高并发那条路径就是**接受率更高**。
- 排除项：不是 folded proposal（`SGLANG_DSPARK_FOLDED_PROPOSAL=0` 时模式完全一样：2.6 / 2.73 / 4.01 / 4.06）；
  也不是 CUDA graph 覆盖（draft/verify graph bs 抓到 48，而 32/40 已经是高值）。
- **结论/待查**：running-req 在 16~24 之间有一个开关，使 sampling 下 accept 从 ~2.7 跳到 ~4.1。没来得及定位是哪段代码。
  如果能让小 batch 也走这条路径，DSpark 在 temp 0.6 低并发下可能从 1.5× → 2×+。值得深挖。
- 顺带发现：开 spec decode 时 sglang 自动把 `max_running_requests` 降到 **48**（日志 "Max running requests is reset to 48"），
  所以 sweep 里 "conc=64" 的 DSpark 实际只跑 48 条并发，baseline 跑 64 条 —— 高并发对比对 DSpark 略不公平。

### 03:48–03:58 准确率校验（gsm8k 前 500 题，max_new_tokens 8192，conc 48；两个 server 挤在 GPU 7 上，所以速度数字不作数）

| temp | baseline acc | DSpark acc | DSpark accept len |
|---|---|---|---|
| 0 | 93.6% | 93.6% | 4.05 |
| 0.6 | 94.0% | 93.4% | 3.94 |

→ DSpark 不掉点（0.6pp 差异在 500 题的噪声内，SE≈1.1pp）。sweep 里 gsm8k 准确率只有 ~60–70% 是因为 max_tokens=1024 把思考截断了，不代表模型能力。

### 04:00 GPU 7 换成 IQuest + NGRAM（TP1），开始 `sweep.sh 30301 iq_ngram`
- NGRAM 默认参数：`num_draft_tokens=12`、BFS breadth 1–10、trie depth 18；同样被降到 `max_running_requests=48`。

### 04:10–04:50 GPU 6 调参变体（Nanbeige）
- **DSpark block 4 vs 7**：block 4 低并发更慢（gsm8k greedy conc1 2.00× vs 2.35×；humaneval 2.26× vs 2.85×），
  temp 0.6 中并发略好（gsm8k conc16 1.12× vs 0.98×）。总体 block 7（官方推荐）更好。
- **NGRAM `--speculative-num-draft-tokens 4`**（默认 12）：accept 1.3–1.8，conc1 仍 0.9×，conc64 吞吐仍卡在 ~3.8k tok/s（baseline 11k）。
  → 瓶颈不在 draft token 数，而是 NGRAM 每步的固定开销（CPU trie 匹配等）。对 3B 这种快模型 NGRAM 不划算。

### 04:55 自投机（self-speculation）可行性：用"只跑第 1 轮"当 draft
- 思路：loop 模型天然有个便宜 draft —— 只跑 loop 0 → norm → lm_head（成本≈半个 forward），而且 draft 的 KV 就是 target 的 loop-0 KV，
  不需要额外显存。类似 LayerSkip。
- 实验：两个 server，full（num_loops=2）和 `--json-model-override-args '{"num_loops": 1}'`；在 full 模型自己的 greedy 输出上
  teacher-forcing，比较 loop-1 exit 的 top-1 和 full 的 token（= greedy chain spec 的精确接受率）。`self_spec_probe.py`，每数据集 64 条 × 512 token。

| 数据集 | top-1 一致率 | 模拟 accept len (γ=1…7) |
|---|---|---|
| gsm8k | 4.4% | 1.04–1.05 |
| humaneval | 2.9% | 1.03 |
| sharegpt | 2.6% | 1.03 |

- loop-1 exit 直接生成是乱码（`' capital capital of the countries \n.\n'`）。Nanbeige config 里 `loop_loss_weights=[]`，中间轮没有被训练成可以出 token。
  → **Nanbeige 直接自投机不可行**；要做需要给中间轮训一个 exit head / 加中间轮 loss（LayerSkip 式）。

### 05:00–05:35 IQuest TP2（GPU 6+7 一起），baseline vs NGRAM，temp 0
- TP2 KV：309k tokens/卡（K 94.4GB/卡），同样按 320 层分配。
- TP2 bs=1 只有 44 tok/s（TP1 是 34），TP2 只带来 1.3×。

| 数据集 | conc | TP2 base tok/s | TP2 NGRAM tok/s | 加速 | accept |
|---|---|---|---|---|---|
| gsm8k | 1 | 44 | 65 | 1.45× | 2.01 |
| gsm8k | 4 | 152 | 212 | 1.39× | 2.05 |
| gsm8k | 16 | 491 | 446 | 0.91× | 1.92 |
| gsm8k | 64 | 1556 | 836 | 0.54× | 1.91 |
| humaneval | 1 | 42 | 72 | 1.69× | 2.26 |
| humaneval | 4 | 144 | 225 | 1.56× | 2.22 |
| humaneval | 16 | 477 | 616 | 1.29× | 2.31 |
| humaneval | 64 | 1074 | 889 | 0.83× | 2.18 |

- 所有 server 已停，GPU 6/7 已释放（05:35）。
