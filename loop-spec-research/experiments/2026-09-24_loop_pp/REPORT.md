# Loop 模型按"循环"切 Pipeline Parallel：能否比 TP 更好地用两张卡？（2026-09-24 傍晚，GPU 4–7，约 2 小时）

## 结论
- **不能（在 sglang 现有 PP 框架上）。** Nanbeige4.2-3B 按循环切的 PP2（GPU A 跑第 1 轮、GPU B 跑第 2 轮）
  在所有并发下都**比单卡还慢**；两卡最好的方案是 **DP2**（两个独立副本），其次 TP2。
- 瓶颈不是"按循环切"本身，而是 **sglang PP 框架的每步固定开销 ≈ 10 ms**：
  普通模型 Qwen3-8B 的原生 PP2 也一样（conc=1 时 9.7 ms/token vs TP2 3.3 ms）。
- "按循环切"在**正确性和显存**上是成立的：输出与原实现一致；每卡只存一轮的 KV，单个 PP2 实例 KV 容量 2.45M token，是单卡（1.22M）的 2 倍。

## 做了什么
### 1. 参考：普通模型上 sglang PP 有多好（Qwen3-8B，random 1024 in / 512 out，output tok/s）
| conc | TP2 | PP2 | PP2/TP2 |
|---|---|---|---|
| 1 | 297 (TPOT 3.3ms) | 103 (TPOT 9.7ms) | 0.35× |
| 16 | 3857 | 1169 | 0.30× |
| 64 | 11311 | 4305 | 0.38× |
| 256 | 16974 | 11630 | 0.69× |
（原计划的 Qwen DP2 / 单卡没跑：脚本里 `wait` 把后台 server 也等上了，卡住 30 分钟，赶时间跳过。）

### 2. 实现：Nanbeige 按展开深度切 PP（代码改动，见本分支 `python/sglang/srt/models/nanbeige.py`）
- 原来 `NanbeigeModel` 直接 `assert pp_size == 1`；IQuest 没 assert，但开 PP 会静默算错（每个 rank 把自己那段层连跑两遍）。
- 新结构：
  - `physical_layers`：22 个物理层（权重），每个 PP rank 只建/加载自己用得到的；checkpoint 名 `model.layers.i` 映射到 `model.physical_layers.i`。
  - `layers`：44 个**展开步**（`NanbeigeLoopStep`），每步引用共享的物理层、持有自己的 RadixAttention（layer_id = 展开步号）。
  - `start_layer/end_layer` 是展开步区间；pp=2 时 rank0 = 第 1 轮全部 22 步、rank1 = 第 2 轮全部 22 步（两卡都存整份权重，3B 只有 ~8GB）。
  - 轮间变换（加 residual + RMSNorm）挪到"进入下一轮的第一步之前"执行，所以跨 rank 边界也对。
  - `layer_setup._get_loop_num` 只再对 IQuest（`loop_num`）乘轮数，Nanbeige 已经上报展开区间。
- 对 IQuest 这种切法**不适用**：它第 2 轮要读第 1 轮的 KV（gated global attention），按轮切需要跨卡搬 KV。

### 3. 正确性（`scripts/check_equiv.py`，gsm8k 64 题 greedy 256 token + teacher-forced logprob，参照 = 原代码 PP1）
| 候选 | greedy 完全一致 | 最早分歧 | logprob 平均/最大差 |
|---|---|---|---|
| 新代码 PP1 | **64/64** | — | **0 / 0**（逐位一致） |
| 新代码 PP2 | 61/64 | 第 89 个 token | 0.00026 / 0.19（batch 形状不同导致的数值噪声，不是逻辑错误） |

### 4. 性能：Nanbeige4.2-3B，两卡的各种用法（output tok/s）
| conc | 单卡 | TP2 | **DP2** | 按循环 PP2 | PP2 + async depth 1 |
|---|---|---|---|---|---|
| 1 | 285 (3.5ms) | 277 (3.6ms) | 283 (3.5ms) | 106 (9.7ms) | 96 (10.5ms) |
| 16 | 3751 | 3503 | **4004** | 1213 | – |
| 64 | 8805 | 10377 | **12108** | 4344 | 3188 |
| 256 | 13176 | 14587 | **16032** | 11089 | 10682 |

- TP2 对 3B 模型几乎没用：bs=1 时延迟不降（3.5ms 是 launch/latency bound）。
- PP 解码是有 CUDA graph 的（日志 `cuda graph: True`），所以 10ms 地板在调度/同步上：sglang PP 把 running 请求切成 `pp_size` 个 microbatch 轮流过各 stage，
  每个 token 要走 stage0 → stage1 → 采样结果回传 stage0，每个 rank 的 scheduler 都有一轮 CPU 开销。`--pp-async-batch-depth 1` 反而更慢。

## 下一步（如果还想走"用多卡加速 loop 模型"这条路）
1. **先 profile sglang PP 的 10ms 地板**（nsys 看两个 rank 的 CPU 调度 / NCCL send-recv / D2H 同步），这是普通模型也有的问题，修了对所有 PP 都有用。
2. **绕开现有 PP 框架做"轮间流水"**：同一个 scheduler 里，GPU A 跑 batch k 的第 1 轮时 GPU B 跑 batch k−1 的第 2 轮，
   只传 hidden（3072×bs×2B，极小），采样只在 B 上做。理论上每步时间 ≈ max(一轮) 而不是两轮之和，吞吐接近 2×、KV 按轮天然切开。
   需要自己写调度，不是改一下模型就行。
3. 对小模型（3B）**DP 已经很强**，loop-PP 的真正价值场景是：模型/KV 放不下一张卡，或者循环次数多（Huginn 32 轮）——那时按轮切的 stage 数可以很多。

## 文件
- `scripts/`：`launch.sh`、`bench.sh`（bench_serving random 1024/512）、`check_equiv.py`、`ref_qwen.sh`（有上面说的 wait bug）
- `results/`：bench_serving 原始 jsonl、`equiv.jsonl`
- `logs/`：每个 server 日志、每档并发的 bench 输出
