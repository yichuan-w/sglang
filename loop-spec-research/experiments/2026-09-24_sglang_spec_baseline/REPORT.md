# Loop 模型 × Spec Decode：结论（2026-09-24 夜间实验）

详细过程、每一步的命令和原始数据：`NOTES.md`（同目录）。目录结构：
- `scripts/`：`launch2.sh`（起 server）、`sweep.sh`（全量扫描）、`bench2.py`（压测 + accept len + gsm8k 准确率）、`summarize.py`（出表）、`self_spec_probe.py`（loop-1 自投机）、`logprob_probe.py` / `determinism_probe.py` / `meta_probe.py`（DSpark 采样调查）
- `results/all.jsonl`：所有压测结果（每行一个配置）；`results/self_spec.jsonl`；`results/nb_ngram_contaminated.jsonl`（作废的 NGRAM 数据，留档）
- `logs/`：每个 server 的日志；`old_round1/`：第一轮（误用 GPU 0–3 那轮）的日志
- 复现：`cd` 到本目录后跑 `scripts/sweep.sh <port> <tag> [model]`（脚本都从本目录相对调用）
环境：sglang main @ `a164c6d9f1`，B300，只用 GPU 6/7。所有速度数字：max_new_tokens 1024，每次跑前 flush cache。

## 一句话结论
- **Nanbeige4.2-3B（22 层 × 2 loop）+ 官方 DSpark draft：能用，效果最好。** greedy 下 conc=1 有 **2.35–2.85×**（gsm8k/humaneval），
  聊天类（sharegpt）1.64×；随并发下降，conc≥64 基本不赚甚至亏。准确率不掉（gsm8k 500 题 93.6% vs 93.6%）。
- **IQuest-Coder-40B-Loop（80 层 × 2 loop）：只有 NGRAM 能跑**，conc=1 有 **1.4–1.9×**，conc≥16 开始亏。DFlash/DSpark/EAGLE/EAGLE3/MTP 全都起不来（模型缺接口）。
- **NGRAM 对小模型不划算**：Nanbeige 上几乎全是减速（最好 1.17×，conc 64 只有 0.2–0.35×），调小 draft token 数也没用。
- **自投机（只跑第 1 轮当 draft）在 Nanbeige 上不可行**：第 1 轮 exit 的 top-1 和完整模型一致率只有 3–4%。

## 主要数字

### Nanbeige4.2-3B，DSpark（block 7）相对 baseline 的吞吐加速
| 数据集 | temp | conc 1 | conc 4 | conc 16 | conc 64 | accept len |
|---|---|---|---|---|---|---|
| gsm8k | 0 | **2.35×** | 2.11× | 1.69× | 1.18× | 4.2 |
| gsm8k | 0.6 | 1.50× | 1.34× | 0.98× | 0.97× | 2.7（conc≥32 时 4.1） |
| humaneval | 0 | **2.85×** | 2.50× | 1.92× | 1.38× | 5.0 |
| humaneval | 0.6 | 1.58× | 1.42× | 1.06× | 1.17× | 2.9 |
| sharegpt | 0 | 1.64× | 1.44× | 1.09× | 0.71× | 2.9 |
| sharegpt | 0.6 | 1.32× | 1.20× | 0.84× | 0.64× | 2.4 |

baseline 单请求 ≈ 288 tok/s；DSpark 单请求最高 ≈ 845 tok/s（humaneval greedy）。
注意：开 spec 后 sglang 自动把 `max_running_requests` 设成 48，所以 "conc 64" 的 DSpark 实际只有 48 并发。

### IQuest-40B-Loop，NGRAM 相对 baseline
| 数据集 | temp | conc 1 | conc 4 | conc 16 | conc 64 | accept |
|---|---|---|---|---|---|---|
| gsm8k | 0 | 1.74× | 1.54× | 1.18× | 0.58× | 2.0 |
| humaneval | 0 | **1.88×** | 1.70× | 1.40× | 0.80× | 2.2 |
| sharegpt | 0 | 1.38× | 1.24× | 0.86× | 0.42× | 1.6 |
| (TP2) humaneval | 0 | 1.69× | 1.56× | 1.29× | 0.83× | 2.3 |

IQuest baseline 单请求只有 34 tok/s（TP1）/ 44 tok/s（TP2）——80GB 权重每个 token 要读两遍，非常 memory-bound，
是最该上 spec decode 的模型，但现在只能用最弱的 NGRAM。

## Spec decode 是怎么接进 loop 的
**把循环展开成 L×loops 个逻辑层**，spec 基础设施基本感知不到循环：
1. 每个物理层给每一轮一个独立的 RadixAttention，`layer_id = loop*L + i`，KV pool 按 L×loops 层分配；各轮共用同一个 token slot（`out_cache_loc`），写在不同层的 buffer。
2. Verify 就是一次普通 forward：所有轮拿同一个 `forward_batch`，树/causal mask 对每轮一样；拒绝的 token 下一轮直接覆盖 slot，EAGLE topk>1 的 KV 搬运覆盖所有层——每轮的 KV 始终一致。
3. CUDA graph 按全局 layer_id 索引展开后的 attention 层。
4. DFlash/DSpark 需要 target 中间层 hidden：层号也在**展开空间**。Nanbeige-DSpark 的 `target_layer_ids=[1,10,21,32,40]`、`num_target_layers=44`，
   即 draft 同时吃第 1 轮（1,10,21）和第 2 轮（32、40）的特征。
5. 每个算法对 target 的要求：NGRAM/STANDALONE 什么都不要；EAGLE 要 `get_embed_and_head`；EAGLE3 还要 `set_eagle3_layers_to_capture`；
   DFlash/DSpark 要捕获 setter + `get_input_embeddings`；MTP 要模型自带 MTP 头。
   → Nanbeige：DSpark/DFlash/EAGLE/NGRAM ✅，EAGLE3 ❌（缺 setter，虽然代码注释说支持），MTP ❌。IQuest：只有 NGRAM/STANDALONE。

## 发现的问题（可以提 PR）
1. **IQuest KV pool 多分配一倍（已实测）**：顶层类缺 `start_layer/end_layer`，层数回落成 160 后又乘了一次 `loop_num` → 分配 320 层（实测 K 76.37GiB / 125k token ⇒ 320 层）。
   KV 容量只有应有的一半。修法：给 `IQuestLoopCoderForCausalLM` 加上和 `NanbeigeForCausalLM` 一样的 `start_layer/end_layer` property。
2. **Nanbeige 缺 `set_eagle3_layers_to_capture`**：社区 EAGLE3 draft（NullSense，vLLM speculators 格式，本身 sglang 也读不了）即使转了格式也会在 `attention_backend_setup.py:55` 崩。
3. **IQuest 缺所有 spec 接口**（`get_embed_and_head` / `get_input_embeddings` / capture setter），`spec_aux_hidden_state.py` 也只认 `num_loops` 不认 `loop_num`。
4. （推断，未验证）IQuest loop>0 的 sliding window 在 FlashInfer 后端可能没生效（wrapper 按 window=-1 plan），输出可能与 HF 不一致。
5. **DSpark 采样下的 accept len 随并发跳变（待查）**：temp 0.6 时 running-req ≤16 → accept 2.7；≥32 → 4.1（几乎等于 greedy）。
   已验证两边都无损（采样 token 的 target logprob 分布和 no-spec baseline 一致；输出不确定、不等于 greedy），也排除了 folded proposal 和 CUDA graph 覆盖。
   如果小 batch 也能走高接受率的那条路径，temp 0.6 低并发可能从 1.5× → 2×+。

## 下一步建议
- 修 IQuest 的 KV bug（一行级别），并给 IQuest 补 EAGLE/DFlash 接口 + 训一个 draft —— 它最慢、最需要加速。
- 查清 DSpark 采样 accept 跳变的开关，可能是白捡的加速。
- Loop 模型的自投机需要训练支持（中间轮 exit loss / exit head）；draft 可以直接复用 target 的 loop-0 KV，零额外显存，结构上很优雅。
- Ouro / Huginn 目前 sglang 不支持，如要评测需要先接模型（Ouro 结构接近 Nanbeige，可先做固定轮数版本）。
