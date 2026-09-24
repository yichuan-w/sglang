"""Self-speculation potential of a looped model: draft = exit after first loop (num_loops=1 server),
target = full model (num_loops=2 server). Teacher-forced on the full model's own greedy continuation, so
agreement of loop-1 top-1 with the continuation == greedy chain-spec acceptance exactly.
usage: self_spec_probe.py --full-port P --draft-port Q --dataset D --n N"""
import argparse, asyncio, json, aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--full-port", type=int); ap.add_argument("--draft-port", type=int)
ap.add_argument("--dataset", default="gsm8k"); ap.add_argument("--n", type=int, default=64); ap.add_argument("--gen", type=int, default=512)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
if a.dataset == "gsm8k":
    qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][: a.n]
elif a.dataset == "humaneval":
    qs = ["Complete the following Python function. Return only the code.\n\n" + r["prompt"] for r in load_dataset("openai/openai_humaneval", split="test")][: a.n]
else:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    qs = [c["conversations"][0]["value"] for c in json.load(open(p)) if c.get("conversations") and c["conversations"][0]["from"] == "human" and 20 < len(c["conversations"][0]["value"]) < 2000][: a.n]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True) for q in qs]
prompts = [p["input_ids"] if hasattr(p, "keys") else list(p) for p in prompts]
async def post(s, port, body):
    async with s.post(f"http://127.0.0.1:{port}/generate", json=body) as r:
        return await r.json()
async def main():
    sem = asyncio.Semaphore(32)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        async def one(pids):
            async with sem:
                g = await post(s, a.full_port, {"input_ids": pids, "sampling_params": {"temperature": 0, "max_new_tokens": a.gen}, "return_logprob": True})
                out = [x[1] for x in g["meta_info"]["output_token_logprobs"]]
                ids = pids + out
                body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "return_logprob": True,
                        "logprob_start_len": len(pids) - 1, "top_logprobs_num": 1}
                d = await post(s, a.draft_port, body)
                tops = [t[0][1] if t else None for t in d["meta_info"]["input_top_logprobs"]]
                # tops[j] predicts ids[len(pids)-1+j+1]; align with out
                return [int(tops[j] == out[j]) for j in range(min(len(out), len(tops)))]
        res = await asyncio.gather(*[one(p) for p in prompts])
    allm = [x for r in res for x in r]
    rec = {"dataset": a.dataset, "n": len(res), "positions": len(allm), "top1_agree": round(sum(allm) / len(allm), 4)}
    for gamma in (1, 2, 3, 4, 5, 7):  # simulate greedy chain spec: each step accepts run of matches (<=gamma) + 1 bonus
        toks = steps = 0
        for m in res:
            t = 0
            while t < len(m):
                k = 0
                while k < gamma and t + k < len(m) and m[t + k]: k += 1
                t += k + 1; toks += k + 1; steps += 1
        rec[f"accept_len_g{gamma}"] = round(toks / steps, 3)
    print(json.dumps(rec))
    open("/ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/results/self_spec.jsonl", "a").write(json.dumps(rec) + "\n")
asyncio.run(main())
