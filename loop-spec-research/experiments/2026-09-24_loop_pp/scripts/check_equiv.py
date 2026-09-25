"""Correctness of the unrolled-PP refactor: compare servers against a reference server.
Greedy continuations must match token-for-token; teacher-forced logprobs of the reference
continuation must agree. usage: check_equiv.py --ref PORT --cand PORT [PORT...] --n N"""
import argparse, asyncio, json, aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=int); ap.add_argument("--cand", type=int, nargs="+"); ap.add_argument("--n", type=int, default=64)
ap.add_argument("--gen", type=int, default=256)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][: a.n]
prompts = []
for q in qs:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True)
    prompts.append(p["input_ids"] if hasattr(p, "keys") else list(p))
async def gen(s, port, ids):
    async with s.post(f"http://127.0.0.1:{port}/generate", json={"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": a.gen}}) as r:
        return (await r.json())["output_ids"]
async def score(s, port, ids, start):
    body = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "return_logprob": True, "logprob_start_len": start}
    async with s.post(f"http://127.0.0.1:{port}/generate", json=body) as r:
        return [x[0] for x in (await r.json())["meta_info"]["input_token_logprobs"] if x[0] is not None]
async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        ref_out = await asyncio.gather(*[gen(s, a.ref, p) for p in prompts])
        ref_lp = await asyncio.gather(*[score(s, a.ref, p + o, len(p)) for p, o in zip(prompts, ref_out)])
        for port in a.cand:
            out = await asyncio.gather(*[gen(s, port, p) for p in prompts])
            lp = await asyncio.gather(*[score(s, port, p + o, len(p)) for p, o in zip(prompts, ref_out)])
            same = sum(x == y for x, y in zip(out, ref_out))
            first_div = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None) for x, y in zip(out, ref_out)]
            diffs = [abs(u - v) for x, y in zip(lp, ref_lp) for u, v in zip(x, y)]
            print(json.dumps({"ref": a.ref, "cand": port, "identical_greedy": f"{same}/{len(prompts)}",
                              "earliest_divergence_tok": min([d for d in first_div if d is not None], default=None),
                              "max_abs_logprob_diff": round(max(diffs), 5), "mean_abs_logprob_diff": round(sum(diffs) / len(diffs), 6)}))
asyncio.run(main())
