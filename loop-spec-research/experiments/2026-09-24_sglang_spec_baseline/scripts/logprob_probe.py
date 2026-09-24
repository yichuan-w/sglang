"""Mean target logprob of sampled tokens (temp 0.6, top_p 0.95). Correct spec sampling must match no-spec baseline
regardless of concurrency. usage: logprob_probe.py port conc nreq"""
import asyncio, aiohttp, sys, statistics, json
from datasets import load_dataset
from transformers import AutoTokenizer
port, conc, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][:n]
ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
async def main():
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as s:
        await s.post(f"http://127.0.0.1:{port}/flush_cache")
        async def one(p):
            async with sem:
                async with s.post(f"http://127.0.0.1:{port}/generate", json={"text": p, "return_logprob": True,
                        "sampling_params": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 512}}) as r:
                    m = (await r.json())["meta_info"]
                    return [x[0] for x in m["output_token_logprobs"]], m.get("spec_verify_ct", 0), m["completion_tokens"]
        res = await asyncio.gather(*[one(p) for p in ps])
    lps = [x for r in res for x in r[0] if x is not None]
    per_req = [statistics.mean([x for x in r[0] if x is not None]) for r in res]
    vct = sum(r[1] for r in res); out = sum(r[2] for r in res)
    print(json.dumps(dict(port=port, conc=conc, n=n, tokens=len(lps), mean_lp=round(statistics.mean(lps), 4),
        sem=round(statistics.stdev(per_req) / len(per_req) ** 0.5, 4), frac_lp_gt_m001=round(sum(x > -0.01 for x in lps) / len(lps), 4),
        accept_len=round(out / vct, 2) if vct else None)))
asyncio.run(main())
