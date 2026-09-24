"""Is sampling really stochastic under spec decode at a given concurrency?
Sends the SAME prompt K times concurrently with temp T; reports #distinct outputs and accept len.
usage: sample_check.py --port P --k K --temp T [--tokens N] [--mix-greedy M]"""
import argparse, asyncio, json, aiohttp
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int); ap.add_argument("--k", type=int, default=64); ap.add_argument("--temp", type=float, default=0.6)
ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--mix-greedy", type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B")
prompt = tok.apply_chat_template([{"role": "user", "content": "Write a short story about a robot learning to paint."}], tokenize=False, add_generation_prompt=True)
async def one(s, temp):
    async with s.post(f"http://127.0.0.1:{a.port}/generate", json={"text": prompt, "sampling_params": {"temperature": temp, "top_p": 0.95 if temp > 0 else 1.0, "max_new_tokens": a.tokens, "ignore_eos": True}}) as r:
        j = await r.json(); return j["text"], j["meta_info"]
async def main():
    async with aiohttp.ClientSession() as s:
        await s.post(f"http://127.0.0.1:{a.port}/flush_cache")
        temps = [a.temp] * a.k + [0.0] * a.mix_greedy
        res = await asyncio.gather(*[one(s, t) for t in temps])
    samp = res[: a.k]
    out = sum(m["completion_tokens"] for _, m in samp); vct = sum(m.get("spec_verify_ct", 0) for _, m in samp)
    print(json.dumps(dict(port=a.port, k=a.k, temp=a.temp, mix_greedy=a.mix_greedy, distinct=len({t for t, _ in samp}),
                          distinct_first32tok=len({t[:120] for t, _ in samp}), accept_len=round(out / vct, 2) if vct else None)))
asyncio.run(main())
