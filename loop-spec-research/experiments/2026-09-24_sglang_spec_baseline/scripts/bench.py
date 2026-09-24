import asyncio, json, sys, time, aiohttp, statistics
from datasets import load_dataset
port, conc, n, temp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
maxtok = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][:n]
async def one(sess, q, sem, res):
    async with sem:
        t = time.perf_counter()
        async with sess.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": q}],
            "temperature": temp, "top_p": 0.95 if temp > 0 else 1.0, "max_tokens": maxtok}) as r:
            j = await r.json()
        res.append((j["usage"]["completion_tokens"], time.perf_counter() - t))
async def main():
    sem, res = asyncio.Semaphore(conc), []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        # warmup
        await one(s, "hi", asyncio.Semaphore(1), [])
        t = time.perf_counter()
        await asyncio.gather(*[one(s, q, sem, res) for q in qs])
        el = time.perf_counter() - t
    toks = sum(a for a, _ in res)
    per_req = statistics.mean(a / b for a, b in res)
    print(json.dumps({"port": port, "conc": conc, "n": n, "temp": temp, "out_tokens": toks,
                      "throughput_tok_s": round(toks / el, 1), "per_req_tok_s": round(per_req, 1)}))
asyncio.run(main())
