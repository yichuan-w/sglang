import asyncio, aiohttp, sys
from datasets import load_dataset
from transformers import AutoTokenizer
port, conc = int(sys.argv[1]), int(sys.argv[2])
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][:conc]
ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
async def run(s, temp):
    await s.post(f"http://127.0.0.1:{port}/flush_cache")
    async def one(p):
        async with s.post(f"http://127.0.0.1:{port}/generate", json={"text": p, "sampling_params": {"temperature": temp, "top_p": 0.95 if temp else 1.0, "max_new_tokens": 512}}) as r:
            return (await r.json())["text"]
    return await asyncio.gather(*[one(p) for p in ps])
async def main():
    async with aiohttp.ClientSession() as s:
        a = await run(s, 0.6); b = await run(s, 0.6); g = await run(s, 0.0)
    same = lambda x, y: sum(i == j for i, j in zip(x, y))
    pre = lambda x, y, n=200: sum(i[:n] == j[:n] for i, j in zip(x, y))
    print(f"conc={conc}: sampled run1==run2: {same(a,b)}/{conc} (first200ch {pre(a,b)}), run1==greedy: {same(a,g)}/{conc} (first200ch {pre(a,g)})")
asyncio.run(main())
