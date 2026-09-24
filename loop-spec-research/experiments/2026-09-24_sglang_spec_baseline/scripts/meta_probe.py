import asyncio, aiohttp, json, sys
from datasets import load_dataset
from transformers import AutoTokenizer
port, conc, temp = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
tok = AutoTokenizer.from_pretrained("Nanbeige/Nanbeige4.2-3B", trust_remote_code=True)
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")][:conc]
async def one(s, q):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    async with s.post(f"http://127.0.0.1:{port}/generate", json={"text": p, "sampling_params": {"temperature": temp, "top_p": 0.95 if temp else 1.0, "max_new_tokens": 1024}}) as r:
        return (await r.json())["meta_info"]
async def main():
    async with aiohttp.ClientSession() as s:
        await s.post(f"http://127.0.0.1:{port}/flush_cache")
        ms = await asyncio.gather(*[one(s, q) for q in qs])
    m = ms[0]; print({k: (v if not isinstance(v, list) or len(v) < 12 else f"list[{len(v)}]") for k, v in m.items() if k.startswith("spec") or k in ("completion_tokens",)})
    tot = lambda k: sum((x.get(k) or 0) if not isinstance(x.get(k), list) else sum(x.get(k)) for x in ms)
    print("sum completion", tot("completion_tokens"), "sum verify_ct", tot("spec_verify_ct"), "=> out/vct", round(tot("completion_tokens")/max(1,tot("spec_verify_ct")),2))
    for k in ("spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_accept_length"):
        print(k, tot(k))
asyncio.run(main())
