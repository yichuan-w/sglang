"""Spec-decode bench via native /generate so we get spec_verify_ct per request.
usage: bench2.py --port P --tag T --dataset {gsm8k,humaneval,sharegpt} --conc C --n N --temp X [--max-tokens M]
Appends one JSON line to results/all.jsonl."""
import argparse, asyncio, json, re, statistics, time, aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
for a, t, d in [("port", int, None), ("tag", str, None), ("dataset", str, "gsm8k"), ("conc", int, 1),
                ("n", int, 16), ("temp", float, 0.0), ("max_tokens", int, 1024), ("model", str, "Nanbeige/Nanbeige4.2-3B")]:
    ap.add_argument("--" + a.replace("_", "-"), type=t, default=d)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if args.dataset == "gsm8k":
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[: args.n]
    items = [(r["question"] + "\nPut the final numeric answer after '####'.", r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
elif args.dataset == "humaneval":
    rows = list(load_dataset("openai/openai_humaneval", split="test"))[: args.n]
    items = [("Complete the following Python function. Return only the code.\n\n" + r["prompt"], None) for r in rows]
else:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    items = []
    for c in json.load(open(p)):
        conv = c.get("conversations", [])
        if conv and conv[0]["from"] == "human" and 20 < len(conv[0]["value"]) < 2000:
            items.append((conv[0]["value"], None))
        if len(items) >= args.n: break
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q, _ in items]
def extract(txt):
    txt = txt.split("</think>")[-1]
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", txt) or re.findall(r"(-?[\d,]*\.?\d+)", txt)
    return m[-1].replace(",", "").rstrip(".") if m else None
async def one(s, i, sem, res):
    async with sem:
        t = time.perf_counter()
        sp = {"temperature": args.temp, "top_p": 0.95 if args.temp > 0 else 1.0, "max_new_tokens": args.max_tokens}
        async with s.post(f"http://127.0.0.1:{args.port}/generate", json={"text": prompts[i], "sampling_params": sp}) as r:
            j = await r.json()
        m = j["meta_info"]
        res.append(dict(i=i, out=m["completion_tokens"], dt=time.perf_counter() - t, vct=m.get("spec_verify_ct", 0), text=j["text"]))
async def main():
    sem, res = asyncio.Semaphore(args.conc), []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        async with s.post(f"http://127.0.0.1:{args.port}/flush_cache") as r:  # reset radix + NGRAM corpus
            assert r.status == 200, await r.text()
        await one(s, 0, asyncio.Semaphore(1), [])  # warmup
        t = time.perf_counter()
        await asyncio.gather(*[one(s, i, sem, res) for i in range(len(prompts))])
        el = time.perf_counter() - t
    out = sum(r["out"] for r in res); vct = sum(r["vct"] for r in res)
    rec = dict(tag=args.tag, dataset=args.dataset, conc=args.conc, n=len(prompts), temp=args.temp, max_tokens=args.max_tokens,
               out_tokens=out, wall_s=round(el, 1), throughput_tok_s=round(out / el, 1),
               per_req_tok_s=round(statistics.mean(r["out"] / r["dt"] for r in res), 1),
               accept_len=round(out / vct, 2) if vct else None)
    if args.dataset == "gsm8k":
        rec["gsm8k_acc"] = round(sum(extract(r["text"]) == items[r["i"]][1] for r in res) / len(res), 3)
    print(json.dumps(rec)); open("/ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/results/all.jsonl", "a").write(json.dumps(rec) + "\n")
asyncio.run(main())
