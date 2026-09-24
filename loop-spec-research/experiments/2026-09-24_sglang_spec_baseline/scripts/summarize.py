"""Build speedup tables from results/all.jsonl. usage: summarize.py <base_tag> <tag> [<tag>...]"""
import json, sys
from collections import defaultdict
R = defaultdict(dict)
for l in open("/ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/results/all.jsonl"):
    r = json.loads(l); R[r["tag"]][(r["dataset"], r["temp"], r["conc"])] = r  # later runs override
base, tags = sys.argv[1], sys.argv[2:]
hdr = "| dataset | temp | conc | base tok/s (per-req) | " + " | ".join(f"{t} tok/s (per-req) | {t} 加速 | {t} accept" for t in tags) + " |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
for k in sorted(R[base]):
    b = R[base][k]; row = f"| {k[0]} | {k[1]} | {k[2]} | {b['throughput_tok_s']:.0f} ({b['per_req_tok_s']:.0f}) |"
    for t in tags:
        r = R[t].get(k)
        row += f" {r['throughput_tok_s']:.0f} ({r['per_req_tok_s']:.0f}) | **{r['throughput_tok_s']/b['throughput_tok_s']:.2f}×** | {r['accept_len']} |" if r else " - | - | - |"
    print(row)
