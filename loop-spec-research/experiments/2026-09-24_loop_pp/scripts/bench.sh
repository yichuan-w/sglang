#!/bin/bash
# usage: scripts/bench.sh <port> <tag> <model> [concs...]  -> results/<tag>.jsonl (bench_serving random, in 1024 / out 512)
V=/ssd1/k3/yichuan/venvs/sglang-main; export HF_HUB_OFFLINE=1
PORT=$1; TAG=$2; MODEL=$3; shift 3
for c in ${@:-1 16 64 256}; do
  n=$(( c*4 > 64 ? c*4 : 64 ))
  $V/bin/python -m sglang.bench_serving --backend sglang --port $PORT --model $MODEL --dataset-name random \
    --random-input-len 1024 --random-output-len 512 --random-range-ratio 1.0 --num-prompts $n --max-concurrency $c \
    --flush-cache --warmup-requests 4 --output-file results/$TAG.jsonl > logs/bench_${TAG}_c$c.txt 2>&1
  grep -E "Output token throughput|Median TPOT|Median TTFT" logs/bench_${TAG}_c$c.txt | awk -v t=$TAG -v c=$c '{printf "%s c=%s %s\n", t, c, $0}'
done
