#!/bin/bash
# usage: sweep.sh <port> <tag> [model]
cd /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline
export HF_HUB_OFFLINE=1
P=/ssd1/k3/yichuan/venvs/sglang-main/bin/python
MODEL=${3:-Nanbeige/Nanbeige4.2-3B}
for ds in gsm8k humaneval sharegpt; do
  for temp in 0 0.6; do
    for conc in 1 4 16 64; do
      n=$(( conc < 8 ? 32 : conc * 4 ))
      $P scripts/bench2.py --port $1 --tag $2 --dataset $ds --conc $conc --n $n --temp $temp --max-tokens 1024 --model $MODEL 2>>logs/sweep_$2.err | tail -1
    done
  done
done
