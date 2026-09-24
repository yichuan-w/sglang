#!/bin/bash
cd /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline; P=/ssd1/k3/yichuan/venvs/sglang-main/bin/python; export HF_HUB_OFFLINE=1
for temp in 0 0.6; do
  $P scripts/bench2.py --port 30205 --tag acc_nb_base --dataset gsm8k --conc 48 --n 500 --temp $temp --max-tokens 8192 2>/dev/null | tail -1 &
  $P scripts/bench2.py --port 30204 --tag acc_nb_dspark7 --dataset gsm8k --conc 48 --n 500 --temp $temp --max-tokens 8192 2>/dev/null | tail -1 &
  wait
done
