#!/bin/bash
# usage: launch2.sh <gpus> <port> <name> <model> [extra sglang args...]
V=/ssd1/k3/yichuan/venvs/sglang-main
export PATH=$V/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 HF_HUB_OFFLINE=1
GPUS=$1; PORT=$2; NAME=$3; MODEL=$4; shift 4
CUDA_VISIBLE_DEVICES=$GPUS exec python -m sglang.launch_server --model-path $MODEL --port $PORT "$@" \
  > /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/logs/$NAME.log 2>&1
