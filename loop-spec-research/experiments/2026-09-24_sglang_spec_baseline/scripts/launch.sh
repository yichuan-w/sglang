#!/bin/bash
# usage: launch.sh <gpu> <port> <name> [extra args...]
V=/ssd1/k3/yichuan/venvs/sglang-main
export PATH=$V/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 HF_HUB_OFFLINE=1
GPU=$1; PORT=$2; NAME=$3; shift 3
CUDA_VISIBLE_DEVICES=$GPU exec python -m sglang.launch_server --model-path Nanbeige/Nanbeige4.2-3B \
  --port $PORT --tp-size 1 --mem-fraction-static 0.8 --reasoning-parser nanbeige --tool-call-parser nanbeige "$@" \
  > /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/$NAME.log 2>&1
