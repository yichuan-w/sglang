#!/bin/bash
# usage: scripts/launch.sh <gpus> <port> <name> <model> [sglang args...]   (run from experiment dir)
V=/ssd1/k3/yichuan/venvs/sglang-main
export PATH=$V/bin:$HOME/.cargo/bin:$PATH CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 HF_HUB_OFFLINE=1
GPUS=$1; PORT=$2; NAME=$3; MODEL=$4; shift 4
CUDA_VISIBLE_DEVICES=$GPUS exec python -m sglang.launch_server --model-path $MODEL --port $PORT "$@" > logs/$NAME.log 2>&1
