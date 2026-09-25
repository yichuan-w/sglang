#!/bin/bash
# Reference: how good is sglang PP on a normal model? Qwen3-8B, 2 GPUs: TP2 vs PP2 vs DP2 (+ TP1 single GPU)
cd /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_loop_pp
M=Qwen/Qwen3-8B; C="--mem-fraction-static 0.85"
stop() { kill $(ps -eo pid,args | grep "[l]aunch_server.*--port $1 " | awk '{print $1}') 2>/dev/null; sleep 30; }
scripts/launch.sh 4,5 30400 q_tp2 $M $C --tp-size 2 &  scripts/launch.sh 6,7 30401 q_pp2 $M $C --pp-size 2 &
scripts/up.sh q_tp2; scripts/up.sh q_pp2
scripts/bench.sh 30400 q_tp2 $M & scripts/bench.sh 30401 q_pp2 $M & wait
stop 30400; stop 30401
scripts/launch.sh 4,5 30402 q_dp2 $M $C --dp-size 2 &  scripts/launch.sh 6 30403 q_tp1 $M $C &
scripts/up.sh q_dp2; scripts/up.sh q_tp1
scripts/bench.sh 30402 q_dp2 $M & scripts/bench.sh 30403 q_tp1 $M & wait
stop 30402; stop 30403
echo REF_DONE
