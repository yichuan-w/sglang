#!/bin/bash
# GPU 6 queue: after iq_base sweep, run Nanbeige tuning variants sequentially.
cd /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline
until [ $(grep -c '^{' logs/sweep_iq_base.out) -ge 24 ]; do sleep 30; done
kill $(ps -eo pid,args | grep "[l]aunch_server.*--port 30300 " | awk '{print $1}'); sleep 30
M=Nanbeige/Nanbeige4.2-3B; C="--tp-size 1 --mem-fraction-static 0.8 --reasoning-parser nanbeige --tool-call-parser nanbeige"
run() { # port name args...
  local port=$1 name=$2; shift 2
  scripts/launch2.sh 6 $port $name $M $C "$@" &
  scripts/up.sh $name
  scripts/sweep.sh $port $name > logs/sweep_$name.out 2>&1
  kill $(ps -eo pid,args | grep "[l]aunch_server.*--port $port " | awk '{print $1}'); sleep 30
}
run 30210 nb_dspark4 --speculative-algorithm DSPARK --speculative-draft-model-path Nanbeige/Nanbeige4.2-3B-DSpark --speculative-dspark-block-size 4
run 30211 nb_ngram4 --speculative-algorithm NGRAM --speculative-num-draft-tokens 4
echo QUEUE6_DONE
