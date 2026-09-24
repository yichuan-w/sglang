#!/bin/bash
cd /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline; P=/ssd1/k3/yichuan/venvs/sglang-main/bin/python; export HF_HUB_OFFLINE=1
IQ=/ssd1/k3/yichuan/hf/IQuest-Coder-V1-40B-Loop-Instruct
sleep 30
run() { local port=$1 name=$2; shift 2
  scripts/launch2.sh 6,7 $port $name $IQ --tp-size 2 --mem-fraction-static 0.85 --trust-remote-code "$@" &
  scripts/up.sh $name; grep "KV Cache is allocated" logs/$name.log
  for ds in gsm8k humaneval; do for conc in 1 4 16 64; do n=$(( conc < 8 ? 32 : conc * 4 ))
    $P scripts/bench2.py --port $port --tag $name --dataset $ds --conc $conc --n $n --temp 0 --max-tokens 1024 --model $IQ 2>/dev/null | tail -1
  done; done
  kill $(ps -eo pid,args | grep "[l]aunch_server.*--port $port " | awk '{print $1}'); sleep 40
}
run 30310 iq_tp2_base
run 30311 iq_tp2_ngram --speculative-algorithm NGRAM
echo TP2_DONE
