#!/bin/bash
# wait until server log $1 shows ready or crash
until grep -qE "fired up|Traceback" /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/logs/$1.log 2>/dev/null; do sleep 5; done
grep -E "fired up|Error:" /ssd1/k3/yichuan/sglang/loop-spec-research/experiments/2026-09-24_sglang_spec_baseline/logs/$1.log | tail -2
