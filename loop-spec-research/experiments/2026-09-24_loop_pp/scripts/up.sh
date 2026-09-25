#!/bin/bash
until grep -qE "fired up|Traceback" logs/$1.log 2>/dev/null; do sleep 5; done; grep -E "fired up|Error" logs/$1.log | tail -2
