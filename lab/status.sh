#!/bin/bash
# Trạng thái chạy nền: log đã lọc, tiến trình, VRAM, bằng chứng đã có.
cd "$(dirname "$0")/.." || exit 1
date
tr '\r' '\n' < work/log.txt | grep -v -e '^\s*$' -e 'Loading weights' -e 'Fetching' -e 'Kwargs passed' \
  -e 'use_cache' -e 'unauthenticated' | tail -n "${1:-12}" | cut -c1-140
echo "--- procs: $(pgrep -fc 'lab[.]run_all')  vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
ls work/evidence
