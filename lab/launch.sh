#!/bin/bash
# Chạy nền run_all với các bước truyền vào (mặc định e0 e1 e2 e3 e4). Không chạy trùng.
cd "$(dirname "$0")/.." || exit 1
if pgrep -f 'lab[.]run_all' > /dev/null; then echo "đang chạy rồi"; exit 1; fi
pip uninstall -y -q torchao 2>/dev/null  # torchao 0.10 của Colab không tương thích peft 0.21 (LoRA trên nn.Linear)
echo "=== launch $* $(date) ===" >> work/log.txt
nohup python -m lab.run_all "$@" >> work/log.txt 2>&1 &
echo "launched pid $!"
