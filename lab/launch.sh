#!/bin/bash
# Chạy nền run_all với các bước truyền vào (mặc định e0 e1 e2 e3 e4).
# GPU=<id> chọn GPU (máy nhiều GPU, vd Kaggle T4 x2) và ghi log riêng work/log_gpu<id>.txt;
# không đặt GPU thì không cho chạy trùng.
cd "$(dirname "$0")/.." || exit 1
LOG=work/log.txt
if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES=$GPU
  LOG=work/log_gpu$GPU.txt
elif pgrep -f 'lab[.]run_all' > /dev/null; then
  echo "đang chạy rồi"; exit 1
fi
pip uninstall -y -q torchao 2>/dev/null  # torchao 0.10 của Colab không tương thích peft 0.21 (LoRA trên nn.Linear)
echo "=== launch $* gpu=${GPU:-0} $(date) ===" >> "$LOG"
nohup python -m lab.run_all "$@" >> "$LOG" 2>&1 &
echo "launched pid $! -> $LOG"
