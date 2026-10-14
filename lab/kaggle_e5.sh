#!/bin/bash
# Kaggle "Save & Run All" cho E5: GPU0 train E3 rồi GRPO lr=5e-6 (đúng đề); GPU1 chờ E3, đo E3 nền, rồi GRPO lr=5e-5 (đối chứng).
# SMOKE=1 chỉ chạy thử nhanh (2 bước) để bắt lỗi.
cd "$(dirname "$0")/.." || exit 1
bash lab/kaggle_setup.sh
pip uninstall -y -q torchao 2>/dev/null
if [ -n "$SMOKE" ]; then
  CUDA_VISIBLE_DEVICES=0 python -m lab.grpo --smoke 2>&1 | tee work/log_smoke.txt | tail -60
  exit
fi
CUDA_VISIBLE_DEVICES=0 nohup python -m lab.grpo --lr 5e-6 --train-base > work/log_gpu0.txt 2>&1 &
sleep 30
CUDA_VISIBLE_DEVICES=1 nohup python -m lab.grpo --lr 5e-5 --baseline > work/log_gpu1.txt 2>&1 &
while pgrep -f 'lab[.]grpo' > /dev/null; do sleep 120; tail -n 2 work/log_gpu0.txt work/log_gpu1.txt; done
tail -n 30 work/log_gpu0.txt work/log_gpu1.txt
zip -qr /kaggle/working/results.zip work/evidence work/runs/*/run.json work/runs/e5*/hacking.json work/runs/e5*/rollouts.jsonl work/*.jsonl work/log*.txt
zip -qr /kaggle/working/e3_adapter.zip work/runs/e3
ls -la /kaggle/working
