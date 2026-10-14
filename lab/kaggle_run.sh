#!/bin/bash
# Kaggle "Save & Run All": chuẩn bị, chạy song song 2 GPU (e0+e4 | e1), CHỜ xong, gom kết quả vào /kaggle/working.
cd "$(dirname "$0")/.." || exit 1
bash lab/kaggle_setup.sh
GPU=0 bash lab/launch.sh e0 e4
sleep 30
GPU=1 bash lab/launch.sh e1
while pgrep -f 'lab[.]run_all' > /dev/null; do sleep 120; bash lab/status.sh 2 | grep -e step -e '/311' -e 'test\]' | tail -4; done
bash lab/status.sh 30
python -m lab.figures e0 e1 e3 2>&1 | tail -1
zip -qr /kaggle/working/results.zip work/evidence work/figures work/runs/*/run.json work/*.jsonl work/*.json work/log*.txt
ls -la /kaggle/working
