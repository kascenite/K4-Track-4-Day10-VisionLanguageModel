#!/bin/bash
# Gom kết quả nhỏ (bằng chứng, run.json, dự đoán, ảnh) thành /content/results.zip để tải về máy.
cd "$(dirname "$0")/.." || exit 1
rm -f /content/results.zip
zip -qr /content/results.zip work/evidence work/figures work/vit4bit/*.json work/vit4bit/*.jsonl \
  work/runs/*/run.json work/*.jsonl work/*.json work/log.txt 2>/dev/null
ls -la /content/results.zip
