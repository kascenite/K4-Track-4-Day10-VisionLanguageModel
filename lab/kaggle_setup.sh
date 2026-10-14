#!/bin/bash
# Chuẩn bị trên Kaggle: cài thư viện ghim, dữ liệu từ input dataset, work/ trong /kaggle/working, tải trước model.
cd "$(dirname "$0")/.." || exit 1
pip -q install -r requirements-colab.txt 2>&1 | tail -2
pip uninstall -y -q torchao 2>/dev/null
mkdir -p work/evidence
cp -n report/evidence/e2.json report/evidence/e3.json work/evidence/  # để E4 chọn đúng setup tốt nhất (E3)
if [ ! -f data/splits/meta.json ]; then
  SRC=$(python -c "import kagglehub; print(kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'))" | tail -1)
  python scripts/prepare_data.py --source "$SRC" --work /kaggle/tmp | tail -5
fi
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-VL-4B-Instruct')" 2>&1 | tail -1
nvidia-smi -L
