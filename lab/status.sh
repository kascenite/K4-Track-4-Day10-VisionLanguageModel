#!/bin/bash
# Trạng thái chạy nền: log đã lọc, tiến trình, VRAM, bằng chứng đã có.
cd "$(dirname "$0")/.." || exit 1
date
tr '\r' '\n' < work/log.txt | grep -v -e '^\s*$' -e 'Loading weights' -e 'Fetching' -e 'Kwargs passed' \
  -e 'use_cache' -e 'unauthenticated' | tail -n "${1:-12}" | cut -c1-140
echo "--- procs: $(pgrep -fc 'lab[.]run_all')  vram: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
ls work/evidence
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("work/evidence").glob("*.json")):
    d = json.loads(p.read_text()); m = d["micro"]; pc = d["per_class"]
    print(f"{p.stem:18s} F1={m['f1']:.3f} CI={d['f1_ci95'][0]:.3f}-{d['f1_ci95'][1]:.3f} valid={d['parse_valid_rate']:.3f} "
          f"FP/img={d['fp_per_image']:.2f} " + " ".join(f"{c}={pc[c]['f1']:.3f}" for c in ("pothole", "crack", "manhole")) + f" {d['meta']}")
PY
