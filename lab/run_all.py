"""Chạy toàn bộ E0–E4 tuần tự, tiếp tục được nếu bị ngắt (bỏ qua bước đã có kết quả).

Colab:  !nohup python -m lab.run_all > work/log.txt 2>&1 &
"""
import gc
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from lab import vlm as V

ROOT = V.ROOT
WORK = ROOT / "work"
EV = WORK / "evidence"  # trên Drive; chép về report/evidence khi nộp
SETUPS = {"e1": dict(train_projector=False, vit_lora=False),
          "e2": dict(train_projector=True, vit_lora=False),
          "e3": dict(train_projector=True, vit_lora=True)}


def log(*a):
    print(*a, flush=True)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def score(pred, name, out, split="test", meta=""):
    cmd = (f"{sys.executable} scripts/score.py --pred {pred} --split {split} --name '{name}' "
           f"--out {out} {meta}")
    log(subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout)


def step_e0(splits):
    if (EV / "e0.json").exists():
        return
    model, proc = V.load_model()
    log("modules:", [n for n, _ in model.named_modules()
                     if n.endswith(("layers.0.self_attn.q_proj", "blocks.0.attn.qkv", "visual.merger"))])
    V.predict(model, proc, splits["val"][:40], WORK / "e0_val40.jsonl", batch_size=16)
    score(WORK / "e0_val40.jsonl", "E0-val40", WORK / "e0_val40.json", split="val")
    V.predict(model, proc, splits["test"], WORK / "e0.jsonl", batch_size=16)
    score(WORK / "e0.jsonl", "E0", EV / "e0.json")
    for r in V.read_jsonl(WORK / "e0.jsonl")[:5]:
        log("RAW", repr(r["raw"][:300]))
    # Bước 4: kiểm tra encode + mask
    e = splits["train"][0]
    ex = V.encode_example(proc, V.load_image(e), V.PROMPT_NORM, V.objects_to_answer(e["objects"]))
    tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    log("encode keys", list(ex), "image tokens", int((ex["input_ids"] == tok).sum()),
        "grid", ex["image_grid_thw"].tolist())
    log("label text", repr(proc.tokenizer.decode(ex["input_ids"][ex["labels"] != -100])))
    log("answer    ", repr(V.objects_to_answer(e["objects"])))
    del model
    free()


def run(key, setup, entries, r=16):
    run_dir = WORK / "runs" / key
    if not (run_dir / "run.json").exists():
        model, proc = V.load_model()
        pm, n_matched, n_train = V.apply_setup(model, r=r, alpha=2 * r, **setup)
        log(key, "matched", n_matched, "trainable", n_train)
        res = V.train(pm, proc, entries, run_dir, setup["train_projector"], grad_ckpt=True)  # như E1–E3: tài nguyên so sánh được
        log(key, {k: v for k, v in res.items() if k != "losses"})
        del model, pm
        free()
    return run_dir


def evaluate(key, name, run_dir, splits):
    if (EV / f"{key}.json").exists():
        return
    pm, proc = V.load_trained(run_dir)
    V.predict(pm, proc, splits["test"], WORK / f"{key}.jsonl", batch_size=16)
    del pm
    free()
    score(WORK / f"{key}.jsonl", name, EV / f"{key}.json", meta=V.meta_args(run_dir))


def f1(key):
    p = EV / f"{key}.json"
    return json.loads(p.read_text())["micro"]["f1"] if p.exists() else -1.0


def e4(splits):
    best = max(SETUPS, key=f1)
    log("best setup:", best, {k: f1(k) for k in SETUPS})
    sub = random.Random(42).sample(splits["train"], int(0.1 * len(splits["train"])))
    key = "e4_data_size_0.1"
    evaluate(key, f"E4 data_size=0.1 ({best.upper()})", run(key, SETUPS[best], sub), splits)
    full = EV / "e4_data_size_1.0.json"
    if not full.exists():
        d = json.loads((EV / f"{best}.json").read_text())
        d["name"] = f"E4 data_size=1.0 ({best.upper()})"
        full.write_text(json.dumps(d, ensure_ascii=False, indent=2))


def main():
    EV.mkdir(parents=True, exist_ok=True)
    splits = {s: V.load_split(s) for s in ("train", "val", "test")}
    steps = sys.argv[1:] or ["e0", "e1", "e2", "e3", "e4"]  # chạy theo đúng thứ tự tham số
    for key in steps:
        if key == "e0":
            step_e0(splits)
        elif key in SETUPS:
            evaluate(key, key.upper(), run(key, SETUPS[key], splits["train"]), splits)
        elif key == "e4":
            e4(splits)
    shutil.copytree(EV, ROOT / "report" / "evidence", dirs_exist_ok=True)
    log("DONE")


if __name__ == "__main__":
    main()
