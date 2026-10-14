"""E5: GRPO với reward IoU (GUIDE bước 9), khởi tạo từ adapter E3, trl.GRPOTrainer.

python -m lab.grpo --lr 5e-6 --train-base   # train E3 nếu chưa có adapter, rồi GRPO + đánh giá (GPU 0)
python -m lab.grpo --lr 5e-5 --baseline     # chờ E3, đo E3 nền, rồi GRPO với lr khác (GPU 1)
python -m lab.grpo --smoke                  # chạy thử 2 bước trên vài ảnh để bắt lỗi sớm
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from lab import run_all as R
from lab import vlm as V
from lab.reward import reward
from scripts.score import _gt_boxes, parse_output

WORK, EV, log = R.WORK, R.EV, R.log


def load_policy(run_dir):
    """Như V.load_trained nhưng LoRA còn trainable; projector nạp lại và đóng băng."""
    from peft import PeftModel
    run_dir = Path(run_dir)
    model, proc = V.load_model()
    pm = PeftModel.from_pretrained(model, run_dir / "adapter", is_trainable=True)
    proj = run_dir / "projector.pt"
    if proj.exists():
        sd = torch.load(proj, map_location="cpu")
        res = pm.load_state_dict(sd, strict=False)
        assert not [k for k in sd if k in res.unexpected_keys], "projector keys không khớp"
    for p in pm.parameters():  # GradScaler cần grad fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return pm, proc


def make_dataset(entries):
    from datasets import Dataset, Image
    ds = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": V.PROMPT_NORM}]}]] * len(entries),
        "image": [str(V.DATA / e["image_path"]) for e in entries],
        "gt": [[{"label": g["label"], "bbox": list(g["bbox"])} for g in _gt_boxes(e)] for e in entries],
    })
    return ds.cast_column("image", Image())


def make_reward(rollouts):
    def reward_iou(completions, gt, trainer_state=None, **kw):
        out = []
        step = trainer_state.global_step if trainer_state is not None else -1
        with open(rollouts, "a") as f:
            for c, g in zip(completions, gt):
                text = c[0]["content"] if isinstance(c, list) else c
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text)
                r = reward(text, g)
                pred = parse_output(text)
                areas = [max(0, b["bbox"][2] - b["bbox"][0]) * max(0, b["bbox"][3] - b["bbox"][1]) / 1e6
                         for b in (pred or [])]
                f.write(json.dumps({"step": step, "reward": r, "n_gt": len(g),
                                    "n_pred": None if pred is None else len(pred),
                                    "max_area": max(areas, default=0.0), "text": text[:300]},
                                   ensure_ascii=False) + "\n")
                out.append(r)
        return out
    return reward_iou


def hacking_summary(rollouts, window=10):
    """Dấu hiệu reward hacking theo cửa sổ bước: parse hợp lệ, tỉ lệ trả [], số box dự đoán/GT, box khổng lồ."""
    rows = V.read_jsonl(rollouts)
    out = []
    for lo in range(0, max(r["step"] for r in rows) + 1, window):
        w = [r for r in rows if lo <= r["step"] < lo + window]
        if not w:
            continue
        ok = [r for r in w if r["n_pred"] is not None]
        out.append({"steps": f"{lo}-{lo + window - 1}", "n": len(w),
                    "reward": round(sum(r["reward"] for r in w) / len(w), 4),
                    "parse_rate": round(len(ok) / len(w), 3),
                    "empty_rate": round(sum(r["n_pred"] == 0 for r in ok) / max(len(ok), 1), 3),
                    "gt_empty_rate": round(sum(r["n_gt"] == 0 for r in w) / len(w), 3),
                    "n_pred": round(sum(r["n_pred"] for r in ok) / max(len(ok), 1), 2),
                    "n_gt": round(sum(r["n_gt"] for r in w) / len(w), 2),
                    "big_box_rate": round(sum(r["max_area"] > 0.5 for r in ok) / max(len(ok), 1), 3)})
    return out


def wait_for(path):
    while not Path(path).exists():
        time.sleep(60)
    time.sleep(5)


def grpo(base_dir, out_dir, entries, lr, max_steps, n_prompts):
    from trl import GRPOConfig, GRPOTrainer
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts = out_dir / "rollouts.jsonl"
    rollouts.unlink(missing_ok=True)
    pm, proc = load_policy(base_dir)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    log("GRPO trainable", n_train, "lr", lr)
    sample = random.Random(42).sample(entries, n_prompts)
    cfg = GRPOConfig(
        output_dir=str(out_dir / "trainer"), num_generations=4, max_completion_length=128,
        max_steps=max_steps, learning_rate=lr, beta=0.0, use_vllm=False,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,  # 8 completion/bước = 2 ảnh x 4
        fp16=True, bf16=False, gradient_checkpointing=True, logging_steps=1, save_strategy="no",
        report_to="none", seed=42, lr_scheduler_type="constant", warmup_steps=0)
    trainer = GRPOTrainer(model=pm, reward_funcs=make_reward(rollouts), args=cfg,
                          train_dataset=make_dataset(sample), processing_class=proc)
    for p in trainer.model.parameters():  # trl đổi tham số trainable sang bf16: với lr nhỏ bf16 nuốt mất cập nhật
        if p.requires_grad:
            p.data = p.data.float()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = round((time.time() - t0) / 60, 1)
    hist = [h for h in trainer.state.log_history if "reward" in h]
    pm.save_pretrained(out_dir / "adapter")
    if (Path(base_dir) / "projector.pt").exists():
        shutil.copy(Path(base_dir) / "projector.pt", out_dir / "projector.pt")  # projector giữ nguyên từ E3
    (out_dir / "run.json").write_text(json.dumps({
        "trainable_params": n_train, "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "train_minutes": minutes, "lr": lr, "max_steps": max_steps, "log_history": hist}))
    (out_dir / "hacking.json").write_text(json.dumps(hacking_summary(rollouts), indent=1, ensure_ascii=False))
    del trainer, pm
    R.free()
    return out_dir


def smoke(splits):
    """2 bước SFT + 2 bước GRPO + suy luận 8 ảnh, chỉ để bắt lỗi API/dtype trước khi chạy dài."""
    model, proc = V.load_model()
    pm, _, _ = V.apply_setup(model, train_projector=True, vit_lora=True)
    base = WORK / "runs" / "smoke_base"
    V.train(pm, proc, splits["train"][:8], base, True, max_steps=2, batch_size=2, grad_accum=1, grad_ckpt=True)
    del model, pm
    R.free()
    out = grpo(base, WORK / "runs" / "smoke_e5", splits["train"][:16], 5e-6, max_steps=2, n_prompts=4)
    log("rollouts:", V.read_jsonl(out / "rollouts.jsonl")[:3])
    log("hacking:", (out / "hacking.json").read_text()[:400])
    pm, proc = V.load_trained(out)
    rows = V.predict(pm, proc, splits["test"][:8], WORK / "smoke_e5.jsonl", batch_size=8)
    log("SMOKE OK", [r["raw"][:80] for r in rows[:2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--train-base", action="store_true", help="train E3 nếu chưa có adapter")
    ap.add_argument("--baseline", action="store_true", help="chờ E3 rồi đo F1 của chính adapter đó")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    EV.mkdir(parents=True, exist_ok=True)
    splits = {s: V.load_split(s) for s in ("train", "val", "test")}
    if a.smoke:
        return smoke(splits)
    base = WORK / "runs" / "e3"
    if a.train_base:
        R.run("e3", R.SETUPS["e3"], splits["train"])
    else:
        wait_for(base / "run.json")
    if a.baseline:  # F1 của adapter E3 dùng làm điểm xuất phát của GRPO (huấn luyện lại, có thể lệch nhẹ so với e3.json)
        R.evaluate("e3_retrain", "E3 (huấn luyện lại, điểm xuất phát của E5)", base, splits)
    key = "e5" if a.lr == 5e-6 else f"e5_lr{a.lr:g}"
    out = grpo(base, WORK / "runs" / key, splits["train"], a.lr, a.steps, n_prompts=2 * a.steps)
    R.evaluate(key, f"E5 GRPO lr={a.lr:g}", out, splits)
    shutil.copy(out / "hacking.json", EV / f"{key}_hacking.json")
    shutil.copytree(EV, R.ROOT / "report" / "evidence", dirs_exist_ok=True)
    log("DONE", key)


if __name__ == "__main__":
    main()
