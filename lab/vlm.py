"""Mã của học viên cho Lab Day 10: nhãn, suy luận, setup LoRA, huấn luyện (GUIDE bước 2–8)."""
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATA = ROOT / "data"

PROMPT_NORM = (
    "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole. "
    "Answer ONLY with a JSON list, no other text: "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(0,0 = top-left, 1000,1000 = bottom-right). Answer [] if there is no damage."
)
PROMPT_PIXEL = (
    "Detect all road damage in this dashcam image (640x360 pixels). Classes: pothole, crack, manhole. "
    "Answer ONLY with a JSON list, no other text: "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integer pixel positions (x in 0-640, y in 0-360, 0,0 = top-left). "
    "Answer [] if there is no damage."
)


def prompt_for(coord):
    return PROMPT_NORM if coord == "norm1000" else PROMPT_PIXEL


# ---------- Bước 2: nhãn ----------

def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    x1, y1 = (xc - w / 2) * sx, (yc - h / 2) * sy
    x2, y2 = (xc + w / 2) * sx, (yc + h / 2) * sy
    x1, x2 = (min(max(round(v), 0), sx) for v in (x1, x2))
    y1, y2 = (min(max(round(v), 0), sy) for v in (y1, y2))
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def load_split(split):
    return read_jsonl(DATA / "splits" / f"{split}.jsonl")


def load_image(entry):
    return Image.open(DATA / entry["image_path"]).convert("RGB")


# ---------- Bước 3: model ----------

def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                           # transformers 5.x so khớp tên từ ĐẦU (re.match) hoặc hậu tố: "visual" không
                           # khớp "model.visual.blocks.0...", phải dùng "model.visual" (prefix + ".")
                           llm_int8_skip_modules=["model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=q, dtype=torch.float16, device_map={"": 0})
    q4 = [n for n, m in model.named_modules() if "visual" in n and type(m).__name__ == "Linear4bit"]
    assert not q4, f"ViT bị lượng tử hoá 4-bit: {q4[:3]}"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def user_msg(img, prompt):
    return {"role": "user", "content": [{"type": "image", "image": img},
                                        {"type": "text", "text": prompt}]}


@torch.no_grad()
def predict(model, processor, entries, out_path, prompt=PROMPT_NORM, batch_size=4,
            max_new_tokens=256, log_every=10):
    model.eval()
    processor.tokenizer.padding_side = "left"
    out_path = Path(out_path)
    done = {}
    if out_path.exists():  # tiếp tục nếu session bị ngắt giữa chừng
        done = {r["image_id"]: r for r in read_jsonl(out_path)}
    todo = [e for e in entries if e["image_id"] not in done]
    rows = list(done.values())
    t0 = time.time()
    for bi in range(0, len(todo), batch_size):
        batch = todo[bi:bi + batch_size]
        msgs = [[user_msg(load_image(e), prompt)] for e in batch]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt",
                                               padding=True).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"image_id": e["image_id"], "raw": r} for e, r in zip(batch, raws)]
        write_jsonl(out_path, rows)
        if (bi // batch_size) % log_every == 0:
            print(f"{len(rows)}/{len(entries)}  {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------- Bước 4: dữ liệu huấn luyện ----------

def encode_example(processor, img, prompt, answer):
    user = user_msg(img, prompt)
    asst = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    f = processor.apply_chat_template([user, asst], add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert torch.equal(f["input_ids"][0, :n], p["input_ids"][0]), "prompt không phải tiền tố"
    labels = f["input_ids"].clone()
    labels[:, :n] = -100
    f = dict(f)
    f["labels"] = labels
    return f


SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids", "labels")


def collate(batch, pad_id):
    L = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for k in batch[0]:
        if k in SEQ_KEYS:
            fill = -100 if k == "labels" else (pad_id if k == "input_ids" else 0)
            rows = []
            for b in batch:
                x = b[k]
                pad = torch.full((x.shape[0], L - x.shape[1]), fill, dtype=x.dtype)
                rows.append(torch.cat([x, pad], dim=1))  # pad phải khi train
            out[k] = torch.cat(rows, 0)
        else:
            out[k] = torch.cat([b[k] for b in batch], 0)
    return out


# ---------- Bước 5: setup LoRA ----------

LLM_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJ_KEYS = ("visual.merger", "visual.deepstack_merger_list")


def apply_setup(model, train_projector, vit_lora, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    regex = f"({LLM_RE})|({VIT_RE})" if vit_lora else LLM_RE
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    print("matched LoRA modules:", len(matched))
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                          target_modules=regex, task_type="CAUSAL_LM"))
    if train_projector:
        for n, p in pm.named_parameters():
            if any(k in n for k in PROJ_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
    for p in pm.parameters():  # GradScaler cần grad fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"trainable params: {n_train:,}")
    return pm, len(matched), n_train


def projector_state(pm):
    return {n: p.detach().cpu() for n, p in pm.named_parameters()
            if p.requires_grad and any(k in n for k in PROJ_KEYS)}


# ---------- Bước 6: huấn luyện ----------

def train(pm, processor, entries, out_dir, train_projector, coord="norm1000", lr=2e-4,
          batch_size=2, grad_accum=4, max_steps=300, seed=42, save_every=100, log_every=10,
          grad_ckpt=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    processor.tokenizer.padding_side = "right"
    pad_id = processor.tokenizer.pad_token_id
    prompt = prompt_for(coord)
    pm.config.use_cache = False
    pm.train()
    params = [p for p in pm.parameters() if p.requires_grad]
    if not grad_ckpt:  # thử batch nặng nhất (nhiều box nhất) không checkpointing; OOM thì bật lại
        worst = sorted(entries, key=lambda e: -len(e["objects"]))[:batch_size]
        exs = [encode_example(processor, load_image(e), prompt, objects_to_answer(e["objects"], coord))
               for e in worst]
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                pm(**{k: v.to("cuda") for k, v in collate(exs, pad_id).items()}).loss.backward()
            # + trạng thái AdamW (2 bản fp32) chưa cấp phát lúc thử; giữ 10% dự phòng
            need = torch.cuda.max_memory_allocated() + 2 * sum(p.numel() * 4 for p in params)
            grad_ckpt = need > 0.9 * torch.cuda.get_device_properties(0).total_memory
        except torch.OutOfMemoryError:
            grad_ckpt = True
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()
    if grad_ckpt:
        pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpointing:", grad_ckpt, flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(0.05 * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / (max_steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    order, pos, losses, t0 = [], 0, [], time.time()

    def next_batch():
        nonlocal order, pos
        if pos + batch_size > len(order):
            order = list(range(len(entries)))
            random.shuffle(order)
            pos = 0
        idx = order[pos:pos + batch_size]
        pos += batch_size
        exs = [encode_example(processor, load_image(entries[i]), prompt,
                              objects_to_answer(entries[i]["objects"], coord)) for i in idx]
        return {k: v.to("cuda") for k, v in collate(exs, pad_id).items()}

    for step in range(1, max_steps + 1):
        tot = 0.0
        for _ in range(grad_accum):
            batch = next_batch()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = pm(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(tot)
        if step % log_every == 0 or step == 1:
            print(f"step {step}/{max_steps} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f}min vram {torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
                  flush=True)
        if step % save_every == 0 or step == max_steps:
            pm.save_pretrained(out_dir / "adapter")
            if train_projector:
                torch.save(projector_state(pm), out_dir / "projector.pt")
    run = {"trainable_params": sum(p.numel() for p in params),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1),
           "grad_ckpt": grad_ckpt, "losses": losses}
    (out_dir / "run.json").write_text(json.dumps(run))
    return run


# ---------- Bước 7: nạp lại để suy luận ----------

def load_trained(run_dir, model_id=MODEL_ID):
    from peft import PeftModel
    run_dir = Path(run_dir)
    model, processor = load_model(model_id)
    pm = PeftModel.from_pretrained(model, run_dir / "adapter")
    proj = run_dir / "projector.pt"
    if proj.exists():
        sd = torch.load(proj, map_location="cpu")
        res = pm.load_state_dict(sd, strict=False)
        bad = [k for k in sd if k in res.unexpected_keys]
        assert not bad, f"projector keys không khớp: {bad[:3]}"
        print(f"đã nạp lại {len(sd)} tensor projector")
    pm.eval()
    return pm, processor


def meta_args(run_dir):
    r = json.loads((Path(run_dir) / "run.json").read_text())
    return " ".join(f"--meta {k}={r[k]}" for k in ("trainable_params", "peak_vram_gb", "train_minutes"))
