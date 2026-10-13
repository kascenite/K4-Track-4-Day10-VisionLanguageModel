"""Ảnh minh hoạ cho báo cáo: nhãn train, và GT vs dự đoán trên test. Chạy: python -m lab.figures [e1 e2 ...]"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import ImageDraw  # noqa: E402

from lab import vlm as V  # noqa: E402
from scripts.score import _gt_boxes, parse_output  # noqa: E402

OUT = V.ROOT / "work" / "figures"
COLORS = {"pothole": "red", "crack": "yellow", "manhole": "cyan"}


def draw(e, boxes, color=None, img=None):
    img = img or V.load_image(e)
    d = ImageDraw.Draw(img)
    W, H = img.size
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        d.rectangle([x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H],
                    outline=color or COLORS.get(b["label"], "white"), width=2)
    return img


def train_samples():
    rows = V.load_split("train")[::240][:6]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    for ax, e in zip(axs.flat, rows):
        ax.imshow(draw(e, _gt_boxes(e)))
        ax.set_title(e["image_id"][:40], fontsize=8)
        ax.axis("off")
    fig.suptitle("Nhãn train (đỏ=pothole, vàng=crack, xanh=manhole)")
    plt.tight_layout()
    plt.savefig(OUT / "train_samples.png", dpi=80)


def pred_vs_gt(key):
    test = V.load_split("test")
    preds = {r["image_id"]: r["raw"] for r in V.read_jsonl(V.ROOT / "work" / f"{key}.jsonl")}
    rows = [e for e in test if e["objects"]][::25][:8]
    fig, axs = plt.subplots(2, 4, figsize=(20, 6))
    for ax, e in zip(axs.flat, rows):
        p = parse_output(preds.get(e["image_id"])) or []
        img = draw(e, _gt_boxes(e), color="lime")
        ax.imshow(draw(e, p, img=img))
        ax.set_title(f"{e['image_id'][:30]}  gt={len(e['objects'])} pred={len(p)}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{key.upper()} trên test: xanh lá = GT, màu theo lớp = dự đoán")
    plt.tight_layout()
    plt.savefig(OUT / f"{key.replace('/', '_')}_pred_vs_gt.png", dpi=80)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    train_samples()
    for k in sys.argv[1:]:
        pred_vs_gt(k)
    print("saved", sorted(p.name for p in OUT.iterdir()))
