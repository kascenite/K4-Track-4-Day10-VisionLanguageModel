"""Chẩn đoán: tham số/gradient nào không phải fp32 sau apply_setup (lỗi 'unscale FP16 gradients')."""
import collections

import torch

from lab import vlm as V

model, proc = V.load_model()
pm, _, _ = V.apply_setup(model, train_projector=True, vit_lora=False)
bad = [(n, p.dtype) for n, p in pm.named_parameters() if p.requires_grad and p.dtype != torch.float32]
print("trainable non-fp32 params:", len(bad), bad[:5])
print("trainable dtypes:", collections.Counter(str(p.dtype) for p in pm.parameters() if p.requires_grad))
proc.tokenizer.padding_side = "right"
e = V.load_split("train")[0]
ex = V.encode_example(proc, V.load_image(e), V.PROMPT_NORM, V.objects_to_answer(e["objects"]))
b = {k: v.cuda() for k, v in V.collate([ex], proc.tokenizer.pad_token_id).items()}
pm.config.use_cache = False
pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
pm.train()
with torch.autocast("cuda", dtype=torch.float16):
    loss = pm(**b).loss
loss.backward()
g = [(n, p.dtype, p.grad.dtype) for n, p in pm.named_parameters() if p.grad is not None and p.grad.dtype != torch.float32]
print("non-fp32 grads:", len(g), g[:5])
ids = collections.Counter(id(p) for p in pm.parameters())
print("params with grad but requires_grad False:",
      [n for n, p in pm.named_parameters() if p.grad is not None and not p.requires_grad][:5])
