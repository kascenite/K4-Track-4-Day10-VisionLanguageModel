Họ tên: Lương Sỹ Khánh
`dataset_version`: d10-21e4070301

> Mọi số trong bảng chạy với ViT fp16 đúng đề. Lần đầu E0/E1 vô tình lượng tử hoá ViT 4-bit (Câu 5); bằng chứng cũ giữ ở `report/evidence/extra/vit4bit_e*.json`. E2, E3 chạy trên Colab T4; E0, E1, E4, E5 chạy trên Kaggle T4 (Colab hết quota), nên cột "Phút" chỉ so được tương đối.

## Bảng tổng hợp (từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0,085 | 0,059–0,113 | 0,81 | 0,130 | 0,029 | 0,169 |
| E1 | LoRA LLM | 33.030.144 | 6,33 | 47,3 | 0,276 | 0,240–0,318 | 1,10 | 0,182 | 0,209 | 0,498 |
| E2 | + projector | 142.135.296 | 7,93 | 55,6 | 0,357 | 0,316–0,397 | 1,07 | 0,216 | 0,283 | 0,625 |
| E3 | + LoRA ViT | 148.426.752 | 8,10 | 65,3 | **0,436** | 0,393–0,480 | 0,98 | 0,322 | 0,353 | 0,687 |
| E4 | E3, `data_size` 0,1 (142 ảnh) | 148.426.752 | 7,77 | 48,9 | 0,317 | 0,279–0,352 | 1,58 | 0,218 | 0,226 | 0,614 |
| E4 | E3, `data_size` 1,0 (= E3) | 148.426.752 | 8,10 | 65,3 | 0,436 | 0,393–0,480 | 0,98 | 0,322 | 0,353 | 0,687 |

Huấn luyện: QLoRA NF4, r = 16, α = 32, dropout 0,05, lr 2e-4 (warmup 5%, cosine), batch 2 × tích luỹ 4, 300 bước, fp16 + GradScaler, gradient checkpointing. GPU T4 16 GB.

## 10 câu hỏi

### Câu 1 — Số tham số ViT, projector, LLM; E1/E2/E3 cập nhật khối nào?
Dựng trên `meta` device: tổng 4,438B. **ViT** 415,3M (24 block + patch embedding 306,2M; `visual.merger` 27,3M; `deepstack_merger_list` 81,8M). **LLM** 4,022B, gồm embedding 389,0M dùng chung với `lm_head`.
- E1: LoRA trên 252 module LLM (36 lớp × q, k, v, o, gate, up, down) = 33.030.144.
- E2: thêm huấn luyện đầy đủ merger + deepstack: 142.135.296 − 33.030.144 = 109.105.152 ≈ 27,3M + 81,8M.
- E3: thêm LoRA trên 96 module ViT (24 block × qkv, proj, fc1, fc2), +6.291.456, tổng 348 module.

### Câu 2 — E0: `parse_valid_rate` và hai lỗi phổ biến?
`parse_valid_rate` = 0,910 (28/311 output hỏng, 29 ảnh trả `[]`). Model gốc dùng đúng thang 0–1000. Recall 0,063. Hai lỗi:
1. **Bịa box, nhất là crack:** 183 box crack, chỉ 7 đúng (crack F1 0,029). Model cắt một vết nứt thành nhiều ô liền kề: `[{"label": "crack", "bbox_2d": [393, 420, 475, 470]}, {"label": "crack", "bbox_2d": [475, 420, 550, 470]}, …`.
2. **JSON hỏng:** dải box lặp chạm `max_new_tokens=256` nên bị cắt; hoặc sai cú pháp, ví dụ `[890, 700, 1000, 776)}]`.

### Câu 3 — Chấm theo pixel thay vì thang 0–1000 thì sao?
Chấm lại file dự đoán E3 bằng `--coord pixel` (`extra/e3_as_pixel.json`): F1 0,436 → **0,000**, `parse_valid_rate` vẫn 1,0 (E1 ViT 4-bit cũng 0,253 → 0,000). Số 0–1000 bị coi là pixel rồi đổi về thang 1000: x nhân 1,56, y nhân 2,78, box lệch hẳn nên IoU < 0,5 với mọi GT. JSON vẫn hợp lệ nên lỗi "im lặng", chỉ lộ qua F1 ≈ 0 khi valid ≈ 1. Qwen3-VL dùng toạ độ tương đối vì ảnh được resize động; toạ độ này không phụ thuộc độ phân giải và là quy ước model học khi pretrain.

### Câu 4 — Vì sao mask loss chỉ trên câu trả lời?
Mỗi ví dụ ~400 token: 220 token ảnh, ~130 token template + prompt, chỉ ~40–70 token đáp án. Tôi giải mã các vị trí `labels != -100`: đúng bằng JSON đáp án + `<|im_end|>\n`. Nếu tính loss trên cả chuỗi, ~85% gradient dồn vào học thuộc prompt cố định, dự đoán `<|image_pad|>` vô nghĩa (chỉ là chỗ giữ chỗ cho embedding ảnh), tín hiệu grounding bị loãng và loss ban đầu (1,7) không phản ánh chất lượng đáp án.

### Câu 5 — Vì sao không train được projector nếu ViT 4-bit? Xử lý ra sao? Vì sao fp32?
Lỗi thật: E2 lần đầu báo `Attempting to unscale FP16 gradients` và chỉ có 87.609.344 tham số trainable (projector đếm được 54,6M, đúng nửa 109,1M vì `Params4bit` nén 2 trọng số/byte). Nguyên nhân: transformers 5.19 khớp `llm_int8_skip_modules` bằng `re.match` từ **đầu** tên, nên `"visual"` không khớp `model.visual.merger…` và cả ViT bị lượng tử hoá. Trọng số 4-bit là `uint8` nén, không có gradient đúng nghĩa; chỉ train được adapter bên trên. Sửa: `llm_int8_skip_modules=["model.visual", "lm_head"]`, `assert` không còn `Linear4bit` trong `visual`, đưa merger/deepstack về fp32 trước khi bật `requires_grad` → 142.135.296 tham số. Phải fp32 vì GradScaler không unscale được gradient fp16, và AdamW với lr 2e-4 trên fp16 làm tròn mất các bước cập nhật nhỏ (LoRA cũng được PEFT giữ fp32). Chạy lại E0/E1 với ViT fp16: E0 0,083 → 0,085, E1 0,253 → 0,276, đều trong CI. Lượng tử hoá ViT đóng băng gần như không hại; chỉ hại khi cần huấn luyện projector/ViT.

### Câu 6 — Split theo phiên quay: vì sao, và số liệu mỗi split?
Ảnh là frame cắt từ video, hai frame liền nhau gần như trùng khít. Chia ngẫu nhiên thì ảnh test hầu như có "song sinh" trong train, F1 đo trí nhớ chứ không đo tổng quát hoá. `prepare_data.py` gom theo phiên (ngày + giờ trong tên file) và `assert` không phiên nào ở hai split.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
| train | 1.422 | 1.074 | 1.750 | 691 |
| val | 276 (2 phiên) | 66 | 470 | 103 |
| test | 311 (3 phiên) | 121 | 299 | 163 |

Phân bố lệch (val chỉ 66 pothole) nên phải đọc F1 theo lớp.

### Câu 7 — E1/E2/E3: mở projector và LoRA ViT có đáng không?
- **E1 → E2:** +109M tham số, VRAM +1,6 GB (fp32 + 2 trạng thái AdamW ≈ 1,3 GB + gradient), F1 +0,081. CI chỉ chạm nhau ở 0,316–0,318 nên khá chắc.
- **E2 → E3:** chỉ +6,3M tham số nhưng thời gian +17% và VRAM 8,10 GB vì gradient lan ngược qua 24 block ViT. F1 +0,079; pothole tăng nhiều nhất (0,216 → 0,322), FP/ảnh 1,07 → 0,98.
E3 đáng nhất: F1 cao nhất, VRAM gần E2, chỉ thêm ~10 phút. Box nhỏ (trung vị 0,7% khung hình) cần đặc trưng thị giác tốt hơn, chỉ chỉnh ViT mới làm được. Thời gian giữa E1 (Kaggle) và E2/E3 (Colab) chỉ so tương đối.

### Câu 8 — CI95 trên 311 ảnh giới hạn kết luận ra sao?
CI bootstrap rộng ~±0,04 (E3 0,393–0,480; E2 0,316–0,397). CI của E2 và E3 chồng nhau ở 0,393–0,397; đây là hai CI riêng, không phải kiểm định ghép cặp, nên **chưa khẳng định chắc E3 > E2** (cần bootstrap ghép cặp). E0 tách hẳn; E1/E2 gần như không chồng; E4 0,1 và E3 tách hẳn (0,352 < 0,393). Test chỉ có 3 phiên quay nên CI còn đánh giá thấp bất định khi sang đường hoặc ánh sáng khác.

### Câu 9 — Lớp nào kém nhất, vì sao? (ảnh minh hoạ)
Ở E3, **pothole** thấp nhất (F1 0,322: TP 33, FP 51, FN 88), rồi **crack** (0,353: TP 111, FP 219, FN 188); manhole tốt nhất (0,687: TP 103, FP 34, FN 60).
- Manhole có hình rõ, ranh giới sắc nên dễ đạt IoU ≥ 0,5.
- Crack mảnh, dài, ranh giới mơ hồ; một vết nứt có thể gán là một box dài hoặc nhiều box ngắn nên IoU 0,5 rất khắt khe. Giả thuyết (chưa đo): phần lớn 219 FP là box lệch hoặc chia khác người gán nhãn.
- Pothole nhỏ, xa camera, dễ nhầm mảng vá, vũng nước, bóng râm nên bỏ sót nhiều (FN 88/121). Giả thuyết (chưa kiểm tra theo phiên): train tập trung ở vài phiên.

Ảnh: `report/figures/e3_pred_vs_gt.png` (xanh lá = GT, đỏ pothole, vàng crack, lục lam manhole), cùng `e0/e1/e2_pred_vs_gt.png`, `train_samples.png`.

![E3 trên test](report/figures/e3_pred_vs_gt.png)

### Câu 10 — E4: biến nào, kết quả, có đáng tin không?
Biến `data_size`: 10% train (142 ảnh, `random.Random(42).sample`) so với 100% (= E3), giữ nguyên 300 bước nên bản 10% xem mỗi ảnh ~17 lần thay vì ~1,7.

| data_size | F1 | CI95 | P | R | FP/ảnh | loss cuối |
|---|---|---|---|---|---|---|
| 0,1 | 0,317 | 0,279–0,352 | 0,291 | 0,346 | 1,58 | 0,019 |
| 1,0 | 0,436 | 0,393–0,480 | 0,448 | 0,424 | 0,98 | 0,484 |

F1 giảm 0,119 và hai CI tách hẳn nhau nên kết luận "nhiều dữ liệu hơn tốt hơn" đáng tin. Bản 10% quá khớp rõ (loss 0,019, học thuộc 142 ảnh): sinh nhiều box hơn, precision 0,448 → 0,291. Dù vậy 142 ảnh đã đạt 0,317, hơn cả E1 trên toàn bộ dữ liệu (0,276), tức chất lượng setup quan trọng ngang lượng dữ liệu. Giới hạn: một seed, một mẫu 10%.

## E5 (tuỳ chọn): GRPO với reward IoU

- **Reward** (`lab/reward.py`): đúng công thức GUIDE (0 nếu JSON hỏng; 1 nếu GT và dự đoán đều rỗng; ngoài ra 0,2 + 0,8·ΣIoU / max(n_gt, n_pred, 1), ghép tham lam cùng lớp, IoU ≥ 0,5). `python -m pytest -q lab/test_reward.py`: **7 passed** (hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng, box khổng lồ không được thưởng, IoU một phần, JSON bọc ```json).
- **GRPO** (`lab/grpo.py`, `trl.GRPOTrainer` 1.9.2): khởi tạo từ adapter E3, `num_generations=4`, `max_completion_length=128`, 60 bước, `beta=0`, `use_vllm=False`; chỉ LoRA (39,3M) được cập nhật, projector đóng băng. Adapter E3 gốc không còn nên huấn luyện lại E3 trên Kaggle (57,9 phút) làm điểm xuất phát. Chạy `lr=5e-6` (đúng đề) và `lr=5e-5` (đối chứng, vì lr 5e-6 có thể gần như không đổi LoRA trong 60 bước).

| Run | F1 | CI95 | P | R | FP/ảnh | phút |
|---|---|---|---|---|---|---|
| E3 gốc | 0,436 | 0,393–0,480 | 0,448 | 0,424 | 0,98 | 65,3 |
| E3 huấn luyện lại (xuất phát) | 0,440 | 0,399–0,483 | 0,452 | 0,429 | 0,97 | 57,9 |
| **E5 lr=5e-6** | 0,437 | 0,396–0,480 | 0,473 | 0,407 | 0,85 | 30,2 |
| E5 lr=5e-5 | 0,432 | 0,388–0,471 | 0,483 | 0,391 | 0,78 | 25,2 |

**Reward hacking?** Không có dạng suy biến. Log ở `extra/e5_rollouts.jsonl`, tổng hợp ở `e5_hacking.json`: trả `[]` 0% ở mọi cửa sổ; box lớn hơn 50% ảnh tối đa 3,9% rồi về 0%; tỉ lệ JSON hợp lệ khi rollout tăng (92,5% → 98,8% và 97,5% → 100%). Chỉ có dịch chuyển nhẹ sang thận trọng: số box dự đoán luôn thấp hơn GT (1,3–2,05 so với 2,1–2,8); trên test FP/ảnh giảm 0,97 → 0,85 → 0,78 và precision tăng, nhưng recall giảm 0,429 → 0,407 → 0,391, nên F1 gần như đứng yên.

**So với SFT:** F1 0,440 → 0,437 (lr 5e-6) và 0,432 (lr 5e-5); chênh tối đa 0,008, CI gần như trùng hoàn toàn, nên **không vượt CI95, không kết luận được GRPO tốt hơn hay kém hơn SFT**. Reward trung bình không tăng rõ sau 60 bước (mỗi cửa sổ là các ảnh khác nhau). Giới hạn: một seed, chưa có bootstrap ghép cặp (không có dự đoán từng ảnh trên máy này).

## Nhìn lại

- Công cụ chấm không bắt được lỗi ViT bị lượng tử hoá (JSON hợp lệ, F1 hợp lý); chỉ lộ nhờ E2 crash và số tham số lệch đúng nửa. Bài học: đối chiếu số tham số với kỳ vọng và `assert` kiểu module sau khi nạp model, vì ngữ nghĩa tham số API có thể đổi giữa các phiên bản.
- Colab: torchao 0.10 làm PEFT 0.21 lỗi trên `nn.Linear` thường (lộ ra ở E3), phải gỡ; session bị ngắt nhiều lần rồi hết quota nên chuyển sang Kaggle. `lab/run_all.py` ghi dự đoán sau mỗi batch và bỏ qua bước đã xong nên tiếp tục được.
- Chưa chắc: CI riêng từng model không phải kiểm định so sánh; muốn khẳng định E3 > E2 hay E5 ≠ E3 cần bootstrap ghép cặp.
