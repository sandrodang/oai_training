# TẦNG 1 TOOLKIT — BỘ ĐỒ NGHỀ HUẤN LUYỆN
### Phần Tầng 1 còn nợ, đóng gói thành code chạy được

> **Vì sao có gói này.** Kế hoạch Phần 4 định 10h cho Tầng 1 (Học sâu cốt lõi),
> nhưng Tuần 1 chỉ giao 1,7h và Tuần 2 giao 0h — tôi viết Tuần 2 thành 100% Tầng 2.
> Gốc rễ sâu hơn: tổng 6 tầng là **56h** trong khi tổng ngân sách đọc là **47h**,
> tức ánh xạ tầng→tuần bất khả thi ngay từ đầu.
>
> Gói này trả nợ phần đó. **Không phải một tuần riêng** — Tầng 1 không phải một chủ đề,
> nó là bộ đồ nghề. Mỗi module được xếp vào **tuần thực sự dùng nó**.

---

## 📅 LỊCH — làm module nào ở tuần nào

| Module | Nội dung | Làm ở | Vì dùng cho |
|---|---|---|---|
| **07** `norm_init_gradflow` 🔴 | LayerNorm vs BatchNorm · Xavier/He · zero-γ · grad flow Pre/Post-LN | **Tuần 2** | **lớp giải thích nằm dưới Pre-LN** của BT 02 tuần 2 — không có nó thì "Pre-LN ổn định hơn" chỉ là câu học thuộc |
| **01** `weight_averaging` | EMA · SWA · checkpoint averaging | **Tuần 3** | model dịch máy — ckpt averaging là kỹ thuật 🔴 của NMT |
| **02** `grad_tricks` | gradient accumulation · checkpointing · thứ tự AMP | **Tuần 3** | batch lớn cho MT trên GPU 16GB |
| **05a** `losses` (focal) | focal loss | **Tuần 4** | phân loại ảnh mất cân bằng |
| **05b** `losses` (dice/tversky/contrastive) | dice · tversky · contrastive · triplet | **Tuần 5** | segmentation & ghép ảnh jigsaw |
| **03** `finetune_lr` | LLRD · gradual unfreezing · freeze BN | **Tuần 6** | fine-tune encoder cho phân loại |
| **04** `adversarial` | FGM · PGD · FreeLB | **Tuần 3** 🔴 | **FGM: +0,0121 OOF, dương 5/5 fold** — tấn công embedding nên dùng được cho cả seq2seq |
| **06** `consistency_multitask` | R-Drop · consistency · uncertainty weighting | **Tuần 3** 🔴 | **R-Drop: +0,0047 OOF, 4/5 fold** — Wu (2021) vốn sinh ra cho NMT |

**Tải mỗi lần: ~45' đọc + ~45' code.** Tổng ~5h đọc + ~5h code rải trên 4 tuần.

> ⚠️ **M04 và M06 đã chuyển từ Tuần 6 về Tuần 3.** Chúng là hai kỹ thuật DUY NHẤT sống sót
> qua đo đạc ở vòng trường (2 trên 11 thứ đã thử), nên không được xếp vào tuần có TỔNG DUYỆT 1.
> Tuần 3 còn cho chúng một **bài thật (Ba Na) để đo ngay** thay vì học chay.

Bù lại đã cắt: **Tầng 4 từ 6h → 2h** (ML không ra thành bài riêng, chỉ là lớp phân loại
cuối) và **Tầng 3 từ 14h → 12h**. Net +2h trên cả phần còn lại của kế hoạch.

---

## ▶️ Dùng

```bash
cd /home/namdp36/oai/tang1_toolkit
python3 -m pytest bai_tap/test_all.py -q            # 26 test, đỏ hết lúc đầu
python3 -m pytest bai_tap/test_all.py -q -k Weight  # chỉ module 01
```

Mỗi module độc lập — làm đúng module được xếp cho tuần đó, đừng làm hết một lượt.

---

## 📦 Nội dung

```
tang1_toolkit/
├── TAI_LIEU.md      ← đọc gì cho từng module
├── bai_tap/
│   ├── 01_weight_averaging.py       EMA · SWA · average_state_dicts · update_bn
│   ├── 02_grad_tricks.py            accumulate_steps · train_step_accum · checkpointing
│   ├── 03_finetune_lr.py            param_groups · LLRD · gradual unfreeze · freeze BN
│   ├── 04_adversarial.py            FGM · PGD · fgm_train_step
│   ├── 05_losses.py                 focal · dice · tversky · contrastive · triplet
│   ├── 06_consistency_multitask.py  R-Drop · consistency · UncertaintyWeighting
│   └── test_all.py                  26 test
└── dap_an/          ⛔ chỉ mở sau khi đã thử ≥ 30 phút mỗi hàm
```

---

## 🔑 Ba phép kiểm đáng giá nhất trong bộ này

**`test_accumulation_equals_full_batch`** — gradient tích luỹ phải **khớp chính xác**
gradient batch lớn. Lệch = bạn quên chia loss cho số micro-batch, tức vô tình
nhân LR lên N lần. Bug này không báo lỗi, chỉ làm model học tệ.

**`test_integer_buffers_take_last_not_mean`** — `num_batches_tracked` của BatchNorm là
số nguyên; trung bình nó là vô nghĩa. Và sau khi trung bình trọng số, **running stats
của BatchNorm sai** vì bộ trọng số mới chưa từng "thấy" dữ liệu → phải chạy `update_bn`.
Đây là lý do checkpoint averaging đôi khi làm điểm **tụt** mà không ai hiểu vì sao.

**`test_consistency_detaches_clean_branch`** — nhánh sạch phải `.detach()`. Nếu không,
model "gian lận" bằng cách kéo dự đoán sạch về phía nhiễu thay vì ngược lại.

---

## ⚠️ Một cảnh báo về EMA/SWA/checkpoint averaging

Cả ba đều thuộc nhóm *"gần như luôn dương, chi phí ~0"* trong kế hoạch §2.5 — **nhưng**
nhớ bài học Tuần 1 BT05: *"gần như luôn dương"* không phải *"luôn dương"*.
Vẫn phải đo bằng held-out, và mức tăng đo trên tập đã dùng để chọn là **lạc quan**.
