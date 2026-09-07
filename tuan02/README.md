# TUẦN 2 · 15–21/09/2026
## TRANSFORMER & SEQ2SEQ — TỰ VIẾT TỪ ĐẦU

> Thuộc `KE_HOACH_OLPAI26.md` Phần 5. Lý thuyết: **Tầng 1 (phần còn lại) + Tầng 2 nửa đầu**.

---

## 🎯 Vì sao tuần này quan trọng

**Dịch máy xuất hiện ở 2/2 đề mẫu 2025** — SOLOAI (Trung→Việt) và VOAI CK (Ba Na→Việt,
trọng số BLEU **0.8**). Đó là kỹ thuật có xác suất ra cao nhất ở vòng miền, và bạn có
**0 kinh nghiệm seq2seq**. Tuần 2 dựng bộ khung, tuần 3 mới là các kỹ thuật ăn điểm.

Với ngôn ngữ như Ba Na, **không pretrained nào giúp được** — bạn buộc phải tự huấn luyện.
Nên "tự viết Transformer" ở đây không phải bài tập học thuật, nó là **kịch bản thi thật**.

---

## 📅 LỊCH 7 NGÀY · **4h/ngày = 28h** (ngân sách bản 3, xem kế hoạch §4B.1)

| Ngày | | Nội dung | Giờ |
|---|---|---|---|
| **N1** | T3 15/09 | 🔴 Đọc [§0.1–0.2 d2l: khởi tạo · ổn định số học · tối ưu](TAI_LIEU.md) (2,5h) · **BT 06** `norm_init_gradflow` phần **a+b** (1,5h) | 4h |
| **N2** | T4 16/09 | 🔴 Đọc [§0.3–0.4 Goodfellow backprop + AdamW](TAI_LIEU.md) (2,5h) · **BT 06** phần **c** Pre-LN vs Post-LN (1,5h) | 4h |
| **N3** | T5 17/09 | Đọc *Illustrated Transformer* (1h) + nửa đầu *Annotated* (1,5h) · **BT 01** attention (1,5h) | 4h |
| **N4** | T6 18/09 | Nốt *Annotated* (1,5h) · nốt **BT 01** · **BT 02** transformer phần 1 (2,5h) | 4h |
| **N5** | T7 19/09 | **BT 02** transformer phần 2 🔴 (2,75h) · Đọc §2 d2l attention + Bahdanau §3 (1,25h) | 4h |
| **N6** | CN 20/09 | Đọc §3 Sennrich BPE (45') · **BT 03** BPE (1,5h) · **BT 04** greedy + beam (1,75h) | 4h |
| **N7** | T2 21/09 | **BT 05** huấn luyện đầu-cuối (2h) · 📓 **khởi tạo sổ assert** 22 mục (1h) · nghiệm thu (1h) | 4h |

**Tổng 28h.** Đọc 11h (§0 5h + §1 4h + §2 1,25h + §3 0,75h) · cài đặt 12h · sổ assert 2h ·
thí nghiệm 2h · đệm 1h.

> 🔴 **N1–N2 là Tầng 1, không phải phần phụ.** `06_norm_init_gradflow` đo ra ba con số:
> BatchNorm lệch ~2,3 khi đổi batch (LayerNorm = 0) · He giữ phương sai còn Xavier **tắt** (7e-07)
> · gradient tầng đáy **Pre-LN / Post-LN ≈ 5·10⁵ lần**.
> Không có hai ngày này thì luật "dùng Pre-LN" ở BT 02 chỉ là câu **học thuộc**.

> ⚠️ BT 02 thực tế **2,5–3,5h ở lần đầu**. Nếu N4–N5 tràn, lấy giờ từ N6
> (BPE có thể dùng `sentencepiece` thay vì tự viết, xem `TAI_LIEU.md` §4) —
> **đừng lấy từ N1–N2 và đừng cắt BT 02.**

---

## 📂 Cấu trúc

```
tuan02/
├── README.md
├── TAI_LIEU.md          ← đọc gì, ở đâu, bao lâu, câu hỏi tự kiểm tra
├── data/
│   ├── make_toy_parallel.py   sinh corpus (đã chạy sẵn)
│   ├── train.tsv  3400 cặp    ← ngôn ngữ "Kơtu" bịa → tiếng Việt
│   ├── dev.tsv     400 cặp
│   └── test.tsv    400 cặp
├── bai_tap/
│   ├── 01_attention.py    scaled dot-product · causal mask · multi-head
│   ├── 02_transformer.py  🔴 PE · Pre-LN block · seq2seq · tied embeddings
│   ├── 03_bpe.py          BPE của Sennrich, từ số 0
│   ├── 04_decoding.py     greedy · beam search · length penalty
│   ├── 05_train_mt.py     collate · Noam · vòng lặp train
│   ├── 06_norm_init_gradflow.py  🔴 TẦNG 1 — LayerNorm vs BatchNorm · Xavier/He · grad Pre/Post-LN
│   │                              (làm TRƯỚC 02_transformer: đây là LÝ DO của luật Pre-LN)
│   └── test_all.py        32 test (31 nhanh + 1 slow)
├── dap_an/              ⛔ chỉ mở sau khi đã thử ≥ 45 phút
└── NGHIEM_THU.md
```

---

## ▶️ Bắt đầu

```bash
cd /home/namdp36/oai/tuan02
python3 -m pytest bai_tap/test_all.py -q -m "not slow"     # đỏ 25/25 — đúng như mong đợi
$EDITOR bai_tap/06_norm_init_gradflow.py   # làm cái này TRƯỚC (nền Tầng 1)
$EDITOR bai_tap/01_attention.py
```

Cuối tuần chạy bài huấn luyện thật (~40s CPU):
```bash
python3 -m pytest bai_tap/test_all.py -q -m slow -s
```

---

## 🧪 Về bộ dữ liệu "Kơtu → Việt"

Ngôn ngữ nhân tạo, sinh bởi `data/make_toy_parallel.py`, mô phỏng đúng bài VOAI 2025 TV1:

- **Trật tự SOV** trong khi tiếng Việt là **SVO** → model phải học **đảo thứ tự**,
  không dịch từng từ tại chỗ được.
- **Chắp dính**: thì/số/phủ định là hậu tố dán liền (`gu-ne-ma` = bắt + quá khứ + phủ định)
  → **BPE thắng rõ rệt** so với tách theo từ. Bạn sẽ đo được điều này.
- Câu **vô nghĩa về ngữ nghĩa** (*"cây lúa uống người cha"*) — **cố ý**. Model không thể
  dựa vào tiên nghiệm ngữ nghĩa, buộc phải học ánh xạ cấu trúc.

**Lợi thế lớn nhất: có trần đã biết.** Ánh xạ là xác định, nên Transformer đúng **phải**
đạt BLEU cao (đáp án: **80,5**). Model bạn ra BLEU thấp ⇒ **code sai**, không phải bài khó.
Dữ liệu thật không cho bạn tín hiệu gỡ lỗi này.

Muốn khó hơn: sửa `make_toy_parallel.py` — thêm từ đồng nghĩa, nhập nhằng, hoặc nhiễu.

---

## ⚖️ Luật chơi

1. **Cấm `nn.Transformer`, `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`**
   trong lời giải — test dùng chúng làm chuẩn đối chiếu.
2. **Cấm ChatGPT/Claude từ tuần này** (kế hoạch §5). Tự giới hạn **2.000 token/phiên**
   nếu có hỏi LLM, đúng như điều kiện phòng thi.
3. Mở `dap_an/` chỉ sau khi đã thử **≥ 45 phút** mỗi bài.
4. BT 01 và BT 02 **phải tự viết** — chúng là nền của cả tuần 3.

---

## ✅ Nghiệm thu → `NGHIEM_THU.md`

- [ ] `pytest -m "not slow"` **31 xanh** *(khung: 29 đỏ + 2 xanh sẵn — 2 test
      `builtin_transformer` là guard LUẬT CHƠI, phải xanh ngay từ đầu)*
- [ ] 🔴 `06_norm_init_gradflow` xanh. Nói được bằng SỐ: BatchNorm lệch bao nhiêu khi
      đổi batch (LayerNorm 0) · gradient tầng đáy Pre-LN/Post-LN gấp bao nhiêu lần
- [ ] `pytest -m slow` xanh, **BLEU > 40**
- [ ] Vẽ được sơ đồ decoder có causal mask + cross-attention, **không nhìn tài liệu**
- [ ] Giải thích được vì sao **beam quá lớn có thể làm BLEU giảm**
- [ ] Có bảng đo: greedy vs beam=4 vs beam=8, và vocab BPE 100/200/400
