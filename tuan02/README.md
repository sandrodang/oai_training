# TUẦN 2 · 13–19/09/2026
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

## 📅 LỊCH 7 NGÀY (~15 giờ)

| Ngày | | Nội dung | Giờ |
|---|---|---|---|
| **N1** | CN 13/09 | Đọc *Illustrated Transformer* (1h) + nửa đầu *Annotated Transformer* (1,5h) | 2,5h |
| **N2** | T2 14/09 | Nốt *Annotated* (1,5h) · **BT 01** attention | 3h |
| **N3** | T3 15/09 | d2l attention (45') · **BT 02** transformer phần 1 | 2,25h |
| **N4** | T4 16/09 | **BT 02** phần 2 · Bahdanau §3 (30') | 2h |
| **N5** | T5 17/09 | Sennrich BPE (45') · **BT 03** BPE | 1,75h |
| **N6** | T6 18/09 | **BT 04** greedy + beam search | 1,5h |
| **N7** | T7 19/09 | **BT 05** huấn luyện đầu-cuối · nghiệm thu | 2,5h |

**Tổng ~15,5h.** Nếu chỉ có 13h: bỏ Bahdanau (30') và d2l (45'), rút BT 04 xuống greedy +
beam đơn giản không có length penalty (−30'). **Đừng cắt BT 02** — nó là xương sống.

> ⚠️ Bài tập nặng hơn tuần 1. BT 02 thực tế **2,5–3,5h ở lần đầu**. Nếu N3–N4 tràn giờ,
> lấy giờ từ N5 (BPE có thể dùng `sentencepiece` thay vì tự viết, xem `TAI_LIEU.md §4`).

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
│   └── test_all.py        26 test (25 nhanh + 1 slow)
├── dap_an/              ⛔ chỉ mở sau khi đã thử ≥ 45 phút
└── NGHIEM_THU.md
```

---

## ▶️ Bắt đầu

```bash
cd /home/namdp36/oai/tuan02
python3 -m pytest bai_tap/test_all.py -q -m "not slow"     # đỏ 25/25 — đúng như mong đợi
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

- [ ] `pytest -m "not slow"` **25 xanh**
- [ ] `pytest -m slow` xanh, **BLEU > 40**
- [ ] Vẽ được sơ đồ decoder có causal mask + cross-attention, **không nhìn tài liệu**
- [ ] Giải thích được vì sao **beam quá lớn có thể làm BLEU giảm**
- [ ] Có bảng đo: greedy vs beam=4 vs beam=8, và vocab BPE 100/200/400
