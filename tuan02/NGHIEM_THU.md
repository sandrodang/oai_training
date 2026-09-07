# ✅ NGHIỆM THU TUẦN 2 — tự chấm T7 19/09

## A · Bài tập

```bash
cd /home/namdp36/oai/tuan02
python3 -m pytest bai_tap/test_all.py -q -m "not slow"    # phải 31 xanh
python3 -m pytest bai_tap/test_all.py -q -m slow -s       # phải xanh, in BLEU
```

- [ ] `01_attention.py` — 6 test
- [ ] `02_transformer.py` — 5 test 🔴
- [ ] `03_bpe.py` — 8 test
- [ ] `04_decoding.py` — 4 test
- [ ] `05_train_mt.py` — 2 test nhanh + 1 slow
- [ ] **BLEU > 40** ở test slow (đáp án đạt ~80)

## B · Bảng đo phải tự làm

**B1 — Ảnh hưởng của cách giải mã** (cùng một model đã train):

| Cách giải mã | BLEU | Độ dài TB câu dịch |
|---|---|---|
| greedy | ___ | ___ |
| beam=4, lp=0.0 | ___ | ___ |
| beam=4, lp=0.7 | ___ | ___ |
| beam=8, lp=0.7 | ___ | ___ |

→ Kết luận: beam nào tốt nhất? beam=8 có tốt hơn beam=4 không? **Vì sao?**

**B2 — Ảnh hưởng của vocab BPE** (cùng model, cùng cách giải mã):

| num_merges | vocab | token/câu | BLEU |
|---|---|---|---|
| 0 (ký tự) | ___ | ___ | ___ |
| 100 | ___ | ___ | ___ |
| 200 | ___ | ___ | ___ |
| 400 | ___ | ___ | ___ |

→ Kết luận: vocab tối ưu là bao nhiêu? Vì sao **không phải càng to càng tốt**?

**B3 — Tied embeddings** (bật/tắt `tie_decoder_output`):

| | Tham số | dev loss | BLEU |
|---|---|---|---|
| tied | ___ | ___ | ___ |
| untied | ___ | ___ | ___ |

## C · Lý thuyết — 8 câu trong `TAI_LIEU.md § TỰ KIỂM TRA`

- [ ] 1. Sơ đồ DecoderLayer (vẽ tay, 3 sublayer + mask)
- [ ] 2. Vì sao chia √d_k
- [ ] 3. tgt_in / tgt_out, hậu quả khi lệch một ô
- [ ] 4. Pre-LN vs Post-LN
- [ ] 5. Tied embeddings buộc gì, giảm bao nhiêu
- [ ] 6. Beam không có length penalty + brevity penalty của BLEU cộng dồn thế nào
- [ ] 7. Vì sao beam quá lớn làm BLEU giảm
- [ ] 8. Vì sao BPE thắng tách-theo-từ với Kơtu (nêu ví dụ một từ)

## D · Kiểm tra "hiểu thật" — không có test nào bắt được

- [ ] **Cố tình phá causal mask** (đổi `tril()` thành `torch.ones`), train lại.
      Ghi lại: train loss ___ (sẽ ĐẸP hơn), BLEU ___ (sẽ SẬP).
      **Đây là bug nguy hiểm nhất của seq2seq** — bạn phải thấy tận mắt một lần.
- [ ] **Cố tình bỏ `* sqrt(d_model)`** trước positional encoding. BLEU đổi bao nhiêu?
- [ ] **Cố tình lệch tgt_in/tgt_out một ô.** Loss xuống thấp bất thường? BLEU?

---

## 🚦 Kết luận

- **≥ 90%** → sang Tuần 3 (dịch máy ít tài nguyên) đúng lịch.
- **60–90%** → sang tuần 3 nhưng kéo phần thiếu theo. Ưu tiên xong BT 02.
- **< 60%** → ⚠️ dừng lại. Tuần 3 xây thẳng trên BT 02 + BT 04; nợ ở đây trả rất đắt.

**Tuần 3 bắt đầu bằng:** Sennrich (2016) *Back-translation* — vũ khí chính khi dữ liệu ít.
