# TUẦN 2 — TÀI LIỆU ĐỌC
### Tổng ~6,5h. Đã cắt bớt so với bản kế hoạch gốc (lý do ghi ở cuối).

---

## §1 · TRANSFORMER — TRỰC GIÁC RỒI ĐẾN CODE · 4h · N1–N2 🔴🔴

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 1.1 | **The Illustrated Transformer** — Jay Alammar<br>https://jalammar.github.io/illustrated-transformer/ | Toàn bộ. Đọc để LẤY TRỰC GIÁC, đừng cố nhớ chi tiết | 1h |
| 1.2 | **The Annotated Transformer** — Harvard NLP<br>http://nlp.seas.harvard.edu/annotated-transformer/ | **Toàn bộ phần Model Architecture** (đến hết Full Model). Đọc CODE TỪNG DÒNG, gõ lại vào notebook riêng | **3h** 🔴🔴 |

> **1.2 là tài liệu quan trọng nhất cả tuần.** Nó chính là bài báo Vaswani 2017 nhưng
> mỗi công thức đi kèm code PyTorch chạy được. Đọc xong nó, BT 01 và BT 02 trở thành
> việc gõ lại có hiểu — không đọc thì bạn sẽ vật lộn 6 tiếng.
>
> ⚠️ Annotated Transformer dùng **Post-LN** (`x = norm(x + sublayer(x))`), còn bài tập
> yêu cầu **Pre-LN** (`x = x + sublayer(norm(x))`). Khác biệt nhỏ về code, lớn về độ ổn định.
> Lý do chọn Pre-LN ghi trong docstring `02_transformer.py`.

**Phải rút ra được**
- Attention là **tra cứu mềm**: query so với mọi key, softmax thành trọng số, lấy tổ hợp value.
- **Vì sao chia √d_k**: q·k là tổng d_k tích số, phương sai tỉ lệ d_k. Không chia → softmax
  bão hoà → gradient ≈ 0. Bug âm thầm: model vẫn chạy, chỉ học rất kém.
- **Multi-head** không phải để "mạnh hơn" mà để nhìn nhiều **kiểu quan hệ** song song
  trong các không gian con khác nhau.
- **Causal mask** làm decoder không nhìn tương lai. Sai chỗ này → train loss đẹp, dịch ra rác.
- **Cross-attention** là chỗ duy nhất decoder chạm vào encoder.
- **Teacher forcing** và **exposure bias**: lúc train decoder luôn được đưa token ĐÚNG,
  lúc dịch nó phải ăn chính đầu ra của mình. Chênh lệch này là lý do beam search có ích.

---

## §2 · ATTENTION — NGUỒN GỐC · 1,25h · N3–N4

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 2.1 | **d2l.ai** — https://d2l.ai/chapter_attention-mechanisms-and-transformers/ | Mục **11.1–11.3** (queries/keys/values, scoring functions) + **11.5** (multi-head) | 45' |
| 2.2 | Bahdanau et al. (2015) — *Neural MT by Jointly Learning to Align and Translate*<br>https://arxiv.org/abs/1409.0473 | **Chỉ mục 3** (mô hình alignment) + Hình 3 (ma trận alignment) | 30' |

**Phải rút ra được**
- Bahdanau (**additive**: `v^T tanh(W_q q + W_k k)`) vs Luong/Transformer (**multiplicative**:
  `q·k`). Multiplicative nhanh hơn nhiều vì chỉ là phép nhân ma trận — đó là lý do
  Transformer chọn nó, và cũng là lý do phải chia √d_k.
- Hình 3 của Bahdanau (ma trận alignment Anh–Pháp) cho thấy attention **học được thứ tự đảo**
  mà không ai dạy. Đó chính xác là thứ model của bạn phải học ở corpus Kơtu (SOV→SVO).

> ✂️ **Đã cắt Vaswani 2017 và Sutskever 2014.** Annotated Transformer *là* Vaswani có chú giải;
> Sutskever là seq2seq RNN 2014, không còn gì dùng được cho bạn.

---

## §3 · BPE & SUBWORD · 45' · N5 🔴

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 3.1 | Sennrich et al. (2016) — *Neural MT of Rare Words with Subword Units*<br>https://arxiv.org/abs/1508.07909 | **Mục 3.2** (thuật toán BPE + ví dụ `low/lower/newest/widest`) và mục 1 (động cơ) | 45' |

**Phải rút ra được**
- BPE giải bài toán **từ hiếm / ngoài từ điển**: thay vì `<unk>`, từ lạ được tách thành
  các mảnh đã biết. Với ngôn ngữ **chắp dính** như Kơtu (hay Ba Na), hậu tố `-ne`, `-ma`
  trở thành đơn vị riêng và model học được quy luật thay vì học thuộc từng dạng từ.
- **Kích thước vocab là siêu tham số phải quét, không phải càng to càng tốt.**
  Bạn sẽ tự đo điều này: gộp quá tay sẽ **phá huỷ** chính các đơn vị chung mà BPE tạo ra
  (xem `test_discovers_shared_est_subword`).
- Dữ liệu càng ít → vocab càng nên **nhỏ** (mỗi token cần đủ ví dụ để học embedding).

---

## §4 · TÙY CHỌN — nếu tràn giờ

Nếu BT 02 ngốn hết thời gian, **được phép** thay BT 03 (tự viết BPE) bằng `sentencepiece`
(đã cài sẵn, `0.2.2`):

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train(input="data/train_src.txt", model_prefix="sp",
                               vocab_size=300, model_type="bpe")
sp = spm.SentencePieceProcessor(model_file="sp.model")
sp.encode("hling rong gu-ne-ma", out_type=str)
```

Nhưng **vẫn phải làm phần khảo sát**: quét `vocab_size ∈ {100, 200, 400}`, đo BLEU trên dev.
Hiểu đánh đổi quan trọng hơn tự viết thuật toán.

---

## 🧪 TỰ KIỂM TRA — viết ra giấy, không nhìn tài liệu

1. Vẽ sơ đồ **một DecoderLayer**, ghi rõ 3 sublayer và mask nào áp vào đâu.
2. Vì sao phải chia `√d_k`? Chuyện gì xảy ra với `d_k = 512` nếu quên?
3. `tgt = [BOS, a, b, EOS]` → `tgt_in` và `tgt_out` là gì? Lệch một ô thì model học được gì?
4. Pre-LN khác Post-LN ở đâu? Vì sao Pre-LN không cần warmup dài?
5. **Tied embeddings** buộc những ma trận nào? Giảm bao nhiêu tham số? Vì sao giúp khi ít dữ liệu?
6. Vì sao beam search **không có** length penalty lại thiên vị câu ngắn? Nối điều đó với
   brevity penalty của BLEU (Tuần 1 §2) — hai lỗi này cộng dồn thế nào?
7. Vì sao **beam quá lớn** có thể làm BLEU **giảm**?
8. Với corpus Kơtu, vì sao BPE thắng tách-theo-từ? Nêu ví dụ cụ thể một từ.

---

## 📚 Để dành Tuần 3 (đừng đọc tuần này)

- Sennrich (2016) **Back-translation** · https://arxiv.org/abs/1511.06709 🔴
- Provilkov (2020) **BPE-Dropout** · https://arxiv.org/abs/1910.13267
- Ott (2018) *Scaling NMT* — checkpoint averaging · https://arxiv.org/abs/1806.00187
- **J&M SLP3 Ch.13** Machine Translation · https://web.stanford.edu/~jurafsky/slp3/
- Koehn *SMT* Ch.4 — IBM Model 1 (phương án SMT khi dữ liệu cực ít)
