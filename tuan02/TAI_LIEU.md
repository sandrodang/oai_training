# TUẦN 2 — TÀI LIỆU ĐỌC
### Tổng **11h** = Tầng 1 phần A (**5h**) + Tầng 2 đầu (**6h**: §1 4h · §2 1,25h · §3 0,75h).

> ⚠️ **Bản trước của file này thiếu hoàn toàn Tầng 1** — cả 3 mục đọc đều là Tầng 2,
> trong khi kế hoạch giao Tuần 2 gánh "Tầng 1 phần A". §0 dưới đây trả nợ phần đó.
> Nó **không phải chủ đề rời**: nó là lớp giải thích nằm DƯỚI Pre-LN mà §1 sẽ bắt bạn dùng.

---

## §0 · TẦNG 1 PHẦN A — NỀN CỦA CẢ TUẦN · 5h · N1 🔴🔴
### Đọc TRƯỚC Transformer. Đọc sau thì §1 chỉ còn là học thuộc.

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 0.1 | **d2l — Builders Guide** https://d2l.ai/chapter_builders-guide/index.html<br>+ **Ổn định số học & khởi tạo** https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html | tầng & khối tuỳ biến · khởi tạo tham số · vanishing/exploding · Xavier | 1,5h |
| 0.2 | **d2l — Tối ưu hoá** https://d2l.ai/chapter_optimization/index.html<br>· [momentum](https://d2l.ai/chapter_optimization/momentum.html) · [adam](https://d2l.ai/chapter_optimization/adam.html) · [lr-scheduler](https://d2l.ai/chapter_optimization/lr-scheduler.html) | ba mục đó là đủ, bỏ phần còn lại | 1h |
| 0.3 | **Goodfellow, *Deep Learning*** · [Ch.6 mạng truyền thẳng](https://www.deeplearningbook.org/contents/mlp.html) · [Ch.8 tối ưu hoá](https://www.deeplearningbook.org/contents/optimization.html) | **Ch.6.5 backprop & đồ thị tính toán** 🔴 · Ch.8.1–8.5 | 2h |
| 0.4 | **Loshchilov & Hutter (2019) — *Decoupled Weight Decay***<br>https://arxiv.org/pdf/1711.05101#section.2 | **Mục 2 + Thuật toán 2**. Chỉ cần hiểu **vì sao weight decay ≠ L2 trong Adam** | 0,5h |

**Phải rút ra được**
- **Vì sao Transformer dùng LayerNorm chứ không BatchNorm.** Câu trả lời ĐÚNG phải nhắc tới
  **padding và độ dài câu thay đổi**: thống kê theo batch bị ô nhiễm bởi ô đệm và đổi theo
  việc batch tình cờ gồm những câu nào. Nói "vì nó chuẩn hoá theo feature" là **chưa trả lời**.
- **Xavier ≠ He.** Xavier thiết kế cho tanh/sigmoid (đối xứng quanh 0); ReLU vứt một nửa tín hiệu
  nên cần hệ số gấp đôi — đó là He. Dùng nhầm thì phương sai **tắt dần**, mạng vẫn chạy, chỉ học kém.
- **AdamW ≠ Adam + L2.** Trong Adam, phạt L2 đi qua bộ chia thích nghi nên tham số có gradient lớn
  bị phạt nhẹ đi. AdamW tách weight decay ra khỏi bước thích nghi.
- **Gradient đi ngược qua residual.** Đường identity giữ gradient; mỗi phép chuẩn hoá chèn trên
  đường đó làm nó suy giảm. Đây là lý do Pre-LN không cần warmup dài còn Post-LN thì cần.

> 💻 Đo ngay bằng `bai_tap/06_norm_init_gradflow.py` — ba thí nghiệm, mỗi cái kết bằng một con số:
> LayerNorm bất biến theo batch (BatchNorm lệch ~2,3) · He giữ phương sai còn Xavier tắt (7e-07)
> · gradient tầng đáy **Pre-LN / Post-LN ≈ 5·10⁵ lần**.

---

## §1 · TRANSFORMER — TRỰC GIÁC RỒI ĐẾN CODE · 4h · N1–N2 🔴🔴

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 1.1 | **The Illustrated Transformer** — Jay Alammar<br>https://jalammar.github.io/illustrated-transformer/ | Toàn bộ. Đọc để LẤY TRỰC GIÁC, đừng cố nhớ chi tiết | 1h |
| 1.2 | **The Annotated Transformer** — Harvard NLP<br>http://nlp.seas.harvard.edu/annotated-transformer/#part-1-model-architecture | **Toàn bộ Part 1 — Model Architecture** (đến hết Full Model). Đọc CODE TỪNG DÒNG, gõ lại vào notebook riêng | **3h** 🔴🔴 |

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
| 2.1 | **d2l — Attention** · [queries-keys-values](https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html) · [attention-scoring-functions](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html) · [multihead-attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html) | đúng ba trang đó, không đọc cả chương | 45' |
| 2.2 | Bahdanau et al. (2015) — *Neural MT by Jointly Learning to Align and Translate*<br>https://arxiv.org/pdf/1409.0473#section.3 | **Chỉ mục 3** (mô hình alignment) + **Hình 3** (ma trận alignment) | 30' |

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
| 3.1 | Sennrich et al. (2016) — *Neural MT of Rare Words with Subword Units*<br>https://arxiv.org/pdf/1508.07909#subsection.3.2 | **Mục 3.2** (thuật toán BPE + ví dụ `low/lower/newest/widest`) và mục 1 (động cơ) | 45' |

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
- **J&M SLP3 Ch.13** Machine Translation · https://web.stanford.edu/~jurafsky/slp3/13.pdf
- Koehn *SMT* Ch.4 — IBM Model 1 (phương án SMT khi dữ liệu cực ít)
