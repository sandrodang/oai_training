# TUẦN 1 — TÀI LIỆU ĐỌC
### Đọc theo thứ tự. Mỗi mục ghi rõ: đọc phần nào, bao lâu, và **rút ra gì**.

> Nếu link đổi, tìm theo **tên tài liệu** — đều là tài liệu kinh điển, không mất.

---

## §1 · ĐÁNH GIÁ PHÂN LOẠI — 80 phút · N2

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 1.1 | **scikit-learn User Guide — Metrics** · https://scikit-learn.org/stable/modules/model_evaluation.html | Mục `precision_recall_fscore_support`, `f1_score` (đọc kỹ đoạn về `average=`), `balanced_accuracy_score`, `confusion_matrix` | 40' |
| 1.2 | **Jurafsky & Martin, SLP3** · https://web.stanford.edu/~jurafsky/slp3/ | **Ch.4 mục 4.7 "Evaluation: Precision, Recall, F-measure"** + 4.8 (test sets, cross-validation) | 30' |
| 1.3 | Wikipedia — *Youden's J statistic* | toàn bộ (ngắn) | 10' |

### Phải rút ra được
- `macro` = trung bình **không trọng số** các F1 từng lớp → **lớp hiếm nặng ký ngang lớp phổ biến**.
  `micro` = gộp TP/FP/FN toàn cục → với phân loại đơn nhãn thì `micro-F1 = accuracy`.
  `weighted` = trung bình có trọng số theo support → gần accuracy, **che mất lớp hiếm**.
- **Vì sao đề OlpAI luôn dùng `macro`**: để ép bạn xử lý lớp hiếm, không cho ăn gian bằng đa số.
- `Balanced Accuracy = (TPR + TNR)/2` **không chứa prevalence** → nộp toàn nhãn 0 luôn được đúng 0,5.
  (Chính là Bài học #3 trong `cv/RESULTS.md` của bạn.)
- `Youden J = TPR + TNR − 1 = 2·BA − 1` → **tối đa BA ⟺ tối đa J**.

---

## §2 · BLEU & SACREBLEU — 90 phút · N3 🔴

> **Ưu tiên cao nhất tuần này.** SacreBLEU xuất hiện ở **2/2 đề mẫu 2025**, VOAI CK cho trọng số **0.8**.

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 2.1 | **Papineni et al. (2002) — BLEU** · https://aclanthology.org/P02-1040/ | **Mục 2 toàn bộ** (modified n-gram precision, brevity penalty). Bỏ qua mục 3–5. | 35' |
| 2.2 | **Post (2018) — A Call for Clarity in Reporting BLEU Scores** · https://arxiv.org/abs/1804.08771 | **Toàn bộ, chỉ 6 trang** 🔴 | 30' |
| 2.3 | SLP3 · https://web.stanford.edu/~jurafsky/slp3/ | Ch.13 mục *"MT Evaluation"* (BLEU, chrF) | 25' |

### Phải rút ra được
- **Modified precision (clipping):** đếm n-gram trong bản dịch nhưng **cắt trần** bằng số lần xuất hiện
  tối đa trong tham chiếu → chặn mẹo lặp một từ đúng nhiều lần.
- **Brevity penalty:** `BP = 1` nếu `c > r`, ngược lại `exp(1 − r/c)`. **Phạt theo hàm mũ.**
  → Model thiếu dữ liệu hay dịch ngắn để "an toàn" — BLEU **trừng phạt rất nặng** hành vi đó.
- BLEU là **trung bình nhân** của `p1..p4` → **một `pn = 0` làm cả điểm về 0** ⇒ cần smoothing.
- **Bài học đắt nhất từ Post (2018):** cùng một bản dịch, đổi tokenizer (`13a` / `intl` / `char`)
  cho ra **điểm khác nhau đáng kể**. Luôn ghi lại **signature** của SacreBLEU.
- `chrF/chrF++` dựa trên n-gram ký tự → **ổn định hơn BLEU khi dữ liệu ít** và khi hình thái phức tạp.
  Rất đáng dùng làm **metric phụ offline** cho bài dịch ngôn ngữ hiếm.

---

## §3 · SAI SỐ, BOOTSTRAP, OVERFIT BẢNG XẾP HẠNG — 60 phút · N4 🔴

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 3.1 | **ESL** · https://hastie.su.domains/ElemStatLearn/ (PDF miễn phí) | **Chỉ mục 7.11 "Bootstrap Methods"**, lướt 7.10. *Bootstrap học bằng cách gõ BT 03 nhanh hơn đọc sách.* | 20' |
| 3.2 | **`PHUONG_PHAP_LUAN.md` — Nguyên lý 1** *(tài liệu của chính chúng ta)* | Toàn bộ §1.1–1.6 🔴 | 40' |

### Phải rút ra được
- SE của metric tỉ lệ `1/√n`. **Kiểm chứng bằng số của chính bạn:**
  `√(3340/51663) = 1/3,93` vs thực đo `0,003/0,0116 = 1/3,87` ✓
- **Bootstrap ghép cặp** (cùng chỉ số resample cho cả 2 model) cho SE của *hiệu số* nhỏ hơn 2–3 lần
  so với `√2 × SE` biên → phân biệt được cải thiện nhỏ.
- ⚠️ **Công thức quan trọng nhất:** chọn "tốt nhất trong k lượt nộp" bị thổi phồng `≈ SE·√(2 ln k)`.
  Với k=20, SE=0,0116 → **thiên lệch +0,028**, lớn hơn toàn bộ khoảng dao động giữa các cấu hình (0,001).

---

## §4 · THIẾT KẾ CHIA FOLD & RÒ RỈ — 90 phút · N5 🔴

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 4.1 | **sklearn — Cross-validation** · https://scikit-learn.org/stable/modules/cross_validation.html | Mục 3.1.2 (iterators), **đọc kỹ `GroupKFold`, `StratifiedGroupKFold`, `TimeSeriesSplit`** + mục về data leakage | 40' |
| 4.2 | sklearn — *Common pitfalls* · https://scikit-learn.org/stable/common_pitfalls.html | **Toàn bộ mục "Data leakage"** 🔴 | 25' |
| 4.3 | **`PHUONG_PHAP_LUAN.md` — Nguyên lý 3** | Toàn bộ §3.1–3.5 (bảng 6 loại rò rỉ + 4 phép thử) | 25' |

### Phải rút ra được
- Cơ chế rò rỉ ở R-ViHSD: ORIGINAL và bản nhiễu của **cùng comment** rơi khác fold → model chỉ cần
  **ghi nhớ**, không cần học. Thổi phồng 0,04 = **13 lần SE**.
- **Nhóm theo tầng cao nhất mà test cũng tách theo** (người ký hiệu ⊃ video ⊃ khung hình).
- 4 phép thử: **khoảng cách CV–public** · quét trùng lặp · **adversarial validation** · xáo nhãn.

---

## §5 · PYTORCH NỀN — 100 phút · N6

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 5.1 | **Dive into Deep Learning** · https://d2l.ai/ | **Ch.5** (MLP: backprop, khởi tạo) · **Ch.12.10–12.11** (Adam, lịch học) | 45' |
| 5.2 | **PyTorch — Reproducibility** · https://pytorch.org/docs/stable/notes/randomness.html | **Toàn bộ** 🔴 (đây là quy chế bắt buộc) | 20' |
| 5.3 | **PyTorch — AMP examples** · https://pytorch.org/docs/stable/notes/amp_examples.html | `autocast` + `GradScaler`, gradient clipping với AMP | 20' |
| 5.4 | He et al. — **Bag of Tricks** · https://arxiv.org/abs/1812.01187 | Mục 3 (tricks) + mục 4 (zero-γ, no-bias-decay, label smoothing) 🔴 | 15' |

### Phải rút ra được
- `.train()` vs `.eval()`: **BatchNorm** đổi hành vi (dùng thống kê batch vs running stats),
  **Dropout** tắt. **LayerNorm KHÔNG đổi** → đây là nguồn bug số 1 khi inference.
- **AdamW ≠ Adam + L2**: weight decay tách khỏi gradient, không bị chia cho `√v̂`. (d2l Ch.12 là đủ — không cần đọc bài báo gốc.)
- **Loại bias và tham số norm khỏi weight decay** (no-bias-decay).
- T4 **không có bf16** → phải dùng `fp16 + GradScaler`.
- Thứ tự đúng khi có AMP + clipping: `scaler.unscale_(opt)` → `clip_grad_norm_` → `scaler.step(opt)`.
- `cudnn.benchmark=True` nhanh hơn nhưng **phá tính xác định** → xung đột với quy chế "cố định seed".

---

## 🧪 TỰ KIỂM TRA — trả lời được hết mới sang tuần 2

Viết câu trả lời ra giấy, **không nhìn tài liệu**:

1. Ba cách tính trung bình F1 (`macro` / `micro` / `weighted`) khác nhau thế nào? Vì sao đề OlpAI chọn `macro`?
2. Vì sao nộp "toàn nhãn 0" cho **Balanced Accuracy = 0,5** bất kể tỉ lệ anomaly là bao nhiêu?
3. Vì sao lượt nộp đó được **50 điểm** ở vòng trường 2026 nhưng sẽ được **~0 điểm** ở công thức 2025?
4. Brevity penalty phạt bản dịch ngắn như thế nào? Vì sao model thiếu dữ liệu hay dịch ngắn?
5. SE của macro-F1 trên 3.340 mẫu ≈ 0,0116. Một cải thiện **0,004** có đáng tin không? Cần bao nhiêu SE?
6. Sau 20 lượt nộp, chọn cái điểm public cao nhất thì thiên lệch lên trên **bao nhiêu**? Viết công thức.
7. Ba tầng nhóm trong bài nhận diện ngôn ngữ ký hiệu là gì? Nên `GroupKFold` theo tầng nào? Vì sao?
8. `model.eval()` đổi hành vi của BatchNorm ra sao? Vì sao LayerNorm không bị ảnh hưởng?

> Đáp án gợi ý nằm rải trong `TAI_LIEU.md` và `PHUONG_PHAP_LUAN.md` — **tự tìm, đừng hỏi LLM**.

---

## 📚 Tài liệu nền để dành (không đọc tuần này)

Đánh dấu sẵn, sẽ dùng ở các tuần sau:
- **The Annotated Transformer** · http://nlp.seas.harvard.edu/annotated-transformer/ → **Tuần 2** 🔴🔴
- **The Illustrated Transformer** · https://jalammar.github.io/illustrated-transformer/ → Tuần 2
- **CS224n** · https://web.stanford.edu/class/cs224n/ → Tuần 2–3
- **CS231n** · https://cs231n.github.io/ → Tuần 4
- Sennrich — BPE · https://arxiv.org/abs/1508.07909 → Tuần 3
- Sennrich — Back-translation · https://arxiv.org/abs/1511.06709 → Tuần 3
- Lin — **TSM** · https://arxiv.org/abs/1811.08383 → Tuần 4
- Noroozi & Favaro — **Jigsaw** · https://arxiv.org/abs/1603.09246 → Tuần 5
