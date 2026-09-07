# TUẦN 1 — TÀI LIỆU ĐỌC
### Tổng **~11h** (Tầng 0 ~9,5h + khởi động Tầng 1 ~1,7h).
### Đọc theo thứ tự. Mỗi mục ghi rõ: đọc phần nào, bao lâu, và **rút ra gì**.

> Nếu link đổi, tìm theo **tên tài liệu** — đều là tài liệu kinh điển, không mất.

---

## §1 · ĐÁNH GIÁ PHÂN LOẠI — 120 phút · N2

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 1.1 | **scikit-learn User Guide — Metrics** · https://scikit-learn.org/stable/modules/model_evaluation.html | Mục `precision_recall_fscore_support`, `f1_score` (đọc kỹ đoạn về `average=`), `balanced_accuracy_score`, `confusion_matrix` | 40' |
| 1.2 | **Jurafsky & Martin, SLP3** · https://web.stanford.edu/~jurafsky/slp3/ | **Ch.4 mục 4.7 "Evaluation: Precision, Recall, F-measure"** + 4.8 (test sets, cross-validation) | 30' |
| 1.3 | Wikipedia — *Youden's J statistic* | toàn bộ (ngắn) | 10' |
| 1.4 | **Saito & Rehmsmeier (2015) — *The Precision-Recall Plot Is More Informative than the ROC Plot***<br>https://doi.org/10.1371/journal.pone.0118432 | Hình 1–4 + phần thảo luận. Đây là câu trả lời dứt điểm cho **ROC-AUC hay PR-AUC** | 40' |

### Phải rút ra được
- `macro` = trung bình **không trọng số** các F1 từng lớp → **lớp hiếm nặng ký ngang lớp phổ biến**.
  `micro` = gộp TP/FP/FN toàn cục → với phân loại đơn nhãn thì `micro-F1 = accuracy`.
  `weighted` = trung bình có trọng số theo support → gần accuracy, **che mất lớp hiếm**.
- **Vì sao đề OlpAI luôn dùng `macro`**: để ép bạn xử lý lớp hiếm, không cho ăn gian bằng đa số.
- `Balanced Accuracy = (TPR + TNR)/2` **không chứa prevalence** → nộp toàn nhãn 0 luôn được đúng 0,5.
  (Chính là Bài học #3 trong `cv/RESULTS.md` của bạn.)
- `Youden J = TPR + TNR − 1 = 2·BA − 1` → **tối đa BA ⟺ tối đa J**.
- 🔴 **ROC-AUC vs PR-AUC.** ROC dùng TPR–FPR; `FPR = FP/N` có **N rất lớn** khi lớp dương hiếm,
  nên vài trăm FP cũng chỉ nhích FPR một chút → **ROC-AUC trông đẹp một cách gây hiểu nhầm**.
  PR dùng precision `TP/(TP+FP)`, không có N ở mẫu số, nên phản ánh đúng cái người dùng chịu.
  **Quy tắc: lớp dương hiếm (anomaly, hate speech) → đọc PR-AUC.** Baseline của PR-AUC là tỉ lệ
  lớp dương (không phải 0,5), nên PR-AUC 0,4 với lớp dương 1% là **rất tốt**, không phải tệ.

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

## §3 · SAI SỐ, BOOTSTRAP, BIAS–VARIANCE, OVERFIT BẢNG XẾP HẠNG — 100 phút · N4 🔴

### Đọc
| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 3.1 | **ESL** · https://hastie.su.domains/ElemStatLearn/ (PDF miễn phí) | **Chỉ mục 7.11 "Bootstrap Methods"**, lướt 7.10. *Bootstrap học bằng cách gõ BT 03 nhanh hơn đọc sách.* | 20' |
| 3.2 | **`PHUONG_PHAP_LUAN.md` — Nguyên lý 1** *(tài liệu của chính chúng ta)* | Toàn bộ §1.1–1.6 🔴 | 40' |
| 3.3 | **ESL Ch.7.3** (bias–variance decomposition) + **Ch.8.7** (bagging) | Chỉ cần công thức phân rã và **vì sao trung bình nhiều model giảm phương sai** | 40' |

### Phải rút ra được
- SE của metric tỉ lệ `1/√n`. **Kiểm chứng bằng số của chính bạn:**
  `√(3340/51663) = 1/3,93` vs thực đo `0,003/0,0116 = 1/3,87` ✓
- **Bootstrap ghép cặp** (cùng chỉ số resample cho cả 2 model) cho SE của *hiệu số* nhỏ hơn 2–3 lần
  so với `√2 × SE` biên → phân biệt được cải thiện nhỏ.
- ⚠️ **Công thức quan trọng nhất:** chọn "tốt nhất trong k lượt nộp" bị thổi phồng `≈ SE·√(2 ln k)`.
  Với k=20, SE=0,0116 → **thiên lệch +0,028**, lớn hơn toàn bộ khoảng dao động giữa các cấu hình (0,001).
- 🔴 **Vì sao ensemble giảm phương sai.** Trung bình `k` model có lỗi tương quan `ρ` cho phương sai
  `σ²·(ρ + (1−ρ)/k)`. Số hạng `(1−ρ)/k` tan đi khi k tăng, **nhưng `ρ·σ²` thì không**.
  ⇒ **Đa dạng (ρ thấp) quan trọng hơn chất lượng từng model.** Đó là lý do ở vòng trường,
  ghép ViSoBERT với TF-IDF+LogReg (yếu hơn hẳn) vẫn có ích, còn ghép hai ViSoBERT gần giống nhau thì không.

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

## §6 · VÒNG LẶP CẢI TIẾN — PHÂN TÍCH LỖI · TRẦN BAYES · BASELINE BTC — 180 phút · N7 🔴🔴

> §1–§4 dạy **ĐO** xem một cải thiện có thật không. §6 dạy **SINH RA giả thuyết nên cải thiện CÁI GÌ**.
> Hai nửa của cùng một vòng lặp. Thiếu nửa này thì 144 kỹ thuật ở các tầng sau chỉ là **thử mò**
> theo thứ tự ngẫu nhiên — mà trong 6 tiếng thi bạn chỉ thử được 3–4 thứ.

### Đọc

| # | Tài liệu | Đọc phần nào | Giờ |
|---|---|---|---|
| 6.1 | **Andrew Ng — *Machine Learning Yearning*** (miễn phí)<br>https://info.deeplearning.ai/machine-learning-yearning-book | **Ch.13–19** (error analysis: eyeball set, phân nhóm nguyên nhân, ước lượng trần) 🔴🔴 | 1h |
| 6.2 | **scikit-learn User Guide — `metrics`** | Mục **confusion matrix** + `ConfusionMatrixDisplay`. Tập đọc ma trận theo CẶP, không theo ô lẻ | 30' |
| 6.3 | **Northcutt et al. (2021) — *Pervasive Label Errors in Test Sets***<br>https://arxiv.org/abs/2103.14749 | Mục 1 + 2. Chỉ cần nắm: **tập test thật cũng đầy nhãn sai**, và điều đó đặt TRẦN lên mọi mô hình | 45' |
| 6.4 | **`de_tham_khao/` + `task1_nlp_fpt26/`** | Đọc **baseline BTC phát sẵn** như một tài liệu: kiến trúc · augmentation · số epoch · cách chia val | 45' |

### Phải rút ra được

**a. Hai chữ ký nhầm lẫn hoàn toàn khác nhau** — đây là ý quan trọng nhất cả mục:

| Dạng | Dấu hiệu trên ma trận | Nghĩa là gì | Nên làm gì |
|---|---|---|---|
| **Đối xứng** | a→b nhiều **và** b→a nhiều | hai lớp thật sự chồng lấn | 🛑 **TRẦN** — đừng đâm đầu tối ưu |
| **Một chiều** | a→b nhiều, b→a ít | **lệch prior / ngưỡng sai** | ✅ sửa rất rẻ, thường vài phút |

> ⚠️ Ở vòng trường 2026, đúng lỗi lệch prior này lấy mất **0,023 điểm** (v8 được 0,697 thay vì 0,720)
> và nó **LẶP LẠI HAI LẦN** — vì không ai nhìn ma trận nhầm lẫn.

**b. Đo trần TRƯỚC khi tối ưu, không phải sau.** Cách rẻ nhất: đếm mẫu **trùng input khác nhãn**.
Bằng chứng thật: **78,5% mẫu nhãn TEENCODE là no-op** (chuỗi sau biến đổi trùng khớp nguyên văn
một câu `ORIGINAL`), trong khi năm nhãn còn lại đều **dưới 10%**. Nghĩa là head noise có **trần cứng**;
mọi giờ đổ thêm vào nó là lãng phí. Phát hiện này đến từ **ĐẾM**, không từ mô hình, và tốn 15 phút.

**c. Baseline BTC là mẫu trực tiếp về trình độ của BTC.** §1.3 kế hoạch: điểm của bạn là
`(S − Min)/(Max − Min)`, tức mục tiêu là **vượt model phức tạp DO BTC huấn luyện** — một đích
**cố định và hữu hạn**, không phải vượt mọi đội. BTC thường phát sẵn baseline; **đọc trước khi chạy**.
Nếu baseline là ResNet18 + 10 epoch thì `Max` **không** phải SOTA, và mục tiêu vừa hạ một bậc.
Baseline cũng tiết lộ **định dạng nộp bài đúng** — nhiều đội mất lượt nộp đầu chỉ vì đoán sai định dạng.

**d. Cổng quyết định.** Mọi cải tiến phải qua 4 câu: gain > 2×SE? · dương trên mấy fold?
· bao nhiêu giây/mẫu khi suy luận? · có vỡ trần `main.py` 20 phút không?

> 💻 Cài đặt tương ứng: `bai_tap/07_error_analysis.py` và `bai_tap/08_bayes_ceiling.py`.
> Nghiệm thu `08`: chạy trên `work/data/training_set.csv` phải ra **TEENCODE 78,5%**, năm nhãn còn lại **< 10%**.

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
9. Cho ma trận nhầm lẫn 3 lớp: chỉ ra cặp nào là **trần** và cặp nào là **lệch prior**. Căn cứ vào đâu?
10. Không có mẫu trùng input thì đo trần Bayes bằng cách nào?
11. **ROC-AUC hay PR-AUC?** Lớp dương chiếm 1% thì chọn cái nào, vì sao cái kia gây hiểu nhầm?
12. Vì sao ensemble giảm phương sai? Vì sao **đa dạng quan trọng hơn chất lượng từng model**?

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
