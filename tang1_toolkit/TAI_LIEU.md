# TẦNG 1 TOOLKIT — TÀI LIỆU ĐỌC
### ~4h tổng, chia theo module. Đọc phần của module nào TRƯỚC KHI code module đó.

---

## 📘 M01 · TRUNG BÌNH TRỌNG SỐ — 45' · làm ở **Tuần 3**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Izmailov et al. (2018) — *Averaging Weights Leads to Wider Optima and Better Generalization* (SWA)<br>https://arxiv.org/abs/1803.05407 | **Mục 1 + Hình 1–2** (trực giác: cực tiểu phẳng), lướt phần lý thuyết | 20' |
| Ott et al. (2018) — *Scaling Neural Machine Translation*<br>https://arxiv.org/abs/1806.00187 | Chỉ đoạn nói về **checkpoint averaging** (mục 3) | 10' |
| PyTorch docs — `torch.optim.swa_utils`<br>https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging | Toàn bộ, đặc biệt **`update_bn`** và VÌ SAO cần nó | 15' |

**Rút ra:** ba kỹ thuật cùng một họ — lấy trung bình trọng số để rơi vào **cực tiểu phẳng**,
tổng quát hoá tốt hơn. Khác nhau ở cách lấy trung bình: EMA (mũ, mỗi bước) · SWA (cộng, mỗi epoch) ·
checkpoint averaging (cộng, vài checkpoint cuối). **Cả ba đều làm hỏng running stats của
BatchNorm** → phải quét lại dữ liệu.

---

## 📘 M02 · MẸO GRADIENT — 40' · làm ở **Tuần 3**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| PyTorch — *Gradient Accumulation* trong Performance Tuning Guide<br>https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html | mục gradient accumulation + `set_to_none` | 15' |
| PyTorch — *CUDA Automatic Mixed Precision examples*<br>https://pytorch.org/docs/stable/notes/amp_examples.html | **"Gradient accumulation"** và **"Gradient clipping"** — đọc kỹ THỨ TỰ gọi | 15' |
| PyTorch — `torch.utils.checkpoint`<br>https://pytorch.org/docs/stable/checkpoint.html | phần mở đầu + cảnh báo về RNG | 10' |

**Rút ra:** thứ tự bắt buộc khi có cả AMP lẫn clipping —
`scaler.scale(loss).backward()` → `scaler.unscale_(opt)` → `clip_grad_norm_` → `scaler.step()`.
Clip trước khi unscale thì ngưỡng sai tỉ lệ với hệ số scale (thường 2^16), tức **không clip gì cả**.

---

## 📘 M05a · FOCAL LOSS — 25' · làm ở **Tuần 4**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Lin et al. (2017) — *Focal Loss for Dense Object Detection*<br>https://arxiv.org/abs/1708.02002 | **Mục 3** (công thức) + Hình 1 | 25' |

**Rút ra:** `(1-p_t)^gamma` hạ trọng số mẫu đã dễ. ⚠️ Với metric là macro-F1, **dịch ngưỡng
sau huấn luyện** thường rẻ ngang và đơn giản hơn — nhưng nhớ Tuần 1 BT05: phải kiểm bằng held-out,
tune ngưỡng ở R-ViHSD **làm tệ đi** task hate.

---

## 📘 M05b · DICE / TVERSKY / METRIC LEARNING — 45' · làm ở **Tuần 5**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Milletari et al. (2016) — *V-Net* (Dice loss)<br>https://arxiv.org/abs/1606.04797 | Chỉ **mục 2, đoạn Dice** | 15' |
| Salehi et al. (2017) — *Tversky loss*<br>https://arxiv.org/abs/1706.05721 | Mục 2 (công thức + vai trò α/β) | 15' |
| Hadsell et al. (2006) — *Dimensionality Reduction by Learning an Invariant Mapping* (contrastive loss) | Chỉ công thức contrastive | 15' |

**Rút ra:** BCE bị nền áp đảo khi 95% pixel là nền; Dice đo **trùng khớp vùng** nên miễn nhiễm.
Thực tế dùng **BCE + Dice**. Tversky với `β > α` thiên về **recall** — đúng thứ cần khi
bỏ sót vùng bệnh đắt hơn báo nhầm. Contrastive/triplet là nền cho bài **ghép ảnh** (VOAI CK TV2):
học "hai mảnh có kề nhau không".

---

## 📘 M03 · FINE-TUNE ENCODER — 40' · làm ở **Tuần 6**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Howard & Ruder (2018) — *ULMFiT*<br>https://arxiv.org/abs/1801.06146 | **Mục 3.2–3.3**: discriminative fine-tuning, gradual unfreezing | 25' |
| Sun et al. (2019) — *How to Fine-Tune BERT for Text Classification?*<br>https://arxiv.org/abs/1905.05583 | Mục 5.3 (**layer-wise LR decay**) + 5.4 (catastrophic forgetting) | 15' |

**Rút ra:** tầng thấp học đặc trưng phổ quát (đã tốt từ pretrain) → LR nhỏ.
Tầng cao + head → LR lớn. Ép tầng thấp học nhanh = **phá kiến thức đã có**.
Với dữ liệu ít + batch nhỏ, **đóng băng BatchNorm** thường thắng.

---

## 📘 M04 · HUẤN LUYỆN ĐỐI KHÁNG — 45' · làm ở **Tuần 6**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Goodfellow et al. (2015) — *Explaining and Harnessing Adversarial Examples* (FGSM)<br>https://arxiv.org/abs/1412.6572 | Mục 3–4 (ý tưởng gốc) | 20' |
| Miyato et al. (2017) — *Adversarial Training Methods for Semi-Supervised Text Classification*<br>https://arxiv.org/abs/1605.07725 | Mục 3: vì sao nhiễu vào **embedding** chứ không vào token | 15' |
| Zhu et al. (2020) — *FreeLB*<br>https://arxiv.org/abs/1909.11764 | Chỉ Thuật toán 1 | 10' |

**Rút ra:** không thể nhiễu token rời rạc → nhiễu vào **embedding liên tục**.
FGM = 1 bước (rẻ, +1 forward/backward). PGD = K bước (đắt gấp K). FreeLB = PGD nhưng
tận dụng gradient của mọi bước. **Trong 6 tiếng thi, FGM là lựa chọn đúng.**

---

## 📘 M06 · CONSISTENCY & ĐA NHIỆM — 40' · làm ở **Tuần 6**

| Tài liệu | Đọc phần nào | Giờ |
|---|---|---|
| Wu et al. (2021) — *R-Drop: Regularized Dropout for Neural Networks*<br>https://arxiv.org/abs/2106.14448 | **Mục 2** (công thức) + Hình 1 | 20' |
| Kendall & Gal (2018) — *Multi-Task Learning Using Uncertainty to Weigh Losses*<br>https://arxiv.org/abs/1705.07115 | **Mục 3** (công thức trọng số học được) | 20' |

**Rút ra:** R-Drop bắt hai lần forward (dropout khác nhau) phải đồng ý → đỡ overfit,
chi phí gấp đôi forward nhưng **không** gấp đôi tham số. Uncertainty weighting **học**
trọng số nhiệm vụ thay vì bạn quét tay `w_noise ∈ {0.15, 0.30}` như ở vòng trường.

---

## 🧪 TỰ KIỂM TRA

1. Ba cách trung bình trọng số khác nhau ở đâu? Cả ba làm hỏng thứ gì của BatchNorm?
2. Tích luỹ gradient 4 micro-batch mà **quên chia 4** thì tương đương chuyện gì?
3. Viết đúng thứ tự 4 lệnh khi có cả AMP lẫn gradient clipping. Đảo thứ tự thì sao?
4. Vì sao adversarial training nhiễu vào **embedding** chứ không vào token?
5. LLRD: tầng thấp LR nhỏ hay lớn? Vì sao?
6. Dice loss giải quyết vấn đề gì mà BCE không giải quyết được?
7. Tversky với `β > α` thiên về precision hay recall? Khi nào bạn muốn thế?
8. R-Drop khác consistency training ở chỗ nào (nguồn nhiễu)?
