# TÁC VỤ 2 — IMAGE ANOMALY DETECTION — CHIẾN LƯỢC 3 GIỜ

## 0. Kết luận nhanh
- Metric = **macro Balanced Accuracy** trên 6 category, nhãn nhị phân do mình tự cắt ngưỡng.
- Bài toán tách làm 2 phần gần như độc lập: (A) **chất lượng ranking score trong nội bộ mỗi category**, (B) **vị trí cắt ngưỡng**.
- Mục tiêu của (A) là **ranking tốt ở vùng quanh decision boundary**, KHÔNG phải tối đa AUROC tuyệt đối. `max_τ BA = (1 + J*)/2` với `J*` = Youden index; AUROC chỉ dùng để chọn model offline, không quy đổi thẳng ra điểm LB.
- (B) rẻ hơn (A) rất nhiều nhưng sai lệch lớn hơn nhiều (xem §4). → **Ưu tiên: có submission hợp lệ thật sớm, rồi dùng LB làm nguồn calibration.**

## 1. Dữ liệu (đã kiểm tra thực tế)
| Category | Train normal | Ảnh mẫu | Kích thước | Ghi chú |
|---|---|---|---|---|
| category_01 | 664 | mạch cảm biến siêu âm | 1404x1070 | vật thể canh giữa, dễ |
| category_02 | 664 | mạch sạc TP4056 | 1358x1104 | xoay/lệch vị trí, trung bình |
| category_03 | 302 | ~25 viên nang xanh | 1500x1000 | **KHÓ NHẤT** — lỗi rất nhỏ, nhiều instance |
| category_04 | 660 | 4 nến tealight (2x2) | 1284x1168 | có lỗi logic (thiếu/thừa/sai bấc) |
| category_05 | 660 | 4 miếng nui trên nền xanh | 1500x1000 | **KHÓ** — lỗi nhỏ, vị trí tự do |
| category_06 | 210 | 1 hạt điều | 1274x1176 | ít train nhất, nhưng dễ |

Public test 80 ảnh/cat, private 160 ảnh/cat. Ảnh **không vuông**, mỗi cat một resolution cố định.

**3 hệ quả kỹ thuật:**
1. **Không center-crop, không resize méo mó** → resize giữ tỉ lệ (letterbox về bội số 14 cho ViT). Vật thể (nui, nang) nằm sát rìa, crop là mất defect.
2. **Vật thể KHÔNG align** (nang/nui/mạch xoay tự do) → ưu tiên **PatchCore/kNN memory bank (position-agnostic)**. Không chọn PaDiM làm baseline vì giả định Gaussian theo từng vị trí kém phù hợp khi object placement/pose biến thiên (rõ nhất ở cat 03/05).
3. **Defect siêu nhỏ ở cat 03/05** → cần **resolution cao** + pooling `top-1% mean` thay vì `max` (max nhạy nhiễu, top-k mean ổn định hơn).

## 2. Môi trường (đã dựng xong, nằm hết trong `cv/`)
```
cv/
  .venv/            torch 2.13+cu130, torchvision, timm 1.0.28   (đã test CUDA OK)
  hf/               HF cache — đã tải sẵn dinov2-reg ViT-B/14 + wide_resnet50_2
  dataset_train/    3160 ảnh normal, 6 category + 3 file csv
  public_test/      480 ảnh + test.csv
  src/ sub/ cache/ logs/
```
GPU: dùng **`CUDA_VISIBLE_DEVICES=7`** (~68GB trống). Các GPU khác gần như full.
Private test giải nén vào `cv/private_test/` ở giờ thứ 6.

## 3. Mô hình — 3 tầng, làm theo thứ tự, mỗi tầng đều nộp được
### Tầng 1 (T+0:20 → T+1:00) — PatchCore WideResNet50-2  ⇒ AUROC ~0.94
- Feature: `layer2 + layer3`, adaptive-avgpool 3x3 neighborhood, concat.
- Input ~512px cạnh dài (giữ tỉ lệ). Coreset greedy 1-2%. kNN k=1.
- Image score = **mean of top-1% patch distances**.
- Flip augment cho memory bank: **phải đo, không mặc định**. Chạy 3 biến thể (none / +HFlip / +HFlip+VFlip) trên synthetic validation — flip làm bank rộng ra, có thể kéo anomaly distance xuống và giảm detection.
- Cache feature ra `cv/cache/` — mọi thí nghiệm sau chỉ tính lại phần nhẹ.

### Tầng 2 (T+1:00 → T+1:40) — DINOv2-reg ViT-B/14  ⇒ AUROC ~0.97
- `vit_base_patch14_reg4_dinov2.lvd142m`, lấy patch token layer ~ {6,9,12}, resize động (dynamic img size) theo tỉ lệ gốc, cạnh dài 518 → cat 03/05 đẩy lên **728-882**.
- Nhánh chính: PatchCore memory bank trên patch token. Nhánh phụ (optional, thêm vì chất lượng chứ không phải vì luật): **Mahalanobis trên CLS**, cov fit từ train normal.
- **Ensemble = trung bình RANK trong nội bộ category** — `R(x) = (rank_patch + rank_cls)/2`. Không cộng raw score.

### Tầng 3 (T+1:40 → T+2:10, nếu còn thời gian) — chỉ cho category yếu (03, 05)
- Tăng resolution + multi-scale ensemble (518 + 728 + 896).
- TTA: flip h/v/hv, lấy mean rank.
- Tuỳ chọn: Dinomaly-style decoder nhẹ train trên normal (~5 phút/cat trên H100) — đẩy AUROC VisA-like lên ~0.98.

## 4. Chiến lược NGƯỠNG (phần ăn điểm nhiều nhất) — ĐÃ KIỂM CHỨNG BẰNG MÔ PHỎNG
Script: `src/sim_threshold.py`, `sim_threshold2.py`, `sim_threshold3.py`.

### Sự thật toán học nền tảng
`BA = (TPR + TNR)/2` — **không chứa prevalence π**. Do đó **ngưỡng tối ưu τ\* trong score-space độc lập hoàn toàn với tỉ lệ anomaly**. Mô phỏng (n=400k):

| π | τ\* | BA\* | p\* = quantile trên TEST | q\* = quantile trên NORMAL |
|---|---|---|---|---|
| 0.15 | 1.05 | 0.8107 | 0.240 | 0.146 |
| 0.25 | 0.99 | 0.8108 | 0.316 | 0.161 |
| 0.40 | 1.04 | 0.8114 | 0.397 | 0.148 |
| 0.55 | 1.01 | 0.8106 | 0.496 | 0.155 |

→ **`p` trôi mạnh theo prevalence; `q` và `τ*` đứng yên.** Quantile trên test bất biến với *scale của score*, KHÔNG bất biến với *prevalence*. Public và private là 2 tập độc lập, prevalence có thể khác.

### Quy tắc chốt: HYBRID hai neo
```
τ(c) = ½ · [ Q_{1−p}( score_test(c) )  +  Q_{1−q_c}( score_train_normal_heldout(c) ) ]
```
Private macro-BA ×100, 500 lần lặp:

| Kịch bản | p-quantile (test) | q-quantile (normal) | **HYBRID** | oracle |
|---|---|---|---|---|
| prevalence giống 0.40→0.40 | 80.73 | 80.55 | **80.85** | 83.04 |
| prevalence lệch 0.40→0.25 | 79.36 | 80.51 | **80.42** | 83.35 |
| prevalence lệch 0.40→0.55 | 79.03 | 80.56 | **80.49** | 83.00 |
| drift normal +0.35σ, prev giống | **76.63** | 75.68 | 76.45 | 79.05 |
| drift normal +0.35σ, prev lệch | 75.69 | 75.67 | **75.84** | 79.40 |

Neo `p` chết khi prevalence lệch; neo `q` chết khi score của normal bị drift. **Hybrid không thua quá 0.2 điểm ở bất kỳ kịch bản nào** → chọn hybrid.

### `q` lấy MIỄN PHÍ, không tốn submission
1. Sweep **duy nhất `p`** trên public LB.
2. Với `p*` tốt nhất → tính `τ_pub(c) = Q_{1−p*}(score_public(c))`.
3. `q_c = tỉ lệ train-normal held-out của category c có score ≥ τ_pub(c)`.
4. **Freeze cả `p*` và `q_c`.** Ở private: dựng lại 2 ngưỡng từ 2 phân bố tương ứng rồi lấy trung bình.

Cột "q suy ra" đạt 80.51 vs 80.55 của q-tuned-bằng-LB → suy ra miễn phí không mất gì.

### Sweep `p`: 5 điểm, không 11
`argmax` thô và `fit parabol` chênh nhau 0.17 điểm (79.35 sd 1.94 vs 79.18 sd 1.70) — nằm trong noise, không đáng tranh luận. Chốt: **5 điểm {0.22, 0.28, 0.34, 0.42, 0.50}**, fit bậc 2 và **clip đỉnh về trong khoảng quan sát**; nếu fit lồi ngược thì lấy argmax. Không tiêu 11/20 submission cho việc này.

### Đọc leaderboard: PHẢI có mốc neo
LB hiển thị `SCORE/MAX × 100%`, và **MAX trôi mỗi khi có đội khác vượt lên** → mọi Δ đo được đều ở đơn vị không xác định.

→ **Lượt nộp #1 = file toàn nhãn 0.** Khi đó `TPR=0, TNR=1 → BA=0.5` mọi category → `SCORE = 50` chính xác.
`MAX = 5000 / d₀` với `d₀` là số hiển thị. Từ đó mọi entry trên bảng quy được về macro-BA tuyệt đối (bảng re-normalize toàn bộ entry theo MAX hiện tại nên các entry đọc cùng thời điểm luôn cùng thang). Lượt này cũng validate format khi chưa có model.

### Probe từng category
Giữ 5 category **bit-identical**, chỉ đổi ngưỡng của category `c`:
```
Δ(điểm hiển thị) × MAX/100 × 6  =  BA_c(mới) − BA_c(cũ)
```
Vì 5 category kia giống hệt từng bit, **sai số của chúng triệt tiêu hoàn toàn** trong hiệu — không chỉ giảm. Độ chính xác chỉ giới hạn bởi 80 ảnh của category đang dò.

### Shrink về prior
Public chỉ 80 ảnh/cat. Nếu ước lượng từ synthetic offline (`p₀`) và từ public (`p_pub`) lệch nhau nhiều, **không freeze thẳng `p_pub`** mà lấy giá trị trung gian, ưu tiên vùng LB phẳng thay vì đỉnh nhọn.

### Ngân sách 20 lượt public
| Lượt | Mục đích |
|---|---|
| 1 | **Toàn 0** → neo MAX + validate format |
| 2 | Tầng 1 (WRN50 PatchCore), p=0.35 — chốt pipeline |
| 3-6 | Sweep p: {0.22, 0.28, 0.42, 0.50} → fit + clip |
| 7-9 | So model: WRN50 vs DINOv2 vs rank-ensemble, p cố định |
| 10-15 | Probe từng category (5 cat bit-identical), tinh chỉnh riêng cat 03/05 |
| 16-18 | Vòng 2 cho cat yếu + xác nhận cấu hình cuối |
| 19 | Re-nộp file toàn 0 → kiểm tra MAX đã trôi bao nhiêu, hiệu chỉnh lại các Δ |
| 20 | Dự phòng |

## 5. Validation offline (để không đốt submission)
Không có nhãn → tự dựng **synthetic anomaly** từ chính train normal (luật cho phép rõ ràng):
- CutPaste (dán patch ngẫu nhiên), scar (vệt mảnh), local blur, color/brightness shift cục bộ, xoá/nhân bản 1 instance (cho cat 03/04/05 — mô phỏng lỗi logic).
- Chia train 90/10, dùng 10% normal + synthetic anomaly để đo AUROC/BA từng category.
→ Dùng để **chọn giữa các biến thể model**, không dùng để chốt `p` (phân bố synthetic ≠ thật). `p` chốt bằng LB.

## 6. Tuân thủ luật — checklist bắt buộc (mục tiêu: không bị loại)
- [ ] **Chỉ dùng ảnh train-normal của BTC.** Không tải VisA/MVTec hay bất kỳ ảnh ngoài nào. Không dùng canonical anomaly images.
- [ ] Pretrained weights (DINOv2, WideResNet ImageNet) — **được phép** (mục 5 đề CV nói rõ).
- [ ] Không dùng external labels / external model outputs.
- [ ] **Không xem rồi tự gán nhãn từng ảnh public test.** Mọi nhãn sinh tự động từ pipeline. Chỉ visualize ảnh TRAIN khi debug.
- [ ] **Phải có tham số học từ dữ liệu** — coreset memory bank fit trên train normal đã thoả yêu cầu "mô hình học máy thực sự". Không thêm component chỉ để "lách luật"; Mahalanobis giữ lại vì lý do chất lượng.
- [ ] **Fix seed** cho `random`, `numpy.random`, `torch` (+ `torch.use_deterministic_algorithms`, `cudnn.deterministic=True`).
- [ ] Lưu **toàn bộ source + checkpoint** (memory bank `.npy`, cov, p) trong `cv/` để hậu kiểm; `predict.py` phải tái tạo đúng file đã nộp.
- [ ] Không sửa tay dữ liệu vào/ra.
- [ ] Freeze model/weights/threshold TRƯỚC khi private mở.

## 7. Format nộp (sai format = mất trắng lượt nộp)
- CSV UTF-8, dấu phẩy, có header, **không index**, đúng 3 cột đúng thứ tự: `sample_id,category,label`
- `label` ∈ {0,1}; đúng 1 dòng/sample_id; `category` khớp 100% với `test.csv`; không thiếu/thừa/trùng ID.
- Tên file trong zip: `task2_public_output.csv` / `task2_private_output.csv`
- Tên zip: `[team]_task2_pub.zip` / `[team]_task2_pri.zip`
- Viết sẵn `src/validate_sub.py` tự kiểm 6 điều kiện trên trước MỌI lượt nộp.

## 8. Timeline 3 giờ — nguyên tắc: KHÔNG BAO GIỜ ở trạng thái chưa có submission
| Thời gian | Việc | Submission |
|---|---|---|
| 0:00-0:15 | loader + seed + `validate_sub.py` + `make_sub.py` | **#1 toàn 0 → neo MAX** |
| 0:15-0:50 | WRN50 PatchCore (layer2+3, kNN, top-1% mean), cache feature | **#2** |
| 0:50-1:20 | Sweep p + thử k / flip-variant / top-k% pooling | **#3-6** |
| 1:20-1:50 | DINOv2-reg PatchCore (+ CLS Mahalanobis) | **#7-9** |
| 1:50-2:10 | Rank-ensemble, multi-scale cho cat 03/05 | |
| 2:10-2:35 | Probe từng category | **#10-15** |
| 2:35-2:50 | Chốt cấu hình, suy ra `q_c`, **FREEZE** `p*` + `q_c` + weights | **#16-18** |
| 2:50-3:00 | Chạy lại `predict.py` từ đầu trên public, diff byte-for-byte với file đã nộp; viết README tái lập | **#19** re-neo MAX |

**Cây quyết định — mỗi nhánh đều phải có checkpoint + submission hợp lệ:**
```
BASE: WRN50 layer2+3 + kNN
   └─ Public #2
        ├─ ổn  → DINOv2 → rank ensemble → category probe
        └─ chưa ổn → tuning feature/resolution trước, KHÔNG nhảy sang model mới
```

### Giờ 6 — Private (5 lượt)
1. Giải nén `private_test/` → `predict.py --test private_test` (cấu hình đã freeze).
2. `validate_sub.py` → nộp **lượt #1**.
3. **Nộp lại ĐÚNG file đó ở lượt cuối (#5).** Dù BTC tính first / last / best thì kết quả đều giống nhau → không cần phụ thuộc vào việc hỏi BTC cơ chế chọn.
4. Lượt #2-#4 chỉ dùng nếu có sự cố kỹ thuật (file lỗi, thiếu ID). **Không** dùng để dò p trên private — vi phạm tinh thần "PRIVATE = FINAL ONLY".

## 9. Rủi ro & xử lý
| Rủi ro | Xử lý |
|---|---|
| GPU dùng chung, chậm | Cache feature ngay lần đầu; mọi sweep chỉ chạy trên feature đã cache (giây, không phải phút) |
| Public 80 ảnh/cat → LB nhiễu (~±1.3 điểm 1σ) | Sweep 5 điểm + fit có clip; probe category dùng 5 cat bit-identical để triệt tiêu nhiễu; shrink về prior khi LB phẳng |
| Prevalence private khác public | Neo hybrid (test-quantile + normal-quantile) — đã kiểm chứng bằng sim, mất <0.2 điểm ở mọi kịch bản |
| Score của normal bị drift giữa train và test | Cùng neo hybrid xử lý; theo dõi bằng cách so phân bố score train-normal vs test |
| LB hiển thị SCORE/MAX, MAX trôi | Nộp file toàn 0 để neo MAX; re-neo ở lượt 19 |
| cat 03/05 kéo tụt mean BA (và tie-break 2 là min BA) | Dồn resolution + TTA vào đúng 2 cat này |
| Hết giờ | Mỗi tầng đều có submission hợp lệ; không bao giờ ở trạng thái "chưa nộp được gì" |
