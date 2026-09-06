# FPTU_Promt_Engineer — Task 1 (R-ViHSD) — Hồ sơ tái lập & hậu kiểm

## ⇒ HẬU KIỂM: CHỈ CẦN MỘT LỆNH

```bash
./VERIFY.sh                 # dữ liệu đã nằm ở ./data
./VERIFY.sh /path/to/data   # hoặc trỏ tới thư mục data của BTC
```

Script tự dựng môi trường, kiểm checksum dữ liệu, chạy audit tuân thủ, rồi sinh lại
dự đoán từ checkpoint và so từng dòng với file đã nộp. Thoát mã `0` khi tất cả đạt.
Các mục dưới đây là phần giải thích chi tiết.

Tài liệu này đáp ứng yêu cầu của BTC: *"Các đội phải lưu giữ toàn bộ mã nguồn
(huấn luyện và suy luận) cùng checkpoint đã dùng để sinh ra kết quả đã nộp."*

---

## 1. Tuyên bố tuân thủ

| Quy định của BTC | Cách đáp ứng | Kiểm chứng tại |
|---|---|---|
| Phải là mô hình học máy thực sự, có tham số học từ dữ liệu | Transformer đa nhiệm (ViSoBERT ~98M tham số) fine-tune. Các đặc trưng luật chỉ là **đầu vào** cho bộ phân loại có học, không bao giờ tự dự đoán | `src/train_mtl.py`, `src/features.py` |
| Cố định seed cho mọi thư viện ngẫu nhiên | `set_seed()` đặt `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED` | `src/common.py:set_seed` |
| Không sửa thủ công dữ liệu đầu vào hoặc tệp kết quả | Toàn bộ qua script; checksum dữ liệu gốc ghi ở mục 3 | `src/make_sub.py` |
| **Không dùng cột `id`** trực tiếp hay gián tiếp | `id` chỉ được đọc một lần duy nhất, để ghép dự đoán vào đúng dòng khi ghi CSV. Không vào feature, embedding, grouping hay suy luận nhãn | mục 6 |
| Mã nguồn tái tạo được kết quả đã nộp | Lệnh đầy đủ ở mục 5; đã kiểm chứng tái lập ở mục 7 | |

**Dữ liệu ngoài: KHÔNG sử dụng.** Chỉ dùng `training_set.csv` và `validation_set.csv` do BTC cấp.

---

## 2. Môi trường

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch --torch-backend=cu124
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

Python 3.10.12 · NVIDIA H100 80GB · CUDA 12.4. Phiên bản khoá trong `requirements.lock.txt`.

---

## 3. Dữ liệu (không chỉnh sửa)

```
d020d0b781352080f789f5ff738db10d  data/training_set.csv    48,092 dòng
725e19db6fdd2069ee4407135858dbda  data/validation_set.csv   5,344 dòng
6e5c0dd7a82ce386a49c19043d9e4282  data/public_test.csv      3,340 dòng
```

Giải nén trực tiếp từ các file `.zip` của BTC, không qua bất kỳ bước biến đổi nào.

---

## 4. Mô hình

Encoder chia sẻ `5CD-AI/visobert-14gb-corpus`, hai đầu ra phân loại:

```
text ──► ViSoBERT ──► mean+max pooling ──┬──► head_hate  (3 lớp)
                                          └──► head_noise (7 lớp)

loss = 0.70 · CE_hate(class-weighted) + 0.30 · CE_noise(class-weighted)
       + FGM adversarial (ε=1.0) trên word embeddings
```

Đánh giá bằng **StratifiedGroupKFold 5 fold**, nhóm theo *comment gốc* được phục hồi
bằng so khớp nearest-neighbour char-ngram (`src/build_groups.py`) — cần thiết vì
tập train ghép cặp (ORIGINAL + một bản nhiễu của cùng comment), nếu không nhóm
đúng thì CV bị rò rỉ và cao giả ~0.04.

---

## 5. Lệnh tái lập

```bash
# (a) khôi phục nhóm comment gốc, dùng cho chia fold
./.venv/bin/python src/build_groups.py

# (b) khai quật luật biến đổi từ các cặp aligned trong TRAIN (chỉ cho consistency loss)
./.venv/bin/python src/mine_rules.py

# (c) huấn luyện 5 fold + lưu checkpoint
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python src/train_mtl.py \
    --tag visobert_fgm --folds all --epochs 8 --maxlen 96 \
    --w_noise 0.30 --fgm 1.0 --save_ckpt 1

# (d) hiệu chỉnh prior cho đầu noise + sinh file nộp
./.venv/bin/python src/tune_prior.py --npz oof/visobert_fgm_oof.npz --out oof/bias_fgm.npz
./.venv/bin/python src/make_sub.py --ver v7_visobert_fgm \
    --npz oof/visobert_fgm_oof.npz --bias_npz oof/bias_fgm.npz

# (e) suy luận trên private test từ checkpoint đã lưu (không train lại)
./run_private.sh <private_test.zip> [password]
```

---

## 6. Cột `id` được dùng ở đâu

Chỉ đúng ba nơi, tất cả đều là ghép dòng khi xuất kết quả:

```
src/make_sub.py     "id": te["id"]                 # ghép dự đoán vào dòng
src/baseline.py     "id": te["id"]                 # ghép dự đoán vào dòng
src/common.py       load_test() đọc CSV nguyên bản
```

Nhóm chia fold (`group`) sinh từ **nội dung `text`**, không từ `id` —
xem `src/common.py:source_key` và `src/build_groups.py`.

---

## 7. Bằng chứng tái lập

Cùng seed cho kết quả trùng khít đến 4 chữ số thập phân, đo ở nhiều lần chạy độc lập:

| cấu hình | lần chạy 1 | lần chạy 2 |
|---|---|---|
| `w_noise=0.30` fold 0 | 0.7114 | 0.7114 |
| `w_noise=0.15` fold 0 | 0.7146 | 0.7146 |
| `FGM ε=1.0` fold 0 | 0.7277 | 0.7277 |
| `FGM + consistency` fold 0 | 0.7320 | 0.7320 |

---

## 8. Kết quả

| model | OOF (51,663 dòng) | hate | noise | public |
|---|---|---|---|---|
| TF-IDF baseline | — | 0.6970¹ | 0.6929¹ | 0.670 |
| `visobert_clean` | 0.7018 | 0.6972 | 0.7280 | 0.703 |
| `visobert_fgm` | **0.7139** | 0.7089 | 0.7421 | 0.702 |

¹ đo trên một fold giữ lại, không phải OOF đầy đủ.

Sai số của một điểm public (3,340 mẫu) là **±0.0116** (bootstrap 400 lần trên OOF),
nên các mức 0.700–0.703 không phân biệt được về mặt thống kê. Quyết định chọn model
dựa trên OOF 51,663 dòng (sai số ≈ ±0.003), không dựa trên bảng public.

---

## 9. BTC tự kiểm chứng bằng một lệnh

```bash
./.venv/bin/python verify_reproduction.py \
    --tag visobert_fgm \
    --sub sub/v7_visobert_fgm/task1_public_output.csv \
    --bias oof/bias_fgm.npz
```

Script nạp checkpoint trong gói, sinh lại dự đoán và so từng dòng với file đã nộp.
Thoát mã 0 chỉ khi trùng khít tuyệt đối.

Kết quả đã chạy:

```
rows=3340  checkpoints=5
pred_label      match: 100.00%
pred_noise_type match: 100.00%
VERDICT: EXACT REPRODUCTION
```

---

## 10. Kết quả vòng Private (chính thức)

| lượt | bài nộp | mô hình | OOF | private |
|---|---|---|---|---|
| 1 | `private` | ensemble fgm+cons, bias prior 0.25 | 0.7204 | 0.718 |
| 2 | `private_v2` | ensemble fgm+cons, bias prior 0.21 | 0.7204 | 0.718 |
| 3 | `pA_ens3` | **ensemble fgm+cons+clean** | **0.7220** | **0.720** |
| 4 | `pE_pertask` | ensemble 3 model, trọng số riêng từng task | 0.7220 | 0.719–0.720 |
| 5 | `pD_ens3even` | ensemble 3 model, trọng số đều | — | 0.719–0.720 |

**Điểm private cuối cùng: 0.720**

### Độ chính xác của OOF trong dự báo private

| | OOF | private | lệch |
|---|---|---|---|
| ensemble 2 model | 0.7204 | 0.718 | −0.002 |
| ensemble 3 model | 0.7220 | 0.720 | −0.002 |

OOF (51,663 dòng, SE ≈ ±0.003) dự báo đúng điểm private hai lần liên tiếp.
Bảng public (3,340 dòng, SE ≈ ±0.0116 đo bằng bootstrap) lệch tới 0.02 và
chỉ sai hướng — mọi quyết định chọn mô hình đều dựa trên OOF.
