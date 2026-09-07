# KẾ HOẠCH ÔN LUYỆN OLPAI'26 — VÒNG LOẠI KHU VỰC MIỀN BẮC
### Bản 2 — đã sửa theo đính chính. Lý thuyết trước, cài đặt sau.

> **Mốc:** hôm nay **T2 07/09/2026** → thi **T7 31/10/2026**. Còn **54 ngày**:
> 7 tuần trọn (Tuần 1 dài 8 ngày) + 4 ngày giảm tải cuối.
> **Địa điểm:** Học viện Công nghệ Bưu chính Viễn thông (Hà Nội).
> **Đội hình:** 3 người · 2 máy · 1 tài khoản DeepSeek (2k ngữ cảnh/phiên).

---

## ĐÍNH CHÍNH BẢN 1

| Bản 1 nói sai | Thực tế |
|---|---|
| "Vòng loại có **3 bài** ML/CV/NLP" | **2 tác vụ**, 3 lĩnh vực **lồng ghép**. Thường là **1 NLP + 1 CV**, phần ML nằm bên trong cả hai. Đúng như 4/4 đề đã ra. |
| Lấy **đề thi thử 2026** làm chuẩn | Đề thi thử **chỉ là tham khảo**. Mẫu thật là **SOLOAI 2025 (sơ loại)** và **VOAI 2025 (chung kết)**. |
| Trần **25M tham số** + danh sách trắng torchvision | **Bỏ hoàn toàn.** Không có trong quy chế thật. SOLOAI 2025 còn *phát sẵn* pretrained CRNN/3D-CNN kèm `download_model.py` liệt kê model được phép. |
| "LLM = Gemini trên Colab" | **`deepseek-r1-distill-qwen-32b`**, offline, **token vô hạn** nhưng **mỗi phiên chỉ 2.000 token ngữ cảnh**. Khoá sau giờ thứ 5. |
| Thiếu hẳn phần lý thuyết | Bản này có **Phần 4 — Chương trình lý thuyết**, **68 giờ đọc** (bản 3), đặt **trước** phần cài đặt. |

Bản 1 lưu tại `scratchpad/KE_HOACH_v1_backup.md`, không dùng nữa.

---

## PHẦN 1 — LUẬT CHƠI THẬT

### 1.1 Khung tổ chức (quy chế OlpAI'26)
- **6 tiếng liên tục**, dự kiến **8h30–14h30**, có mặt **7h30**.
- Đội **2–3 sinh viên** + tối đa 2 HLV. Tối đa 5 đội/trường ở vòng loại.
- **60 đội** vào chung kết (≥10 đội/miền, **tối đa 2 đội/trường**).
- *"Tập mã nguồn phải đảm bảo chạy được hoàn toàn trong môi trường **Colab hoặc Kaggle**."*
- Môi trường 2025: *"Làm bài trên **FPT Smart Cloud** (cấp theo account sử dụng GPU), nộp bài trên **hệ thống VNOJ**."*
- Có **buổi practice hôm trước** (2025: 1/11, 15h–17h). **Bắt buộc tham gia** — đó là lúc đo GPU thật.

### 1.2 Cấu trúc đề: **2 tác vụ, 3 lĩnh vực**

Quy chế nói *"03 lĩnh vực: ML, CV, NLP"*, nhưng **thực tế 4/4 đề đều đúng 2 tác vụ**:

| Kỳ thi | Tác vụ 1 | Tác vụ 2 | ML nằm ở đâu |
|---|---|---|---|
| **SOLOAI 2025** (mẫu chính) | Dịch **Trung → Việt**, SacreBLEU | **Nhận diện ngôn ngữ ký hiệu** (video), Macro-F1 | Đề *cho phép* rule-based / **SMT** / NMT ở TV1 |
| **VOAI 2025 CK** (mẫu chính) | Phân loại + **Dịch Ba Na → Việt**<br>`0.2·MacroF1 + 0.8·BLEU` | **Ghép ảnh jigsaw 3×5**, PPA<br>`0.3·NV1 + 0.7·NV2` | Phân loại ở TV1; TV2 là bài **tối ưu tổ hợp + ML** |
| Vòng trường 2026 | Hate speech + noise type | Image anomaly detection | Cả hai đều là phân loại |
| Đề thi thử 2026 *(tham khảo)* | — | Bệnh tôm + segmentation | — |

**Kết luận vận hành:** chuẩn bị **1 tác vụ NLP + 1 tác vụ CV**, mỗi tác vụ có **2 nhiệm vụ con trọng số lệch**,
và phần "ML" là lớp phân loại/hồi quy/tối ưu nằm bên trong. **Không cần ôn tabular Kaggle như bản 1 đề xuất.**

### 1.3 Công thức tính điểm — bạn đấu với **BTC**, không phải với các đội khác

Cả SOLOAI 2025 lẫn VOAI 2025 dùng **cùng một công thức**, khác hẳn vòng trường 2026:

```
Norm_score = (Submission_Score − Min_Score) / (Max_Score − Min_Score) × 100
```
- `Min_Score` = điểm của **model rất đơn giản do BTC huấn luyện**
- `Max_Score` = điểm của **model phức tạp do BTC huấn luyện**
- Điểm cuối = **trung bình cộng Norm_score của 2 tác vụ**, làm tròn 2 chữ số.

> ⚠️ Đề ghi: *"Nếu Norm_score < 0 thì quy về 0. Nếu Norm_score > **10** thì quy về **10**."*
> Con số 10 mâu thuẫn với `×100` — nhiều khả năng là lỗi đánh máy của **100**. **Phải hỏi BTC** (§11).

**Ba hệ quả chiến lược — đây là điều quan trọng nhất trong tài liệu này:**

1. **Mục tiêu cụ thể là _vượt model phức tạp của BTC_, không phải vượt mọi đội.** Khác hoàn toàn với
   vòng trường 2026 (`SCORE/MAX` với MAX = đội mạnh nhất, chỉ 1 đội được 100). Ở công thức 2025,
   **mọi đội đều có thể đạt trần**. Đây là mục tiêu hữu hạn, đo được, và **thấp hơn bạn tưởng**:
   model BTC train trong vài ngày, không phải SOTA.
2. **Có `Min_Score` nghĩa là baseline tầm thường được ~0 điểm.** Ở vòng trường bạn nộp "toàn nhãn 0"
   và được 50.0 vì công thức `SCORE/MAX`. **Ở công thức này, mẹo đó cho 0.** Baseline chỉ còn giá trị
   kiểm tra định dạng, **không còn giá trị neo điểm**.
3. **Chênh lệch `Max − Min` ở mẫu số phóng đại mọi cải thiện nhỏ.** Nếu BTC đặt Min/Max sát nhau,
   vài phần trăm BLEU có thể là hàng chục điểm Norm_score.

### 1.4 Ràng buộc bị bỏ sót ở bản 1: thư mục `Final/` + `main.py` chạy ≤ 20 phút

**Cả hai đề mẫu 2025 đều yêu cầu** (nguyên văn):

> *"Thí sinh phải tạo một thư mục trên hệ thống cloud có tên `Final/` chứa toàn bộ chương trình tái tạo
> kết quả tốt nhất của 2 tác vụ. Bên trong `Final/` phải có 2 thư mục con: `Tac_vu_1` và `Tac_vu_2`.
> Mỗi thư mục con chứa tệp `main.py`. Khi chạy lệnh `python main.py`, chương trình phải sinh ra đúng tệp
> `submission.csv` tương ứng với kết quả tốt nhất, và **thời gian chạy tối đa 20 phút**."*

Đây là **ràng buộc kỹ thuật cứng** và nó **loại bỏ nhiều hướng đi**:
- Không thể để `main.py` huấn luyện lại từ đầu → phải **lưu checkpoint và chỉ inference**.
- **Ensemble 6 nhánh của bạn ở vòng trường sẽ không chạy nổi trong 20 phút** trên GPU Colab.
- Phải tính **ngân sách thời gian inference** ngay từ khi chọn kiến trúc.
- `main.py` phải **tự tìm dữ liệu, tự nạp weights, tự ghi CSV** — không phụ thuộc notebook.

→ **Thêm vào thư viện chiến đấu: mẫu `main.py` inference-only, viết được trong 10 phút.**

### 1.5 LLM: `deepseek-r1-distill-qwen-32b` — 2.000 token ngữ cảnh

- Truy cập qua trình duyệt, **hoạt động hoàn toàn offline**, BTC cấp tài khoản trước giờ thi.
- **Chỉ dùng trong 5 tiếng đầu.** Giờ thứ 6 **tự động khoá**.
- **Token vô hạn, nhưng mỗi phiên chỉ 2.000 token ngữ cảnh.**
- Đây là mô hình **suy luận** (R1-distill) → nó tự sinh chuỗi `<think>` rất dài, **ăn hết ngân sách 2k
  trước khi kịp trả lời**. Nếu không ép ngắn, bạn sẽ nhận về câu trả lời **bị cắt giữa chừng**.

Chiến thuật chi tiết ở **§8**. Tóm tắt: *nhiều phiên ngắn, mỗi phiên một câu hỏi nguyên tử, cấm dán code dài.*

### 1.6 Quy định dữ liệu, tái lập, hậu kiểm (áp dụng chắc chắn)
- *"Chỉ được sử dụng dữ liệu do Ban giám khảo cung cấp. Không được dùng thêm dữ liệu thật từ nguồn khác."*
- **Được phép** sinh synthetic data và augmentation từ dữ liệu BTC.
- *"Không được sử dụng tập test dưới bất kỳ hình thức nào để huấn luyện."*
- *"Phải cố định seed cho tất cả các thư viện sử dụng ngẫu nhiên"* (`random`, `numpy`, `sklearn`, `torch`…).
- *"Không được chỉnh sửa thủ công dữ liệu đầu vào/đầu ra."* — *"Quá trình chạy sẽ được ghi log toàn bộ."*
- Nộp cuối vòng: **code + trọng số mô hình + báo cáo kỹ thuật vắn tắt**, ứng với **lần nộp điểm cao nhất**.
- Đánh giá = **bảng xếp hạng tự động** *"và các tiêu chí của cuộc thi"* (định lượng + **định tính**).

### 1.7 Nộp bài
- **20 lượt** public trong 5 tiếng đầu · **5 lượt** private trong 1 tiếng cuối · **mỗi tác vụ**.
- Private test mở bằng **mật khẩu** ở giờ thứ 6. Truy cập trước = vi phạm quy chế.
- Mỗi lượt: **một file `.zip`** chứa `output.csv` **+ file `.ipynb`**.
  ⚠️ *"File kết quả đầu ra phải đảm bảo được lấy từ việc chạy **tuần tự** file `.ipynb`, các code cell chạy
  lần lượt từ trên xuống. Bài chỉ hợp lệ khi mã nguồn đã nộp sinh ra kết quả tương đồng với file đầu ra."*
  → **Notebook phải chạy được từ trên xuống, sạch.** Không được để cell chạy lộn xộn rồi nộp.
- CSV: UTF-8, dấu phẩy, **không có cột index**.

### 1.8 Bẫy "phải có mô hình học máy thực sự"

> *"Bài làm bắt buộc phải cài đặt mô hình học máy thực sự. Lời giải thuần giải thuật, không chứa tham số
> học được từ dữ liệu, sẽ không được công nhận."*

**Đây là bẫy chết người cho dạng bài như VOAI 2025 TV2 (ghép ảnh jigsaw)** — bài đó giải được bằng
so khớp cạnh + tối ưu tổ hợp thuần, và lời giải như vậy **bị loại**.

**Cách phòng:** mọi bài có mùi "thuật toán" phải có một thành phần **học từ dữ liệu** *thực sự tham gia
quyết định*. Với jigsaw: huấn luyện CNN Siamese dự đoán *"hai mảnh này có kề nhau không"* → dùng logit
làm trọng số cạnh → **rồi mới** chạy tối ưu tổ hợp. Ghi rõ trong báo cáo: tham số nào, học từ đâu, ảnh hưởng ra sao.

---

### 1.9 Framework cao cấp & thư viện — được dùng đến đâu?

**Kết luận: KHÔNG có danh sách cấm nào.** Rà toàn bộ 4 đề (SOLOAI 2025, VOAI CK 2025, 2 đề vòng trường 2026,
đề thi thử 2026) + 4 trang quy chế, chỉ tìm thấy đúng 3 chỗ nhắc tới thư viện:

| Chỗ nhắc | Nguyên văn | Ý nghĩa |
|---|---|---|
| Quy chế chung | *"Phải cố định seed cho mọi **thư viện** có sử dụng yếu tố ngẫu nhiên (random, numpy.random, torch, …)"* | Ngầm định: **dùng thư viện gì cũng được**, miễn seed được |
| Đề vòng trường NLP | *"…miễn là tuân thủ **quy định về thư viện và tài nguyên của kỳ thi**"* | ⚠️ **Tham chiếu treo** — quy định này **không được công bố ở đâu**. → Phải hỏi BTC (§11.10) |
| Đề vòng trường NLP | `pip install -r requirements.txt` | BTC **phát kèm `requirements.txt`** trong bộ kit → đó là môi trường chuẩn de facto |

**Bằng chứng mạnh nhất — chính bài của bạn đã qua hậu kiểm:**
`work/requirements.lock.txt` ghi `transformers==4.46.3` · `accelerate==1.1.1` · `tokenizers==0.20.3` ·
`scikit-learn==1.7.2`, và `cv/` dùng `timm 1.0.28` + `huggingface-hub`.
Bài đã chạy `VERIFY.sh` → **`VERDICT: EXACT REPRODUCTION`**. Tức **HF stack và `accelerate` đã được chấp nhận trên thực tế.**

#### Ràng buộc thật là ràng buộc *chức năng*, không phải theo tên thư viện

Framework nào cũng được, miễn thoả **đồng thời** 5 điều:

1. Chạy **hoàn toàn trong Colab hoặc Kaggle** (quy chế)
2. **Seed cố định → tái lập khít** (quy chế; BTC có quyền hậu kiểm bất kỳ lúc nào)
3. **Notebook chạy tuần tự từ trên xuống** phải sinh đúng file đã nộp (đề 2025)
4. **`Final/main.py` chạy ≤ 20 phút** (đề 2025)
5. Không kéo thêm **dữ liệu/trọng số ngoài** danh sách BTC cho phép

#### Đánh giá rủi ro từng loại

| Thư viện | Phán quyết | Rủi ro thật (không phải "bị cấm") |
|---|---|---|
| `numpy` `pandas` `scipy` `scikit-learn` `opencv` `albumentations` `sacrebleu` `sentencepiece` `lightgbm` | ✅ **Dùng thoải mái** | Không. Đều có sẵn trên Colab/Kaggle |
| **`timm`** (làm *model zoo*) | ✅ Rất nên | ⚠️ Rủi ro nằm ở `pretrained=True` (§dưới), **không** ở framework |
| **`accelerate`** | ✅ An toàn | Lớp bọc mỏng, bạn **đã dùng và qua hậu kiểm** |
| `transformers` (dùng `AutoModel` + tự viết vòng lặp) | ✅ An toàn | Như trên |
| **`transformers.Trainer`** | 🟠 Cân nhắc | Tự set seed/dataloader bên trong · `load_best_model_at_end` giấu trạng thái · import chậm |
| **PyTorch Lightning** | 🟠 Cân nhắc | **Bẫy tái lập:** checkpoint tự đánh version `version_0/`, `version_1/`… → chạy lại notebook có thể nạp **checkpoint khác** cái đã sinh ra bài nộp ⇒ vi phạm điều 3 |
| **fastai** | 🔴 Tránh | Mặc định "ý kiến riêng" (augment, normalize, one-cycle) ẩn · nhạy phiên bản · hay xung đột với torch mới trên Colab |
| **Keras / tf.keras** | 🔴 Tránh | TF + PyTorch tranh VRAM trên cùng GPU · tính xác định khó hơn · lạc hệ sinh thái, phí thời gian |

#### ⚡ Lý do thật sự nên tự viết vòng lặp: **DeepSeek 2k ngữ cảnh không debug nổi framework**

Đây mới là lập luận mạnh nhất, không phải chuyện luật:

> Một traceback của Lightning/HF Trainer dài **40+ frame**, xuyên qua 5 lớp trừu tượng.
> **Bạn không thể dán nó vào cửa sổ 2.000 token.** Còn vòng lặp 80 dòng bạn tự viết thì bạn
> **tự đọc được**, và nếu cần hỏi thì dán vừa 10 dòng liên quan.

Đó là lý do Tuần 1 bắt bạn gõ `train_loop.py` từ trí nhớ — không phải khổ hạnh, mà là **để giữ khả năng
tự sửa lỗi khi không có LLM ngữ cảnh dài**. Ở vòng trường bạn có LLM tự do nên framework không sao;
ở vòng miền thì không.

#### ⚠️ Rủi ro lớn hơn framework: `pretrained=True`

Câu hỏi đáng lo không phải *"được dùng Lightning không"* mà là *"được tải trọng số nào"*:
- SOLOAI 2025 phát kèm **`download_model.py` liệt kê model được phép** → tức **có** danh sách trắng theo từng bài.
- `timm.create_model(..., pretrained=True)` và `AutoModel.from_pretrained(...)` đều **tải từ internet**
  → phụ thuộc mạng, và **có thể nằm ngoài danh sách cho phép**.

**Quy tắc ngày thi:** mở bộ kit → **tìm `download_model.py` / `requirements.txt` TRƯỚC KHI viết code**.
Đó là văn bản quy định thư viện & pretrained thật sự, không phải file PDF đề.

#### Khuyến nghị chốt

| Dùng | Không dùng |
|---|---|
| **Tự viết vòng lặp train** (~80 dòng, thư viện chiến đấu §6 module 2) | Trainer/Learner dày của bất kỳ framework nào |
| `timm`/`transformers` **chỉ để lấy kiến trúc** (`AutoModel`, `create_model`) | `Trainer.train()`, `Learner.fit_one_cycle()` |
| `accelerate` nếu cần AMP/thiết bị gọn | Lightning `Trainer` (bẫy version checkpoint) |
| sklearn, LightGBM, albumentations, sacrebleu — thoải mái | Keras/TF song song với PyTorch |

**Việc phải làm ở buổi practice hôm trước:** liệt kê **những gì đã cài sẵn** trên môi trường BTC
(`pip list`), và đo **thời gian `pip install`** cho thứ còn thiếu. Trong 6 tiếng, 90 giây cài đặt
nhân với 3 lần thử là 5 phút mất trắng.

---

## PHẦN 2 — CHẨN ĐOÁN: BẠN ĐANG Ở ĐÂU

### Điểm mạnh thật (không cần ôn — đây là thứ khó dạy nhất)

Đọc `work/REPRODUCE.md` và `cv/RESULTS.md`, bạn đã thể hiện tư duy trình độ giải cao:

1. **Đo được nhiễu của bảng public bằng bootstrap** (SE ±1.3 điểm với 480 ảnh; ±0.0116 với 3.340 mẫu),
   rồi **từ chối chọn model theo public**, chọn theo OOF 51.663 dòng (SE ±0.003).
   **OOF dự báo đúng điểm private 2 lần liên tiếp.**
2. **Phát hiện hàm mục tiêu offline bị hỏng** (`AUC_proxy` sai hướng 4/4 lần) và **dừng search** thay vì
   search nhanh hơn trên hàm hỏng.
3. **StratifiedGroupKFold theo comment gốc** — tránh rò rỉ 0.04.
4. Nhận ra **ensemble đa dạng mới tạo bước nhảy thật** (9/10 cấu hình đơn lẻ nằm gọn trong nhiễu 76.3–77.9;
   ensemble 4 nhánh → 80.4).

> 📎 **Bốn điều trên được giải thích sâu trong [`PHUONG_PHAP_LUAN.md`](PHUONG_PHAP_LUAN.md)** — nền toán
> của từng nguyên lý, một **nguyên lý thứ 5** bạn đã làm nhưng chưa đặt tên, và **3 điểm nâng cấp** cho vòng miền:
> bootstrap **ghép cặp** · kiểm định proxy bằng **Spearman ρ** · **ensemble chia sẻ encoder** để lọt trần 20 phút.

### Lỗ hổng thật — sắp xếp lại theo bằng chứng từ 2 đề mẫu 2025

| # | Lỗ hổng | Vì sao nghiêm trọng | Mức |
|---|---|---|---|
| **A** | **Dịch máy / seq2seq — chưa từng làm** | **2/2 đề mẫu 2025 đều có dịch máy chấm SacreBLEU.** VOAI CK còn cho BLEU trọng số **0.8**. Đây là kỹ thuật có xác suất ra cao nhất và bạn có 0 kinh nghiệm. | 🔴🔴 **Cao nhất** |
| **B** | **CV "phi chuẩn": video/action + jigsaw/self-supervised** | 2/2 đề CV mẫu 2025 **không phải phân loại ảnh thường**: nhận diện ký hiệu (video) và ghép ảnh (không gian). Bạn mới làm anomaly detection. | 🔴🔴 **Cao nhất** |
| **C** | **PyTorch cấp thấp đã quên** (bạn tự nhận) | Không có pretrained nào cứu được tiếng Ba Na. Phải **tự viết Transformer, tự viết decoder, tự viết loss**. | 🔴 Nghiêm trọng |
| **D** | **Chưa từng bị ép ngân sách inference 20 phút** | Ràng buộc `Final/main.py ≤ 20 phút` (§1.4) giết chết lối đánh "ensemble mọi thứ" của bạn. | 🔴 Nghiêm trọng |
| **E** | **Chưa chạy Colab/Kaggle bao giờ** | Toàn bộ `cv/PLAN.md` viết cho H100 80GB, `CUDA_VISIBLE_DEVICES=7`. Bạn **không biết** 1 epoch mất bao lâu trên T4. | 🟠 Cao |
| **F** | **Phụ thuộc LLM ngữ cảnh dài** | Giờ chỉ còn **2.000 token/phiên** với một model suy luận hay "nghĩ" tràn ngữ cảnh. | 🟠 Cao |
| **G** | **Chưa có kịch bản 3 người / 2 máy / 1 LLM** | Xem §7. | 🟠 Cao |

---

## PHẦN 3 — DỰ ĐOÁN DẠNG ĐỀ 31/10 (từ 2 mẫu chính)

### 3.1 Quy luật rút ra

1. **Mỗi tác vụ = 2 nhiệm vụ con, trọng số lệch mạnh** (0.2/0.8 · 0.3/0.7 · 0.85/0.15).
   → *Việc đầu tiên khi mở đề: tìm nhiệm vụ trọng số lớn.* Ở VOAI CK, làm phân loại hoàn hảo (0.2)
   mà dịch kém (0.8) thì thua đội ngược lại.
2. **Bối cảnh Việt Nam, dữ liệu hiếm, ngôn ngữ/miền ít tài nguyên** (Ba Na, ký hiệu Việt, tôm, mạng xã hội Việt).
3. **Luôn có một "khúc cua"** thay vì bài sách giáo khoa: ngôn ngữ không ai biết, không có nhãn dương,
   phải tự chọn ngưỡng, phải suy ra cấu trúc không gian.
4. **Dữ liệu vừa phải** (vài nghìn → vài chục nghìn mẫu) — vừa đủ train trên GPU Colab trong ~1 giờ.
5. **BTC hay phát sẵn baseline + danh sách model được phép** (SOLOAI: `download_model.py`).
   → *Luôn đọc file đó đầu tiên.*

### 3.2 Xác suất dạng đề (để phân bổ thời gian ôn)

**Tác vụ NLP:**

| Xác suất | Dạng | Bằng chứng |
|---|---|---|
| ★★★★ | **Dịch máy ít tài nguyên** (Việt ↔ ngôn ngữ hiếm), SacreBLEU | SOLOAI T1, VOAI CK T1 |
| ★★★ | Phân loại văn bản tiếng Việt (kèm nhiễu/robustness) | VOAI CK T1, vòng trường 2026 |
| ★★ | Tóm tắt / sinh văn bản / QA | Chưa ra, nhưng cùng họ seq2seq |
| ★★ | NER / gán nhãn chuỗi | Chưa ra |

**Tác vụ CV:**

| Xác suất | Dạng | Bằng chứng |
|---|---|---|
| ★★★ | **Video / chuỗi ảnh — nhận dạng hành động, cử chỉ** | SOLOAI T2 |
| ★★★ | **Suy luận cấu trúc không gian** (ghép ảnh, sắp xếp, đếm) | VOAI CK T2 |
| ★★★ | Phân loại ảnh chuyên ngành + phát hiện bất thường | Vòng trường 2026 |
| ★★ | Segmentation / định vị vùng | Đề thi thử 2026 |
| ★★ | OCR / chữ viết tay tiếng Việt | Chưa ra, rất hợp bối cảnh |

---

## PHẦN 4 — CHƯƠNG TRÌNH LÝ THUYẾT
### *Đọc → tự kiểm tra → mới cài đặt.* Tổng **68 giờ đọc** trong 8 tuần (**10h/tuần**, xem §4B).
### Ngân sách mới: **4 giờ/ngày = 28 giờ/tuần** — đọc chỉ là 10h trong đó.

**Cách dùng phần này:** mỗi Tầng có 4 mục —
**(a) Khái niệm nền** (phải hiểu, không được mơ hồ) ·
**(b) Kỹ thuật nâng cao** (đây là thứ tạo khác biệt điểm số) ·
**(c) Đọc gì** (tài liệu cụ thể, có thứ tự) ·
**(d) Tự kiểm tra** (trả lời được mới đi tiếp).

Ký hiệu ưu tiên: 🔴 bắt buộc · 🟠 nên có · 🟡 nếu dư thời gian.

---

### ⬛ TẦNG 0 — ĐÁNH GIÁ & PHƯƠNG PHÁP THỰC NGHIỆM · 10h · 🔴🔴
> *Đặt đầu tiên vì mọi quyết định trong 6 tiếng thi đều dựa trên câu hỏi "cải thiện này có thật không?"*

**(a) Khái niệm nền**
- Precision / Recall / F1; **macro vs micro vs weighted** — vì sao macro phạt nặng lớp hiếm.
- **Balanced Accuracy**, Youden's J; ROC-AUC vs **PR-AUC** (khi nào PR-AUC mới đúng).
- **BLEU**: n-gram precision có cắt tỉa (clipping), **brevity penalty**, smoothing, vì sao là geometric mean.
- **AP / AP50**: đường cong PR, nội suy, greedy matching theo IoU.
- Cross-validation: k-fold, stratified, **group**, **stratified-group**, time-series. **Rò rỉ dữ liệu**.
- **Bootstrap** để ước lượng sai số chuẩn của một metric.
- Bias–variance; vì sao **ensemble giảm phương sai** và vì sao **đa dạng > chất lượng đơn lẻ**.

**(b) Kỹ thuật nâng cao**
- **SacreBLEU signature** — cùng bản dịch, đổi tokenizer (`13a` / `intl` / `char`) đổi điểm rất nhiều.
- **chrF / chrF++** — ổn định hơn BLEU với ngôn ngữ hình thái phức tạp; dùng làm metric phụ offline.
- **Adaptive overfitting** lên bảng public: mỗi lần nộp là một lần "hỏi" tập test → 20 lượt nộp là
  20 bậc tự do. Reusable holdout.
- **Adversarial validation**: train classifier phân biệt train vs test để phát hiện lệch phân phối.
- **Tối ưu ngưỡng theo metric** (F1/BA không tối ưu bằng cách tối đa accuracy) — coordinate search trên OOF.
- **Nelder–Mead / hill-climbing** tìm trọng số ensemble trên OOF.

**(c) 🔴 VÒNG LẶP CẢI TIẾN — *nửa còn thiếu của Tầng 0***

> (a) và (b) dạy cách **đo** xem một cải thiện có thật không. Mục này dạy cách **sinh ra giả thuyết
> nên cải thiện cái gì**. Hai nửa của cùng một vòng lặp. Thiếu nửa này thì 144 kỹ thuật ở các tầng
> sau chỉ là thử mò theo thứ tự ngẫu nhiên — và trong 6 tiếng bạn chỉ thử được 3–4 thứ.

**c1 · Phân tích lỗi có kỷ luật — 30 phút mỗi vòng** 🔴

- Lấy **50 mẫu sai nặng nhất** theo loss (hoặc theo confidence sai cao nhất). **Đọc bằng mắt.**
- Phân loại thủ công vào **≤ 6 nhóm nguyên nhân**. Đếm. Nhóm lớn nhất là nơi *duy nhất* đáng đổ giờ tiếp theo.
- Đọc **ma trận nhầm lẫn**, phân biệt hai chữ ký khác hẳn nhau:
  - Nhầm **đối xứng** (a→b nhiều *và* b→a nhiều) ⇒ hai lớp thật sự chồng lấn ⇒ đây là **trần**, đừng đâm đầu.
  - Nhầm **một chiều** (a→b nhiều, b→a ít) ⇒ **lệch prior / ngưỡng sai** ⇒ sửa rất rẻ, thường vài phút.
- Ghi vào bảng cố định: `nhóm lỗi · số mẫu · giả thuyết · chi phí sửa · đã thử chưa · kết quả`.

> Đây là hoạt động ROI cao nhất trong 6 tiếng và là thứ dễ bị bỏ qua nhất khi vội.
> Nó rẻ hơn train thêm một model, và nó là cách duy nhất để **chọn đúng kỹ thuật thứ 4** thay vì thử cả 11.

**c2 · Trần Bayes từ nhiễu nhãn — đo TRƯỚC khi tối ưu** 🔴

- Trước khi đổ giờ vào một head/nhiệm vụ con, hỏi: **trần của nó là bao nhiêu?**
- Cách đo rẻ nhất: tìm các mẫu **trùng input nhưng khác nhãn**. Tỉ lệ đó **chặn trên** accuracy đạt được.
  Không có mẫu trùng thì gom theo khoá gần đúng (char-ngram) rồi đo trong từng nhóm.
- Bằng chứng thật, vòng trường 2026: **78,5% mẫu nhãn TEENCODE là no-op** — chuỗi sau biến đổi
  *không khác* chuỗi gốc. Head noise vì thế có **trần cứng**; mọi giờ đổ thêm vào nó là lãng phí.
  Phát hiện này đến từ **đếm**, không từ mô hình, và tốn 15 phút.
- Hệ quả: khi một nhiệm vụ con chỉ chiếm 15% trọng số **và** có trần thấp, chiến lược đúng là
  **bỏ nó ở mức đủ dùng** và dồn toàn bộ giờ sang nhiệm vụ chiếm 85%.

**c3 · Đọc baseline BTC phát sẵn — 20 phút đầu giờ thi** 🔴

- §1.3: điểm của bạn là `(S − Min)/(Max − Min)`. Mục tiêu **không phải** vượt các đội khác —
  mà là **vượt model phức tạp do chính BTC huấn luyện**. Đó là một đích **cố định và hữu hạn**.
- BTC thường **phát sẵn baseline** (SOLOAI 2025: `download_model.py`; CRNN cho nhánh video).
  **Baseline đó là mẫu trực tiếp về trình độ kỹ thuật của BTC** — đọc nó là cách rẻ nhất để ước lượng
  `Max_Score` nằm ở đâu.
- Việc phải làm: **đọc trước khi chạy**. Ghi lại kiến trúc · augmentation · số epoch · cách chia val ·
  có dùng pretrained không. Nếu baseline là ResNet18 + 10 epoch, `Max` **không** phải SOTA —
  và mục tiêu của bạn vừa hạ xuống một bậc.
- ⚠️ Baseline cũng tiết lộ **định dạng nộp bài đúng**. Nhiều đội mất lượt nộp đầu chỉ vì đoán sai định dạng.

**c4 · Cổng quyết định — mọi cải tiến phải qua 4 câu hỏi**

1. Gain trên OOF có **> 2×SE** không? (SE ước bằng bootstrap, §b)
2. **Dương trên bao nhiêu fold?** 5/5 đáng tin hơn 3/5 cùng mức gain.
3. Chi phí **inference** bao nhiêu giây/mẫu?
4. Có làm vỡ ràng buộc **`main.py` ≤ 20 phút** (§1.4) không?

> Không qua đủ 4 cổng thì **không đưa vào bản nộp**, dù nó "có vẻ đúng về lý thuyết".

**(d) Đọc**
1. Post (2018) *"A Call for Clarity in Reporting BLEU Scores"* — 🔴 ngắn, bắt buộc.
2. Papineni et al. (2002) *BLEU* — chỉ mục 2.
3. *The Elements of Statistical Learning* Ch.7 (Model Assessment & Selection).
4. Jurafsky & Martin *SLP3* Ch.4 mục đánh giá + phụ lục kiểm định thống kê.
5. scikit-learn User Guide: `model_selection`, `metrics`, `calibration`.

**(e) Tự kiểm tra**
- Tính tay BLEU-4 cho một cặp câu 8 từ, có BP.
- SE của macro-F1 trên 3.340 mẫu ≈ bao nhiêu? Cải thiện 0.004 có đáng tin không?
- Vì sao nộp "toàn nhãn 0" cho **50 điểm** ở công thức `SCORE/MAX` nhưng **0 điểm** ở công thức `(S−Min)/(Max−Min)`?
- Cho một ma trận nhầm lẫn 3 lớp: chỉ ra nhầm lẫn nào là **trần** và nhầm lẫn nào là **lệch prior**.
- Không có mẫu trùng input thì đo trần Bayes bằng cách nào?
- BTC phát baseline ResNet18 + 10 epoch. Điều đó nói gì về `Max_Score`?

---

### ⬛ TẦNG 1 — HỌC SÂU CỐT LÕI · 14h · 🔴
> *Mục tiêu: hiểu đủ sâu để tự viết mọi thứ khi không có pretrained và không có LLM ngữ cảnh dài.*

**(a) Khái niệm nền**
- Backpropagation, đồ thị tính toán, vanishing/exploding gradient.
- **Optimizer**: SGD+momentum, Adam, **AdamW** (vì sao weight decay ≠ L2 trong Adam).
- **Lịch học**: warmup, cosine, **Noam / inverse-sqrt** (chuẩn cho Transformer), OneCycle.
- **Chuẩn hoá**: BatchNorm vs **LayerNorm** vs GroupNorm vs RMSNorm — vì sao Transformer dùng LayerNorm.
- Khởi tạo Xavier/He. Dropout. **Label smoothing**.
- Loss: CE (weight, label_smoothing), BCE (pos_weight), **Focal**, **Dice**, contrastive/triplet.
- Transfer learning: feature-extraction vs fine-tuning, catastrophic forgetting.

**(b) Kỹ thuật nâng cao**
- **Pre-LN vs Post-LN** — Pre-LN huấn luyện ổn định **không cần warmup**; Post-LN mạnh hơn nhưng khó train.
  *Với 6 tiếng thi, chọn Pre-LN.*
- **Gradient accumulation** để mô phỏng batch lớn (dịch máy cần batch ~25k token mới ổn định).
- **Gradient clipping** theo norm (bắt buộc cho seq2seq).
- **EMA (Exponential Moving Average)** của trọng số · **SWA (Stochastic Weight Averaging)** ·
  **Checkpoint averaging** (trung bình 5–10 checkpoint cuối) — *gần như luôn dương, chi phí ~0.*
- **Layer-wise LR decay (LLRD)** khi fine-tune encoder pretrained.
- **Gradual unfreezing** (ULMFiT).
- **Mixed precision** (fp16 + GradScaler; T4 không có bf16), **gradient checkpointing** để đổi tính toán lấy VRAM.
- **Multi-sample dropout**, **Mixout**, **LayerDrop**.
- **Adversarial training**: FGSM → **FGM** (đo thật: **+0.0121** OOF, dương trên **5/5 fold**) → **PGD** → **FreeLB** → **AWP**.
- **R-Drop / consistency regularization** — hai lần forward cùng input, phạt KL giữa hai phân phối.
- Cân bằng đa nhiệm: trọng số tay vs **uncertainty weighting (Kendall & Gal)** vs GradNorm.

**(c) Đọc**
1. **d2l.ai** (miễn phí, có code) Ch.4–7 (MLP, tối ưu, CNN), Ch.11 (tối ưu nâng cao) — 🔴
2. Goodfellow *Deep Learning* Ch.6–8.
3. He et al. (2018) *"Bag of Tricks for Image Classification"* — 🔴 ngắn, cực nhiều mẹo áp dụng ngay.
4. Loshchilov & Hutter (2019) *Decoupled Weight Decay (AdamW)*.
5. Izmailov et al. (2018) *SWA*.
6. Wu et al. (2021) *R-Drop*; Zhu et al. (2020) *FreeLB*.

**(d) Tự kiểm tra**
- Vì sao `model.eval()` đổi kết quả BatchNorm nhưng không đổi LayerNorm?
- Vì sao Transformer Post-LN cần warmup còn Pre-LN thì không?
- Viết công thức cập nhật AdamW, chỉ ra chỗ khác Adam+L2.

---

### ⬛ TẦNG 2 — NLP & DỊCH MÁY · 20h · 🔴🔴 **TRỌNG TÂM SỐ 1**
> *2/2 đề mẫu 2025 có dịch máy. VOAI CK cho BLEU trọng số 0.8. Đây là tầng quan trọng nhất.*

**(a) Khái niệm nền**
- Tokenization: word / char / **subword**. **BPE**, WordPiece, **SentencePiece (unigram)**. Vocab chia sẻ nguồn–đích.
- Embeddings: word2vec, GloVe, **fastText** (subword — quan trọng cho ngôn ngữ hiếm/sai chính tả).
- **Seq2seq**: encoder–decoder, teacher forcing, **exposure bias**.
- **Attention**: Bahdanau (additive) vs Luong (multiplicative); alignment.
- **Transformer**: self-attention, multi-head, positional encoding, causal mask, cross-attention.
- **Beam search**: length penalty, coverage penalty, vì sao greedy kém.
- Encoder pretrained: BERT, RoBERTa, **XLM-R**, **PhoBERT** (cần tách từ VnCoreNLP), **ViSoBERT** (không cần).

**(b) Kỹ thuật nâng cao — *đây là danh mục bạn yêu cầu***

*Huấn luyện NMT:*
- **Label smoothing 0.1** (chuẩn de facto cho MT)
- **Checkpoint averaging** 5–10 checkpoint cuối 🔴🔴 — *lựa chọn ensemble MẶC ĐỊNH của kỳ thi này:*
  gần như luôn dương, chi phí huấn luyện ~0, **chi phí suy luận ×1** nên không đụng trần 20 phút.
- **Tied embeddings** (encoder-in / decoder-in / decoder-out dùng chung ma trận) — giảm ~1/3 tham số
- **Noam schedule** (warmup 4000 bước, inverse-sqrt)
- **BPE-dropout / subword regularization** — augmentation cực rẻ cho dữ liệu ít 🔴
- **SwitchOut**, word dropout, token masking
- **Back-translation** (Sennrich) và **iterative back-translation** 🔴 — vũ khí chính khi dữ liệu ít
- **Sequence-level knowledge distillation** (Kim & Rush)
- Dropout cao (0.3) cho tập nhỏ; giảm chiều model thay vì tăng

*Giải mã (decoding) — nơi ăn điểm rẻ nhất:*
- **Ensemble decoding**: trung bình log-prob của nhiều model **tại mỗi bước** (mạnh hơn ensemble đầu ra) 🟠
  ⚠️ **MÂU THUẪN PHẢI GIẢI TRƯỚC KHI DÙNG:** chi phí gấp N lần beam search, trong khi §1.4 áp trần
  `main.py ≤ 20 phút`. **Quy tắc:** đo `giây/mẫu × số mẫu private × N` *trước*; nếu > 15 phút thì
  chuyển sang **checkpoint averaging** (gộp N checkpoint thành 1 model — chi phí suy luận **×1**,
  giữ phần lớn lợi ích) hoặc **cascade** (§Tầng 5). Hạ từ 🔴 xuống 🟠 chính vì ràng buộc này.
- **MBR decoding** (Minimum Bayes Risk) — sinh n-best rồi chọn câu có BLEU/chrF kỳ vọng cao nhất 🟠
- **Reranking n-best** bằng model ngược chiều hoặc language model (noisy channel)
- **Lexically constrained decoding** — ép giữ tên riêng/số
- **Copy mechanism / pointer-generator** — cho từ hiếm, tên, chữ số 🔴

*Ít tài nguyên (đúng kịch bản Ba Na / ngôn ngữ hiếm):*
- **Char-level hoặc byte-level** khi vocab quá nhỏ/nhiễu 🔴
- **Chia sẻ vocab** khi hai ngôn ngữ dùng chung bảng chữ cái Latin (Ba Na ↔ Việt: rất hợp)
- Transfer từ cặp ngôn ngữ "cha", **multilingual training**, **pivot translation**
- **Từ điển song ngữ khai thác tự động** rồi tiêm vào NMT (dictionary injection)
- **SMT nền tảng**: **IBM Model 1**, word alignment (fast_align / eflomal), phrase table, LM n-gram.
  ⚠️ SOLOAI 2025 **cho phép rõ ràng** hướng SMT. Với vài nghìn cặp câu, **SMT có thể vượt NMT train 30 phút.**
  Nhưng phải nhúng thành phần học được (§1.8) — bảng xác suất IBM Model 1 **là** tham số học từ dữ liệu ✅

*Phân loại văn bản:*
- **TF-IDF `char_wb` + mô hình tuyến tính** — baseline 10 phút. ⚠️ **Đo thật trên R-ViHSD**,
  cùng bộ fold không rò rỉ, OOF 51.663 dòng (`work/logs/mlpure.log`):

  | mô hình | OOF | % so với transformer |
  |---|---|---|
  | LogisticRegression | 0.6621 | **91.7%** |
  | LinearSVC (calibrated) | 0.6371 | 88.2% |
  | SGD modified_huber | 0.5843 | 80.9% |
  | ComplementNB | 0.5088 | 70.5% |
  | LightGBM trên feature luật | 0.4481 | 62.1% |
  | *ensemble 3 transformer* | *0.7220* | *100%* |

  → Con số đúng là **~92%, không phải 95%**, và **LogReg > LinearSVC** (ngược với trực giác thường gặp).
  Thiếu 8% là đủ để rớt hạng — dùng làm **lưới an toàn**, đừng dùng làm đích. 🔴
- LLRD, gradual unfreezing, multi-sample dropout
- **FGM / FreeLB / AWP**, **R-Drop**
- Focal loss, class-balanced loss, **LDAM**
- **Pseudo-labeling** (⚠️ đọc kỹ luật: *"không được dùng tập test dưới bất kỳ hình thức nào để huấn luyện"*
  → **pseudo-labeling trên test là VI PHẠM**. Chỉ làm trên dữ liệu train chưa nhãn nếu có.)
- Đa nhiệm: encoder chung + nhiều head, uncertainty weighting
- **TTA cho văn bản**: dự đoán trên nhiều biến thể chuẩn hoá rồi trung bình

*Tiếng Việt riêng:*
- Unicode **NFC**, tổ hợp vs dựng sẵn (nguồn bug âm thầm)
- Bỏ dấu / phục hồi dấu, teencode, gộp ký tự lặp, khử che ký tự (`*`, `.`, `_`)
- Tách từ (word segmentation) — cần cho PhoBERT, không cần cho ViSoBERT/XLM-R

**(c) Đọc — theo đúng thứ tự này**
1. Jay Alammar — *The Illustrated Transformer* (1h) 🔴
2. **Harvard NLP — *The Annotated Transformer*** (3h) 🔴🔴 *quan trọng nhất cả tầng: code từng dòng, đọc xong tự viết lại được*
3. Vaswani et al. (2017) *Attention Is All You Need*
4. Sutskever (2014) *Seq2Seq*; Bahdanau (2015) *Neural MT by Jointly Learning to Align and Translate*
5. Sennrich (2016) *Neural MT of Rare Words with Subword Units* (BPE) 🔴
6. Sennrich (2016) *Improving NMT Models with Monolingual Data* (back-translation) 🔴
7. Post (2018) *SacreBLEU*
8. **J&M SLP3 Ch.13 — Machine Translation** 🔴
9. Stanford **CS224n** lecture MT / attention / subword
10. Koehn *Statistical Machine Translation* Ch.4 (IBM Model 1 + alignment) 🟠 — cho phương án SMT
11. Provilkov (2020) *BPE-Dropout* 🟠 · Ott (2018) *Scaling NMT* (checkpoint averaging) 🟠

**(d) Tự kiểm tra**
- Vẽ sơ đồ Transformer decoder có causal mask và cross-attention, không nhìn tài liệu.
- Vì sao beam search cần length penalty? Chuyện gì xảy ra khi beam quá lớn?
- Với 3.000 cặp câu song ngữ, chọn NMT hay SMT? Lý do bằng số.
- Giải thích BP của BLEU phạt bản dịch ngắn như thế nào, và vì sao model hay dịch ngắn khi thiếu dữ liệu.

---

### ⬛ TẦNG 3 — THỊ GIÁC MÁY TÍNH · 16h · 🔴🔴 **TRỌNG TÂM SỐ 2**
> *Hai đề CV mẫu 2025 đều KHÔNG phải phân loại ảnh thường: video (ký hiệu) và cấu trúc không gian (jigsaw).
> Phải ôn cả ba nhánh: ảnh tĩnh, video, và tự giám sát/so khớp.*

**(a) Khái niệm nền**
- CNN: convolution, **receptive field**, stride/padding/dilation, pooling, translation equivariance.
- Kiến trúc: **ResNet** (residual), **EfficientNet** (compound scaling), DenseNet, ConvNeXt.
- **ViT**: patch embedding, tại sao ViT cần nhiều dữ liệu hơn CNN; Swin (cửa sổ trượt).
- Transfer learning cho ảnh; đóng băng BatchNorm khi dữ liệu ít.
- Segmentation: **U-Net**, **FPN**; Dice / IoU loss.
- Detection: anchor vs anchor-free, **NMS**, AP50.

**(b) Kỹ thuật nâng cao — danh mục đầy đủ**

*Huấn luyện ảnh (áp dụng được cho mọi bài CV):*
- **Bag of Tricks** (He 2018): cosine LR, label smoothing, **zero-γ init** cho residual cuối, no-bias-decay 🔴
- Augmentation: **RandAugment**, TrivialAugment, AutoAugment, **AugMix**
- **Mixup / CutMix / CutOut / GridMask / Copy-Paste** — ⚠️ *hại* với segmentation và anomaly detection
- **Progressive resizing** (train nhỏ → tinh chỉnh lớn): tiết kiệm thời gian rất nhiều trên T4 🔴
- **EMA / SWA**, checkpoint averaging
- **TTA**: hflip + đa tỉ lệ — gần như luôn dương, chi phí ~0 🔴
- Sampler cân bằng, focal loss, **dịch ngưỡng sau huấn luyện** — ⚠️ *đôi khi giúp, đôi khi hại*:
  ở R-ViHSD nó **làm tệ đi** task hate (0.7545 → 0.7525) là task chiếm 85% điểm. **Luôn kiểm bằng held-out.**

*Segmentation nâng cao:*
- U-Net++ / PAN / DeepLabV3+ (atrous), SegFormer
- Loss: **Dice + BCE**, **Tversky** (thiên recall), **Lovász-Softmax**, **boundary loss**
- **Deep supervision**, auxiliary head
- Hậu xử lý: connected components, **watershed** để tách instance dính nhau, `approxPolyDP`
- **Weighted Box/Mask Fusion (WBF)** để ensemble kết quả định vị

*Video / hành động — SOLOAI T2 dạng này 🔴:*
> 🔴 **QUY TẮC CẮT:** 14h không đủ để giỏi cả 3 nhánh CV. **Chọn ĐÚNG MỘT hướng video và đào sâu
> đến mức cài được từ trí nhớ**, phần còn lại chỉ đọc để nhận dạng đề:
> · **TSM** nếu đề cho video RGB thô · **ST-GCN** nếu đề cho (hoặc dễ trích) keypoint.
> Quyết định ở **Tuần 4** sau khi đọc lại 2 đề mẫu, rồi **không đổi nữa**. I3D / SlowFast /
> two-stream / TimeSformer hạ xuống 🟡 — biết tên, không cài.
- **Lấy mẫu khung hình**: dense vs **TSN sparse sampling** (chia video thành K đoạn, lấy 1 khung mỗi đoạn) 🔴
- **CNN + LSTM/GRU (CRNN)** — BTC 2025 phát sẵn baseline dạng này
- **3D-CNN**: C3D, **I3D**, R(2+1)D, **X3D**, **SlowFast**
- **TSM (Temporal Shift Module)** 🔴 — *đạt hiệu năng 3D-CNN với chi phí 2D-CNN; lựa chọn tốt nhất cho GPU yếu*
- **Two-stream** (RGB + optical flow) — flow đắt, cân nhắc kỹ với ngân sách 20 phút
- **Gộp theo thời gian**: average / max / **attention pooling** / **NetVLAD**
- **Hướng keypoint cho ngôn ngữ ký hiệu** 🔴: trích khung xương bàn tay/cơ thể → **ST-GCN** (graph conv
  không–thời gian). Cực nhẹ, cực mạnh cho cử chỉ. ⚠️ Kiểm tra luật: bộ trích keypoint dựng sẵn
  (MediaPipe/OpenPose) có bị coi là "dữ liệu/model ngoài" không → **hỏi BTC** (§11).
- Video transformer: TimeSformer, VideoMAE 🟡 (quá nặng cho 6 tiếng)

*Tự giám sát & so khớp — VOAI CK T2 (jigsaw) dạng này 🔴:*
> 🔴 **QUY TẮC CẮT:** đi **một đường duy nhất** — *đo tương thích cạnh bằng CNN Siamese* +
> *lắp ghép bằng greedy/Hungarian*. Đó là đường **thoả §1.8** (có thành phần học được) và
> chạy được trong 20 phút. SimCLR / MoCo / BYOL / DINO / MAE hạ xuống 🟡 — chúng cần
> pretraining dài, **không khả thi trong 6 tiếng**.
- **Pretext tasks**: **Jigsaw (Noroozi & Favaro)** 🔴, rotation prediction, context prediction (Doersch),
  colorization, **inpainting**
- **Contrastive**: SimCLR, MoCo, BYOL, **DINO**; masked image modeling: MAE, SimMIM
- **Metric learning**: Siamese network, **contrastive loss**, **triplet loss**, ArcFace
- **Đo tương thích cạnh (edge compatibility)** cho ghép ảnh:
  - Cổ điển: **Mahalanobis Gradient Compatibility (MGC)** — chuẩn vàng của jigsaw solver
  - Học được: CNN Siamese nhận 2 dải pixel biên → xác suất kề nhau ✅ *thoả điều khoản §1.8*
- **Thuật toán lắp ghép**: greedy theo độ tin cậy, **best-buddies**, Kruskal-like (rừng hợp nhất),
  **Hungarian** cho gán vị trí, ràng buộc vòng (loop constraint) để lọc cặp sai
- Mẹo: bài 3×5 chỉ có 15 mảnh → **có thể beam search / branch-and-bound trên không gian hoán vị**
  với hàm chi phí do CNN học

*Phát hiện bất thường (bạn đã có nền):*
- **PaDiM**, **PatchCore** (coreset), SPADE, **Reverse Distillation**, **DRAEM**, FastFlow/CFlow
- Mahalanobis, kNN; chọn tầng đặc trưng; hiệu chỉnh ngưỡng không nhãn (quantile / GMM 2 thành phần / Otsu)

**(c) Đọc**
1. Stanford **CS231n** notes: convolutional networks + training tricks 🔴
2. He et al. (2016) *ResNet*; Tan & Le (2019) *EfficientNet*
3. He et al. (2018) *Bag of Tricks* 🔴
4. Ronneberger (2015) *U-Net*; Lin (2017) *FPN*
5. **Noroozi & Favaro (2016)** *Unsupervised Learning by Solving Jigsaw Puzzles* 🔴🔴 *đọc kỹ — sát VOAI CK T2*
6. **Lin et al. (2019) *TSM: Temporal Shift Module*** 🔴 · Wang (2016) *TSN* · Carreira (2017) *I3D*
7. Yan et al. (2018) *ST-GCN* 🟠 (cho ngôn ngữ ký hiệu)
8. Chen (2020) *SimCLR* 🟠
9. Roth (2022) *PatchCore*; Defard (2021) *PaDiM* 🟠
10. Cho et al. — *A probabilistic image jigsaw puzzle solver* / Gallagher (2012) — nền tảng MGC 🟠

**(d) Tự kiểm tra**
- Cho video 60 khung, ngân sách 20 phút inference: chọn TSN-sparse hay dense sampling? Tính ra con số.
- Vì sao TSM đạt hiệu năng gần 3D-CNN mà chi phí như 2D?
- Với jigsaw 15 mảnh: không gian hoán vị bằng bao nhiêu? Chiến lược nào khả thi và **thành phần học ở đâu**?
- Vì sao mixup làm hỏng bài anomaly detection?

---

### ⬛ TẦNG 4 — HỌC MÁY LỒNG GHÉP · 4h · 🟠
> *ML không ra thành bài riêng mà nằm **bên trong** 2 tác vụ: lớp phân loại cuối, đặc trưng thủ công,
> hiệu chỉnh ngưỡng, ghép mô hình. Ôn vừa đủ, đừng ôn tabular Kaggle như bản 1 nói.*

**(a) Khái niệm nền**
- Bias–variance, chính quy hoá L1/L2, hồi quy tuyến tính/logistic, **SVM** (kernel), Naive Bayes.
- Cây → bagging → Random Forest → **boosting** → GBDT.
- Giảm chiều: PCA, SVD, UMAP/t-SNE. Phân cụm: KMeans, **GMM**.

**(b) Kỹ thuật nâng cao**
- **LightGBM / XGBoost / CatBoost**: histogram binning, leaf-wise vs level-wise, GOSS/EFB, DART.
  *Dùng khi bài có đặc trưng thủ công (độ dài câu, tỉ lệ ký tự, thống kê pixel…) — rất hay gặp ở nhiệm vụ con trọng số nhỏ.*
- **Target encoding out-of-fold** (in-fold = rò rỉ), CatBoost ordered target statistics.
- **Hiệu chỉnh xác suất**: Platt, isotonic, **temperature scaling** (rẻ nhất cho mạng nơ-ron).
- **Tối ưu ngưỡng** cho F1 / Balanced Accuracy trên OOF (coordinate search / quét lưới).
- **Ghép mô hình**: blending theo trọng số OOF, **rank averaging** (khi thang điểm khác nhau),
  **stacking 2 tầng**, **hill-climbing chọn tập con model** 🔴
- **Optuna (TPE)** + pruning ASHA — ⚠️ chỉ dùng khi có hàm mục tiêu offline đáng tin (bài học của chính bạn).
- Mất cân bằng: class weight, `scale_pos_weight`, focal, undersampling + bagging.
- **Adversarial validation**, permutation importance, **null importance** để chọn đặc trưng.

**(c) Đọc**
1. *ESL* Ch.9 (cây), Ch.10 (boosting), Ch.15 (rừng).
2. Ke et al. (2017) *LightGBM*; Chen & Guestrin (2016) *XGBoost*.
3. scikit-learn User Guide: `calibration`, `model_selection`.
4. Guo et al. (2017) *On Calibration of Modern Neural Networks* (temperature scaling) 🟠

**(d) Tự kiểm tra**
- Vì sao target encoding tính trong cùng fold gây rò rỉ? Mô tả cơ chế.
- Khi nào rank-average tốt hơn trung bình xác suất?

---

### ⬛ TẦNG 5 — KỸ THUẬT HẠ TẦNG & TỐI ƯU SUY LUẬN · 6h · 🔴
> *Tầng này tồn tại vì ràng buộc `Final/main.py ≤ 20 phút` (§1.4) và GPU Colab yếu.
> Bản 1 bỏ sót hoàn toàn.*

**(a) Khái niệm nền**
- Đo lường: thời gian/epoch, `torch.cuda.max_memory_allocated`, nghẽn cổ chai CPU vs GPU vs I/O.
- DataLoader: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`.

**(b) Kỹ thuật nâng cao**
- **AMP fp16** (T4 không có bf16) · **gradient checkpointing** (đổi tính toán lấy VRAM)
- **`torch.compile`** 🟡 (thời gian biên dịch có thể không đáng trong 6 tiếng)
- **Cache đặc trưng**: chạy encoder đóng băng **một lần**, lưu vector, rồi chỉ train head 🔴
  *Mẹo mạnh nhất khi GPU yếu — biến bài học sâu thành bài học máy nông.*
- **Batch inference** + `torch.inference_mode()` + `channels_last`
- **Lượng tử hoá động int8**, pruning, **distillation** để rút gọn model cho `main.py`
- **Cascade / early-exit**: model nhẹ xử lý đa số mẫu dễ, model nặng chỉ chạy mẫu khó
  → *cách duy nhất để "ensemble" mà vẫn dưới 20 phút* 🔴
- Checkpoint xuống Drive mỗi epoch (chống Colab ngắt kết nối)
- Cố định tính xác định: `cudnn.benchmark` (nhanh nhưng phá determinism) vs `use_deterministic_algorithms`

**(c) Đọc**
- PyTorch docs: *Performance Tuning Guide*, *Automatic Mixed Precision*, *Reproducibility* 🔴
- HuggingFace docs: *Efficient Training on a Single GPU*

**(d) Tự kiểm tra**
- Model của bạn cần bao nhiêu giây/ảnh để cả tập private chạy xong trong 20 phút? Tính ngược ra.
- Nếu ensemble 4 model làm vượt 20 phút, ba cách xử lý là gì?

---

### 📋 BẢNG TỔNG HỢP: KỸ THUẬT NÂNG CAO THEO MỨC ƯU TIÊN

**Nhóm 🔴 "phải có" — 18 kỹ thuật, ôn kỹ, cài đặt được:**

| Lĩnh vực | Kỹ thuật |
|---|---|
| Chung | Checkpoint averaging · EMA/SWA · Gradient accumulation + clipping · AMP fp16 · TTA |
| Chung | Tối ưu ngưỡng trên OOF · Hill-climbing chọn trọng số ensemble · Bootstrap SE |
| NLP/MT | Transformer Pre-LN tự viết · BPE/SentencePiece · Beam search + length penalty |
| NLP/MT | **Back-translation** · **BPE-dropout** · **Ensemble decoding** · Copy mechanism |
| NLP | TF-IDF char_wb + **LogReg** (baseline 10 phút, ~92% điểm transformer) · FGM · LLRD |
| CV | Bag of Tricks · Progressive resizing · **TSM hoặc TSN sparse sampling** (video) |
| CV | **Siamese/contrastive học độ tương thích** (jigsaw) · U-Net decoder tự viết |
| Hạ tầng | **Cache đặc trưng encoder đóng băng** · **Cascade/early-exit** |

**Nhóm 🟠 "nên có" — dùng khi bài toán cụ thể gọi tên:**
MBR decoding · Reranking n-best · SMT/IBM Model 1 · Lexically constrained decoding ·
FreeLB/AWP · R-Drop · ST-GCN · SimCLR · PatchCore/PaDiM · Lovász loss · WBF ·
Temperature scaling · Optuna · Uncertainty weighting đa nhiệm

**Nhóm 🟡 "biết là có" — đọc 1 lần, không cần cài:**
torch.compile · VideoMAE/TimeSformer · DETR · SegFormer · MoCo/BYOL · DART · nested CV

---

## PHẦN 4B — NGÂN SÁCH 4h/NGÀY & BẢNG ĐỐI CHIẾU TẦNG ↔ TUẦN
### *(bản 3 — tái ngân sách sau quyết định dồn toàn lực cho Olympic AI)*

### 4B.1 · Ngân sách tuần: **4 giờ/ngày × 7 = 28 giờ/tuần**

Bản 1 sai số học (56h tầng vs 47h tuần). Bản 2 sửa bằng cách **cắt nội dung**.
Bản 3 sửa bằng cách **tăng ngân sách** — và vì thế **khôi phục lại phần đã cắt**,
đồng thời nạp thêm ba mục vốn thiếu hoàn toàn (phân tích lỗi · trần Bayes · đọc baseline BTC).

| Hạng mục mỗi tuần | Giờ | Ghi chú |
|---|---|---|
| 📖 Đọc lý thuyết | **10h** | theo bảng 4B.2 |
| 💻 Cài đặt + bài tập có bộ chấm | **12h** | `tuan0X/` — mỗi tuần một thư mục tự chứa |
| 📓 Sổ assert (viết mới + gõ lại từ trí nhớ) | **2h** | §4C — từ Tuần 3 trở đi bắt buộc |
| 🧪 Thí nghiệm / tổng duyệt / phân tích lỗi | **3h** | |
| 🫙 Dự phòng | **1h** | *đừng lấp đầy — tuần nào cũng có việc tràn* |
| **Tổng** | **28h** | |

> **Tuần 8 giảm tải còn ~10h** (nghi thức hoá, không kiến thức mới).
> Tổng cả khoá: 7 × 28 + 10 = **206 giờ**.

### 4B.2 · Bảng đối chiếu — **số học đã kiểm, khớp tuyệt đối**

| Tầng | Giờ | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|---|
| **0** Đánh giá & PP thực nghiệm 🔴🔴 | **10h** | **10** | — | — | — | — | — | ôn |
| **1** Học sâu cốt lõi 🔴 | **14h** | **3** | **5** | **3** | **1** | **1** | **1** | ôn |
| **2** NLP & Dịch máy 🔴🔴 | **20h** | — | **6** | **7** | — | — | **4** | **3** |
| **3** Thị giác máy tính 🔴🔴 | **16h** | — | — | — | **9** | **7** | — | ôn |
| **4** ML lồng ghép 🟠 | **4h** | — | — | — | — | — | **4** | — |
| **5** Hạ tầng & tối ưu suy luận 🔴 | **6h** | — | — | — | — | **2** | **1** | **3** |
| **📖 Đọc mỗi tuần** | **70h** | **13** | **11** | **10** | **10** | **10** | **10** | **6** |

```
Tổng 6 tầng   = 10+14+20+16+4+6 = 70h
Tổng 7 tuần   = 13+11+10+10+10+10+6 = 70h     ✅ KHỚP

Tuần 1 đọc 13h: Tầng 0 (10h) + Tầng 1 khởi động (3h) = **§5 TORCH CƠ BẢN 2h**
+ d2l/Bag of Tricks 1h. §5 (`tuan01/TAI_LIEU.md`) là phần trả nợ muộn: bài tập Tuần 1–2 đòi 22 API torch
(`nn.Module` 15 lần, `Dataset` 11, `.view`/`.transpose`/`.contiguous`, `register_buffer`…)
mà trước đó KHÔNG tài liệu nào dạy — §6 mang tên "PyTorch nền" nhưng nội dung là
optimizer · AMP · tái lập, không phải cơ học tensor.
```

**Thay đổi so với bản 2 và lý do:**

| Tầng | Bản 2 | Bản 3 | Vì sao |
|---|---|---|---|
| 0 | 6h | **10h** | thêm §(c) *Vòng lặp cải tiến*: phân tích lỗi · trần Bayes · đọc baseline BTC — **ba mục trước đây có 0 dòng** |
| 1 | 10h | **12h** | gộp thẳng vào `tuan02/`–`tuan06/`, có bài tập + assert, không chỉ đọc |
| 2 | 16h | **20h** | trọng tâm số 1 (2/2 đề mẫu có dịch máy, BLEU trọng số 0.8) — bản 2 cắt nhầm chỗ |
| 3 | ~~12h~~ | **16h** | khôi phục, **nhưng cắt theo chiều sâu**: 1 nhánh video + 1 nhánh so khớp, phần còn lại 🟡 |
| 4 | 2h | **4h** | giữ mức thấp — ML không ra thành bài riêng |
| 5 | 4h | **6h** | ràng buộc `main.py ≤ 20 phút` là cổng loại trực tiếp |

### 4B.3 · BẢN ĐỒ CHUẨN — **78 đơn vị kiến thức → tuần nhận**

> Bảng này được **sinh ra bằng script dò thân bài Phần 5**, không phải tự khai.
> Mỗi dòng là một đơn vị kiến thức trong Phần 4; cột *Tuần* là tuần **thân bài thật sự nhắc tới nó**
> (trong 🔧 Kỹ thuật / 💻 Cài đặt / ✅ Nghiệm thu). **Không đơn vị nào để trống.**

**⬛ Tầng 0 — Đánh giá & PP thực nghiệm**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| macro/micro/weighted F1 | **1 · 4** |
| Balanced Accuracy · Youden's J | **1** |
| ROC-AUC vs PR-AUC | **1** |
| BLEU: clipping · brevity penalty | **1** |
| AP / AP50 · IoU matching | **1 · 4** |
| CV: stratified · group · leakage | **1** |
| Bootstrap SE | **1** |
| bias–variance · đa dạng ensemble | **1** |
| SacreBLEU signature | **1 · 3** |
| chrF / chrF++ | **3** |
| adaptive overfitting · reusable holdout | **1** |
| adversarial validation | **1** |
| tối ưu ngưỡng trên OOF | **1 · 4 · 5** |
| hill-climbing trọng số ensemble | **6** |
| 🔴 PHÂN TÍCH LỖI (mới) | **1** |
| 🔴 TRẦN BAYES / nhiễu nhãn (mới) | **1** |
| 🔴 ĐỌC BASELINE BTC (mới) | **1** |

**⬛ Tầng 1 — Học sâu cốt lõi**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| backprop · vanishing/exploding | **2** |
| AdamW vs Adam+L2 | **1 · 2** |
| warmup · cosine · Noam | **1 · 2** |
| LayerNorm vs BatchNorm | **1 · 2** |
| Xavier/He · zero-γ init | **2** |
| label smoothing | **2** |
| focal · dice · contrastive · triplet | **4** |
| transfer learning · gradual unfreezing | **6** |
| gradient accumulation + clipping | **2** |
| EMA · SWA · checkpoint averaging | **3** |
| LLRD | **6** |
| AMP fp16 + GradScaler | **1 · 3 · 4** |
| gradient checkpointing | **5** |
| 🔴 FGM / FreeLB / AWP | **3** |
| 🔴 R-Drop / consistency | **3** |
| uncertainty weighting đa nhiệm | **6** |

**⬛ Tầng 2 — NLP & Dịch máy**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| tokenization · BPE · SentencePiece | **2** |
| seq2seq · teacher forcing · exposure bias | **2** |
| attention: additive vs multiplicative | **2** |
| Transformer · PE · causal mask | **2** |
| beam search + length penalty | **2** |
| encoder pretrained: PhoBERT/XLM-R/ViSoBERT | **6** |
| 🔴 back-translation | **3** |
| BPE-dropout | **3** |
| ensemble decoding 🟠 | **3** |
| MBR · reranking n-best | **3** |
| copy mechanism / pointer-gen | **3** |
| char-level / byte-level | **3** |
| SMT · IBM Model 1 · alignment | **3** |
| TF-IDF char_wb + LogReg (lưới an toàn) | **6** |
| tiếng Việt: NFC · teencode · khử che | **1** |

**⬛ Tầng 3 — Thị giác máy tính**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| CNN · ResNet · EfficientNet | **1 · 4** |
| ViT · Swin | **1** |
| U-Net · FPN · Dice | **5** |
| detection: anchor · NMS · AP50 | **4** |
| Bag of Tricks · zero-γ · no-bias-decay | **1 · 2 · 4** |
| RandAugment · Mixup/CutMix (và khi hại) | **4** |
| progressive resizing | **4** |
| TTA đa tỉ lệ | **4** |
| TSN sparse sampling | **4** |
| CRNN | **4** |
| 🔴 TSM | **4** |
| 🔴 ST-GCN | **4** |
| jigsaw pretext | **5 · 8** |
| Siamese · contrastive · triplet | **5** |
| MGC | **5** |
| Hungarian · best-buddies · assembly | **5** |
| PaDiM · PatchCore | **5** |

**⬛ Tầng 4 — ML lồng ghép**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| LightGBM · XGBoost · CatBoost | **6** |
| target encoding OOF | **6** |
| temperature scaling · Platt · isotonic | **6** |
| stacking · rank averaging | **6** |
| permutation · null importance | **6** |
| Optuna TPE + ASHA | **6** |

**⬛ Tầng 5 — Hạ tầng & suy luận**

| Đơn vị kiến thức | Tuần nhận |
|---|---|
| đo GPU · bottleneck · env_report | **1** |
| DataLoader: workers · pin_memory | **5** |
| 🔴 cache đặc trưng encoder đóng băng | **5** |
| 🔴 cascade / early-exit | **5** |
| int8 · pruning · distillation | **5** |
| 🔴 main.py ≤ 20 phút | **5** |
| determinism · seed | **1** |

```
Tổng 78 đơn vị · đã có tuần nhận: 78 · còn trống: 0
```
---

### 4B.4 · Tầng 1 nằm Ở ĐÂU — **đã gộp thẳng vào thư mục từng tuần**

Tầng 1 là **bộ đồ nghề**, không phải một chủ đề. Trước đây nó đứng riêng ở một thư mục `tang1_toolkit/`
và **không dùng được** vì phải nhảy thư mục. Nay mỗi module đã nằm trong tuần thực sự dùng nó:

| Module | Nội dung | File | Vì dùng cho | Assert §4C |
|---|---|---|---|---|
| norm · init · gradflow 🔴 | LayerNorm vs BatchNorm · Xavier/He · grad flow Pre/Post-LN | `tuan02/…/06_norm_init_gradflow.py` | **lớp giải thích dưới Pre-LN** của BT 02 | **#10 · #11** |
| weight averaging | EMA · SWA · **checkpoint averaging** | `tuan03/…/06_weight_averaging.py` | 🔴 ensemble mặc định (suy luận ×1) | — |
| grad tricks | gradient accumulation · checkpointing · thứ tự AMP | `tuan03/…/07_grad_tricks.py` | batch lớn cho MT trên GPU yếu | **#20** |
| adversarial 🔴 | **FGM** · PGD · FreeLB | `tuan03/…/08_adversarial.py` | **+0,0121 OOF, 5/5 fold** · tấn công embedding ⇒ dùng được cho seq2seq | **#17** |
| consistency 🔴 | **R-Drop** · consistency · uncertainty weighting | `tuan03/…/09_consistency_multitask.py` | **+0,0047 OOF, 4/5 fold** · R-Drop (Wu 2021) **sinh ra cho NMT** | **#18** |
| focal loss | focal | `tuan04/…/06_focal_loss.py` | phân loại ảnh mất cân bằng | — |
| seg & match losses | dice · tversky · contrastive · triplet | `tuan05/…/06_seg_match_losses.py` | segmentation & Siamese cạnh (jigsaw) | — |
| finetune lr | LLRD · gradual unfreezing · freeze BN | `tuan06/…/06_finetune_lr.py` | fine-tune encoder pretrained | — |

▶️ **Cách dùng: mỗi tuần một lệnh, ngay trong thư mục tuần đó.** Không còn thư mục lẻ.

```bash
cd /home/namdp36/oai/tuan03
python3 -m pytest bai_tap/test_all.py -q          # khung: đỏ hết
SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q   # đáp án: xanh hết
```

**Tải mỗi module: ~45' đọc + ~45' code.** Giờ đọc nằm ở cột 📖 §4B.2; giờ code nằm trong quỹ 💻 12h/tuần.

> ⚠️ **`08_adversarial` và `09_consistency_multitask` là hai kỹ thuật DUY NHẤT sống sót qua đo đạc
> ở vòng trường** (2 trên 11 thứ đã thử). Chúng ở **Tuần 3** — cùng họ với 06/07, có bài thật
> (Ba Na) để đo ngay, và **cố ý tránh Tuần 6** vì tuần đó có TỔNG DUYỆT 1 chiếm trọn một ngày.
> Nếu Tuần 6 vỡ, cắt `06_finetune_lr` — module duy nhất còn lại ở đó.

### 4B.5 · Nếu tuần nào bị vỡ — thứ tự hy sinh

1. 🫙 Dự phòng (1h)
2. 🧪 Thí nghiệm (3h → 1h)
3. 📖 Đọc phần 🟡, rồi 🟠
4. 💻 Bài tập nhóm 🟠
5. **KHÔNG BAO GIỜ CẮT:** 📓 sổ assert · bài tập 🔴 · Tầng 0 §(c) · `08_adversarial` · `09_consistency_multitask`

---

## PHẦN 4C — SỔ ASSERT: THỨ THẬT SỰ MANG VÀO PHÒNG THI

### Nguyên tắc: **nhờ máy cái hỏng TO TIẾNG, thuộc cái hỏng ÂM THẦM**

Trong phòng thi có `deepseek-r1-distill-qwen-32b`. Đo thật trên code của chính chúng ta:

| hàm | dòng | ~token | lọt 2.000? |
|---|---|---|---|
| FGM (vòng trường 2026) | 18 | 260 | ✅ |
| DecoderLayer | 16 | 245 | ✅ |
| beam_search | 45 | 560 | ✅ |
| train_model | 33 | 569 | ✅ |
| stratified_group_kfold | 43 | 546 | ✅ |
| Seq2SeqTransformer | 43 | 657 | ⚠️ chật |

→ **Từng hàm đơn lẻ lọt thoải mái.** Nghĩa là bạn *không cần thuộc lòng phần cài đặt*.
Ràng buộc thắt không phải token mà là: **~40–60 câu hỏi cho cả 5 tiếng, chia cho 2 tác vụ và 3 người**
(§8.1), và **giờ thứ 6 LLM bị khoá** đúng lúc mở private test.

**Nhưng có một lớp lỗi không thể thuê ngoài.** Bằng chứng từ chính vòng trường 2026:

| Bug | Hậu quả | Code có chạy không? |
|---|---|---|
| Rò rỉ nhóm khi chia fold | OOF **thổi phồng 0,04** (0,7545 → 0,7067) | chạy hoàn hảo |
| Bias dùng train-prior thay test-prior | v8 được **0,697** thay vì 0,720 — **lặp 2 lần** | chạy hoàn hảo |

Và các bẫy kinh điển của seq2seq đều cùng chữ ký đó: causal mask sai → *train loss **đẹp hơn**, BLEU **sập***;
lệch `tgt_in`/`tgt_out` một ô → *loss giảm **rất đẹp**, BLEU **bằng 0***; quên `√d_k` → *model **vẫn chạy**, chỉ học kém*.

> **Những kỹ thuật này không hỏng bằng cách crash. Chúng hỏng bằng cách trả về một con số hợp lý.**
> DeepSeek sẽ đưa bạn code chạy được. Bạn **không thể** phân biệt FGM đúng với FGM sai bằng cách chạy nó.
> Tệ hơn: bản sai không chỉ làm mất kỹ thuật đó — nó **dạy bạn một bài học sai**
> ("FGM không giúp ở bài này") rồi bạn bỏ đi một thứ đáng **+0,0121 trên 5/5 fold**.

### Ranh giới quyết định

| ✅ Nhờ DeepSeek — sai là **thấy ngay** | ❌ Phải tự biết — sai vẫn **ra số đẹp** |
|---|---|
| Signature API, hằng số mặc định | Chiều của causal mask |
| Boilerplate sklearn, TF-IDF + LogReg | Dịch `tgt_in`/`tgt_out` một ô |
| Ghi CSV, đóng zip, `argparse` | FGM: tấn công tham số nào, khôi phục ra sao |
| Khung thuật toán kinh điển (Hungarian) | Chia fold có nhóm / chống rò rỉ |
| Sửa cú pháp ≤ 10 dòng | Hiệu chỉnh prior (train-prior vs test-prior) |
| Vẽ đồ thị, đọc/ghi file | Chiều KL trong R-Drop, áp lên head nào |

Trục phân loại **không phải** "thuật toán vs boilerplate" — mà là ***bản sai sẽ nổ, hay sẽ im lặng***.

### Hệ quả: thuộc **câu assert**, không thuộc phần cài đặt

```
test_model_is_causal             9 dòng  ~141 token
test_collate_pads_and_shifts     6 dòng  ~122 token
test_scaling_by_sqrt_dk          7 dòng  ~130 token
──────────────────────────────────────────────────
Seq2SeqTransformer (cài đặt)    43 dòng  ~657 token   → đắt gấp ~7 lần
```

Nhờ DeepSeek viết decoder, rồi **thả assert của mình lên nó** — 5 giây biết đúng hay sai.
Đó là lý do giá trị thật của `tuan01/` … `tuan06/` nằm ở **bộ test**, không ở đáp án.

---

### 📓 SỔ ASSERT — 22 mục nhóm 🔴, phải gõ được từ trí nhớ

> Mỗi mục: *chế độ hỏng âm thầm* → *câu kiểm tra*. Gõ lại toàn bộ sổ này **1 lần mỗi tuần** từ Tuần 3.

**Nhóm A — Seq2seq / Transformer**

| # | Kỹ thuật | Hỏng âm thầm kiểu gì | Assert |
|---|---|---|---|
| 1 | **Causal mask** | rò rỉ tương lai → train loss *đẹp hơn*, BLEU sập | `a=net(s,t); t2=t.clone(); t2[:,-1]=(t2[:,-1]+7)%V;`<br>`assert allclose(a[:,:-1], net(s,t2)[:,:-1])` |
| 2 | **tgt_in / tgt_out** | lệch 1 ô → model học chép đầu vào, BLEU 0 | `_,ti,to = collate([([5,6,7],[1,8,9,2])])`<br>`assert ti[0].tolist()==[1,8,9] and to[0].tolist()==[8,9,2]` |
| 3 | **Chia √d_k** | softmax bão hoà → gradient ≈ 0, học kém | `_,w = sdpa(q,k,v)  # d_k=64`<br>`assert -(w*(w+1e-12).log()).sum(-1).mean() > 0.5` |
| 4 | **Padding mask** | chú ý vào ô trống | `assert allclose(net(x5,t), net(pad_to(x5,8),t))`<br>⚠️ *phải đổi **lượng đệm**, không phải đổi nội dung ô pad* |
| 5 | **Tied embeddings** | quên buộc → thừa tham số, kém khi ít dữ liệu | `assert net.out.weight is net.tgt_emb.weight` |
| 6 | **Nhân `√d_model`** trước PE | tín hiệu vị trí lấn tín hiệu từ | `assert emb_scaled.std() > pe.std()` |
| 7 | **Beam length penalty** | lp=0 → thiên vị câu ngắn, cộng dồn với BP của BLEU | `assert len(beam(lp=1.5)) >= len(beam(lp=0.0))` |
| 8 | **beam=1 ≡ greedy** | beam cài sai vẫn ra câu hợp lý | `assert beam(k=1,lp=0.0) == greedy()` |
| 9 | **BPE theo rank** | quét trái→phải thay vì rank nhỏ nhất | `assert apply_bpe("abc",[("a","b"),("b","c")]) == ["ab","c","</w>"]` |
| 10 | **LayerNorm ≠ BatchNorm** | BN trong Transformer: thống kê ô nhiễm bởi padding, đổi theo batch — **không crash**, chỉ kém | `y1=norm(x)[0]; batch2=x.clone(); batch2[1:]=randn_like(batch2[1:])`<br>`assert allclose(norm(batch2)[0], y1)  # LN đúng; BN sẽ TRƯỢT` |
| 11 | **Pre-LN vs Post-LN** | Post-LN không warmup → gradient tầng đầu tắt, train phân kỳ *hoặc* chỉ học kém | `assert grad_norm_layer0(preln) > 10 * grad_norm_layer0(postln)` |

**Nhóm B — Đánh giá & chia dữ liệu** *(nơi mất nhiều điểm nhất ở vòng trường)*

| # | Kỹ thuật | Hỏng âm thầm kiểu gì | Assert |
|---|---|---|---|
| 12 | **Chia fold có nhóm** | rò rỉ → OOF thổi phồng **0,04** | `assert not (set(g[tr]) & set(g[va]))` |
| 13 | **Hiệu chỉnh prior** | dùng train-prior → lệch phân phối dự đoán | `assert abs(pred_dist - test_prior).max() < 0.05` |
| 14 | **Mọi đại lượng đã tune** | fit và eval cùng dữ liệu → gain ảo | nửa-fit / nửa-eval: `assert gain_eval < gain_fit` |
| 15 | **Trọng số ensemble trên OOF** | tối ưu trên chính tập chấm | như #12, cộng: `assert w.sum()≈1 and (w>=0).all()` |
| 16 | **Brevity penalty của BLEU** | quên BP → điểm ảo cho câu ngắn | `assert bleu(["a"],["a b c d"]) < bleu(["a b c d"],["a b c d"])` |

**Nhóm C — Huấn luyện**

| # | Kỹ thuật | Hỏng âm thầm kiểu gì | Assert |
|---|---|---|---|
| 17 | **FGM** | tấn công sai tham số / quên khôi phục | `w0=emb.weight.clone(); fgm.attack()`<br>`assert not equal(emb.weight,w0); fgm.restore(); assert equal(emb.weight,w0)` |
| 18 | **R-Drop / consistency** | KL sai chiều, áp nhầm head | `assert kl(p,p) < 1e-6` và `assert kl(p,q)≈kl(q,p)` *(bản đối xứng)* |
| 19 | **label_smoothing + ignore_index** | tính loss trên ô PAD → loãng | `assert loss(logits, all_pad_targets) == 0` |
| 20 | **AMP: GradScaler ↔ scheduler** | `sched.step()` khi scaler đã bỏ bước | `s0=scaler.get_scale(); scaler.step(opt); scaler.update()`<br>`if scaler.get_scale() >= s0: sched.step()` |

**Nhóm D — Hai cổng cuối, không có ngoại lệ**

| # | Ràng buộc | Assert |
|---|---|---|
| 21 | **`main.py` ≤ 20 phút** (§1.4) | `t0=time(); main(); assert time()-t0 < 1200` |
| 22 | **Tái lập** (quy chế bắt buộc) | `assert run(seed=42) == run(seed=42)` |

> 🔴 **Mục 19 và 20 phải chạy được ở giờ thứ 6 khi KHÔNG có DeepSeek.** Không thương lượng.

---

## PHẦN 5 — LỘ TRÌNH 8 TUẦN
### Mỗi tuần: **📖 đọc lý thuyết → 🔧 nắm kỹ thuật → 💻 cài đặt → ✅ nghiệm thu**

**Ngân sách (bản 3): 4 giờ/ngày × 7 = 28h/tuần** — xem chi tiết §4B.1.

| 📖 đọc | 💻 cài đặt | 📓 sổ assert | 🧪 thí nghiệm | 🫙 dự phòng | **tổng** |
|---|---|---|---|---|---|
| **10h** | **12h** | **2h** | **3h** | **1h** | **28h** |

> 📓 **Sổ assert (§4C) bắt buộc từ Tuần 3**: mỗi tuần gõ lại toàn bộ sổ từ trí nhớ **1 lần**.
> Đó là thứ duy nhất mang vào được giờ thứ 6 khi DeepSeek bị khoá.
**Từ tuần 2 trở đi: mọi thứ chạy trên Colab/Kaggle.** H100 chỉ dùng chuẩn bị dữ liệu.
**Từ tuần 3 trở đi: cấm ChatGPT/Claude khi luyện.** Tự giới hạn 2.000 token/phiên (§8).

---

### 🗓️ TUẦN 1 · 07–14/09 · NỀN TẢNG: ĐÁNH GIÁ + PYTORCH + ĐO MÔI TRƯỜNG THẬT

**📖 Lý thuyết (13h)** — **Tầng 0 toàn bộ** (10h, gồm §(c) *Vòng lặp cải tiến*)
+ Tầng 1 khởi động (3h): 🔴🔴 **§5 torch cơ bản (2h)** · d2l + Bag of Tricks (1h)
- 🔴🔴 **PyTorch basics: Tensors · Broadcasting · view/reshape · Autograd · nn.Module · DataLoader**
- Post (2018) SacreBLEU · Papineni BLEU · ESL Ch.7
- d2l.ai Ch.4–5 · He (2018) *Bag of Tricks*

**🔧 Kỹ thuật phải nắm tuần này**
> macro/micro/weighted F1 · Balanced Accuracy · Youden's J · **ROC-AUC vs PR-AUC (khi nào PR-AUC mới đúng)** ·
> **bias–variance · vì sao ensemble giảm phương sai · vì sao ĐA DẠNG > chất lượng đơn lẻ** · BLEU (clipping + brevity penalty + smoothing) ·
> AP50 · StratifiedGroupKFold · **bootstrap SE** · adversarial validation · adaptive overfitting ·
> tối ưu ngưỡng trên OOF · autograd · AdamW vs Adam+L2 · warmup/cosine/Noam · BatchNorm vs LayerNorm ·
> AMP fp16 + GradScaler · seed toàn cục & determinism

**💻 Cài đặt**
0. 🔴🔴 **`00_torch_basics.py`** — 7 hàm, mỗi hàm là một chỗ **bản sai vẫn chạy và vẫn ra số**:
   broadcasting `(L,)` vs `(D,)` khi L==D · `view` vs `reshape` sau `transpose` ·
   `register_buffer` vs `Parameter` · `.detach()` khi cộng dồn loss.
   `split_heads`/`merge_heads` chính là bước 2 và 4 của MultiHeadAttention Tuần 2.
1. **`train_loop.py` gõ từ số 0**, không framework cao cấp: AMP, scheduler, early stopping,
   best-checkpoint, log thời gian/epoch + `max_memory_allocated`. **Gõ lại 2 lần, lần 2 dưới 15 phút.**
2. Tự cài bằng numpy (không sklearn): `macro_f1`, `balanced_accuracy`, `bleu4`, `bootstrap_se`.
3. 🔴 **`error_analysis.py` — vòng lặp phân tích lỗi** (Tầng 0 §c1). Nhận `(y_true, y_pred, proba, texts)`,
   xuất: 50 mẫu sai nặng nhất · ma trận nhầm lẫn · **tách nhầm-đối-xứng (trần) khỏi nhầm-một-chiều (lệch prior)**.
   Chạy nó trên chính OOF vòng trường 2026 (`work/oof/`) và **tìm lại được** kết luận đã biết.
4. 🔴 **`bayes_ceiling.py`** (Tầng 0 §c2). Đếm mẫu **trùng input khác nhãn** → trần accuracy.
   Nghiệm thu: chạy trên `work/data/training_set.csv` phải ra **TEENCODE 78,5%** no-op
   (định nghĩa no-op: *văn bản nhiễu trùng khớp nguyên văn một câu `ORIGINAL` bất kỳ*).
   Đối chiếu: OBFUSCATION 9,4% · NO_DIACRITICS 7,0% · CHAR_REPEAT 1,7% · MIXED 1,4% · PUNCT_NOISE 0,0%.
   → **Chỉ TEENCODE có trần**; 5 nhãn còn lại thì không. Đó là kết luận phải tự rút ra.
5. **Đo môi trường thi** → `env_report.md`. Bắt buộc có ≥8 con số thật:
   GPU được cấp (T4/P100/L4?), VRAM, RAM, số core · thời gian 1 epoch `resnet34` @224 trên 3k ảnh (AMP on/off) ·
   thời gian fine-tune 1 epoch encoder ~100M trên 48k câu `max_len=96` · **thời gian `pip install` các gói hay dùng** ·
   quota GPU Kaggle còn lại · hành vi ngắt kết nối của Colab.

**✅ Nghiệm thu**
- [ ] 🔴 `00_torch_basics` xanh 8/8. Trả lời không nhìn: *`x` shape (B,L,D) với L==D, cộng
      `bias` shape (L,) — torch báo lỗi hay chạy?* và *khi nào `view` lỗi mà `reshape` không?*
- [ ] Gõ vòng lặp train từ trí nhớ < 15 phút, chạy đúng ngay lần đầu.
- [ ] Tính tay BLEU-4 một cặp câu, khớp với `sacrebleu`.
- [ ] Có `env_report.md` ≥ 8 con số đo thật.
- [ ] Trả lời được: *"Vì sao 'toàn nhãn 0' được 50 điểm ở vòng trường nhưng 0 điểm ở công thức 2025?"*
- [ ] 🔴 `bayes_ceiling.py` ra **78,5%** no-op cho TEENCODE và **< 10%** cho cả 5 nhãn còn lại.
- [ ] 🔴 Từ ma trận nhầm lẫn, chỉ đúng cặp nào là **trần** và cặp nào là **lệch prior**.
- [ ] 🔴 Đọc `task1_nlp_fpt26/` **như thể đó là baseline BTC** (Tầng 0 §c3): ghi ra kiến trúc ·
      augmentation · số epoch · cách chia val — trong **20 phút**, không chạy code.

---

### 🗓️ TUẦN 2 · 15–21/09 · TRANSFORMER & SEQ2SEQ — TỰ VIẾT TỪ ĐẦU

**📖 Lý thuyết (11h)** — Tầng 1 phần A (**5h**) + Tầng 2 đầu (**6h**)

*Tầng 1 phần A — 5h · nền của mọi thứ tuần này, đọc TRƯỚC Transformer:*
- **d2l.ai Ch.6–7** (tính toán sâu · **khởi tạo** · ổn định số học) + **Ch.11** (tối ưu nâng cao) — 2,5h
- **Goodfellow Ch.6–8** (backprop · đồ thị tính toán · vanishing/exploding · tối ưu hoá) — 2h
- **Loshchilov & Hutter (2019) AdamW** — 0,5h *(Bag of Tricks đã đọc ở Tuần 1)*

*Tầng 2 đầu — 6h:*
- *The Illustrated Transformer* → **The Annotated Transformer** (3h, đọc code từng dòng) 🔴🔴
- Bahdanau 2015 §3 · d2l.ai Ch.10–11 (attention) — 1,25h · Sennrich 2016 (BPE) — 0,75h
- *(chi tiết từng mục: `tuan02/TAI_LIEU.md` §1–§3)*

**🔧 Kỹ thuật phải nắm tuần này**

*Tầng 1 phần A — **đây là lớp giải thích nằm DƯỚI Pre-LN**, không phải chủ đề rời:*
> backprop & đồ thị tính toán · vanishing/exploding gradient · SGD+momentum vs Adam vs **AdamW**
> (vì sao weight decay ≠ L2 trong Adam) · **LayerNorm vs BatchNorm vs RMSNorm — vì sao Transformer
> buộc phải dùng LayerNorm** · khởi tạo **Xavier/He** · **zero-γ init** · dropout · **label smoothing**

*Tầng 2 đầu:*
> Encoder–decoder · teacher forcing · **exposure bias** · additive vs multiplicative attention ·
> multi-head · positional encoding · **causal mask** · cross-attention · **Pre-LN vs Post-LN** ·
> Noam schedule · **tied embeddings** · label smoothing 0.1 · gradient clipping · gradient accumulation ·
> BPE / SentencePiece unigram · vocab chia sẻ nguồn–đích · **beam search + length penalty**

**💻 Cài đặt**
0. 🔴 **`06_norm_init_gradflow.py`** (~1,5h code) — **làm TRƯỚC Transformer**.
   Chấm: `pytest bai_tap/test_all.py -q`.
   Ba thí nghiệm, mỗi cái kết thúc bằng một con số, không phải một câu chữ:
   - **a. LayerNorm tự viết** ≡ `nn.LayerNorm`. Rồi chứng minh **vì sao Transformer không dùng BatchNorm**:
     giữ nguyên câu thứ nhất, **đổi các câu KHÁC trong batch** → output của **BN đổi**, của **LN không đổi**.
     Với batch có padding và độ dài thay đổi, thống kê BN bị ô nhiễm bởi ô đệm. *(assert §4C #10)*
   - **b. Khởi tạo**: đo phương sai kích hoạt qua **20 tầng** với Xavier/He vs `randn()`.
     Xavier giữ phương sai ổn định; khởi tạo ẩu làm nó **nổ hoặc tắt** — và không hề crash.
   - **c. Pre-LN vs Post-LN**: dựng hai stack 12 tầng, đo `grad_norm` **tại tầng ĐẦU**.
     Post-LN nhỏ hơn Pre-LN ít nhất một bậc ⇒ **đó chính là lý do Post-LN cần warmup dài**,
     và là lý do BT 02 bắt dùng Pre-LN. *(assert §4C #11)*
   > Không có mục 0 này thì "Pre-LN ổn định hơn" chỉ là câu **học thuộc**. Có nó, đó là **số bạn tự đo**.
1. **Viết Transformer seq2seq từ số 0** (~300 dòng, Pre-LN, tied embeddings), **không dùng HuggingFace**.
   Train trên một cặp ngôn ngữ nhỏ bất kỳ. Đạt BLEU > 0 và giải thích được từng con số.
2. Cài **beam search** tay, có length penalty. So greedy vs beam=4 vs beam=8.
3. Train SentencePiece, khảo sát ảnh hưởng kích thước vocab (1k / 4k / 8k) lên BLEU với dữ liệu ít.

**✅ Nghiệm thu**
- [ ] 🔴 `06_norm_init_gradflow` xanh hết. **Nói được bằng SỐ**: BN đổi bao nhiêu khi batch đổi, LN đổi bao nhiêu;
      grad tầng đầu Post-LN nhỏ hơn Pre-LN bao nhiêu lần.
- [ ] 🔴 Trả lời không nhìn tài liệu: *"Vì sao Transformer dùng LayerNorm chứ không BatchNorm?"* —
      câu trả lời phải nhắc tới **padding + độ dài thay đổi**, không chỉ "vì nó chuẩn hoá theo feature".
- [ ] 📓 Khởi tạo **sổ assert (§4C)**: chép đủ 22 mục, tự gõ lại nhóm A một lần.
- [ ] Vẽ được sơ đồ decoder có mask, không nhìn tài liệu.
- [ ] Transformer tự viết chạy được, BLEU > 0.
- [ ] Giải thích được vì sao beam quá lớn làm BLEU giảm.

---

### 🗓️ TUẦN 3 · 22–28/09 · DỊCH MÁY ÍT TÀI NGUYÊN — 🔴 TUẦN QUAN TRỌNG NHẤT

**📖 Lý thuyết (10h)** — Tầng 2 phần giữa (7h) + Tầng 1: **weight averaging · grad tricks · FGM · R-Drop** (3h) 🔴
- **J&M SLP3 Ch.13 Machine Translation** · CS224n lecture MT/subword
- Sennrich (2016) back-translation · Provilkov (2020) BPE-dropout · Ott (2018) Scaling NMT
- Koehn *SMT* Ch.4 (IBM Model 1, word alignment)

**🔧 Kỹ thuật phải nắm tuần này**
> **Back-translation + iterative BT** · **BPE-dropout / subword regularization** · SwitchOut · word dropout ·
> **checkpoint averaging** · **ensemble decoding (trung bình log-prob mỗi bước)** · **MBR decoding** ·
> reranking n-best · noisy channel · **copy mechanism / pointer-generator** ·
> **lexically constrained decoding** · char-level & byte-level · pivot & multilingual ·
> **IBM Model 1 + word alignment + phrase table** · dictionary injection · sequence-level KD ·
> SacreBLEU signature & tokenizer · chrF++

**💻 Cài đặt**
0. 🔴 **`06_weight_averaging` · `07_grad_tricks` · `08_adversarial` · `09_consistency_multitask`** (~3h code). Làm TRƯỚC phần dịch máy bên dưới —
   `06_weight_averaging` và `08_adversarial` được dùng ngay ở mục 1–3 của tuần này.
   Chấm: `pytest bai_tap/test_all.py -q` (15 test).
   **Assert §4C #17 (FGM) · #18 (R-Drop) · #20 (AMP)**
   phải gõ được từ trí nhớ trước khi sang tuần 4.
1. **Giải trọn VOAI 2025 CK Tác vụ 1 (Ba Na → Việt), bấm giờ 3 tiếng trên Colab.**
   Đây là bài luyện sát đề nhất trong toàn bộ kế hoạch.
2. So sánh có số liệu trên **cùng** tập dữ liệu:
   NMT thuần · NMT + BPE-dropout · NMT + back-translation · **SMT (IBM Model 1 + LM)** · lai SMT-NMT.
   → **Kết luận bằng số: với ~N nghìn cặp câu thì hướng nào thắng.** Con số này bạn sẽ dùng trong phòng thi.
3. Cài **checkpoint averaging** và **ensemble decoding**, đo mức tăng BLEU.

**✅ Nghiệm thu**
- [ ] 🔴 Bốn module Tầng 1 (`06`–`09`) **xanh hết**, và đo được **FGM có dương trên bài Ba Na không** (đừng tin số của bài khác).
- [ ] 📓 **Gõ lại toàn bộ sổ assert (§4C) từ trí nhớ**, không nhìn — sai mục nào thì học lại mục đó.
- [ ] Có bảng so sánh 5 hướng dịch máy, kèm BLEU và thời gian train.
- [ ] Trả lời chắc: *"3.000 cặp câu → chọn gì? 30.000 cặp → chọn gì?"*
- [ ] Cài được copy mechanism cho tên riêng/chữ số.

---

### 🗓️ TUẦN 4 · 29/09–05/10 · CV NỀN + VIDEO / HÀNH ĐỘNG

**📖 Lý thuyết (10h)** — Tầng 3 nhánh ảnh + **một** hướng video (9h) + `06_focal_loss` (1h)
- CS231n notes (CNN + training) · He 2016 ResNet · Tan&Le EfficientNet · He 2018 Bag of Tricks
- **Lin (2019) TSM** 🔴 · Wang (2016) TSN · Carreira (2017) I3D · Yan (2018) ST-GCN

**🔧 Kỹ thuật phải nắm tuần này**
> Receptive field · residual · compound scaling · **detection: anchor vs anchor-free · NMS · AP50** · đóng băng BatchNorm khi ít dữ liệu ·
> RandAugment / AugMix · **Mixup / CutMix và khi nào chúng gây hại** · **progressive resizing** ·
> zero-γ init · no-bias-decay · **TTA đa tỉ lệ** ·
> **TSN sparse sampling** · **CNN+LSTM (CRNN)** · **3D-CNN: I3D / R(2+1)D / X3D / SlowFast** ·
> **TSM** · two-stream & optical flow · **attention pooling / NetVLAD** · **ST-GCN cho keypoint**

**💻 Cài đặt**
0. **`06_focal_loss.py`** (~45' code) — `pytest bai_tap/test_all.py -q`.
   Dùng ngay cho phân loại ảnh mất cân bằng.
1. **Giải SOLOAI 2025 Tác vụ 2 (nhận diện ngôn ngữ ký hiệu)** trên Colab, bấm giờ 3 tiếng.
   Nếu không lấy được dữ liệu gốc → dùng một tập video cử chỉ công khai làm thay.
2. So sánh 3 hướng **trên cùng ngân sách 20 phút inference**:
   khung đơn + gộp trung bình · CRNN · **TSM**. Ghi thời gian và Macro-F1.
3. Cài **TSN sparse sampling** và khảo sát K = 4/8/16 đoạn.

**✅ Nghiệm thu**
- [ ] `06_focal_loss` xanh; so focal vs CE trên tập ảnh mất cân bằng, ghi lại con số.
- [ ] 📓 **Gõ lại toàn bộ sổ assert (§4C) từ trí nhớ**, không nhìn — sai mục nào thì học lại mục đó.
- [ ] Có bảng so sánh 3 kiến trúc video kèm thời gian inference thật.
- [ ] Giải thích được cơ chế TSM bằng lời, không nhìn bài báo.
- [ ] Biết chính xác K khung/video là ngưỡng vượt 20 phút.

---

### 🗓️ TUẦN 5 · 06–12/10 · TỰ GIÁM SÁT, SO KHỚP, GHÉP ẢNH + TỐI ƯU SUY LUẬN

**📖 Lý thuyết (10h)** — Tầng 3 nhánh so khớp (7h) + Tầng 5 (2h) + `06_seg_match_losses` (1h)
- **Noroozi & Favaro (2016) Jigsaw** 🔴🔴 · Doersch (2015) context prediction · Chen (2020) SimCLR
- Gallagher (2012) / Cho — jigsaw solver & MGC
- Ronneberger U-Net · Lin FPN · Roth PatchCore
- PyTorch *Performance Tuning Guide* + *Reproducibility*

**🔧 Kỹ thuật phải nắm tuần này**
> **DataLoader: `num_workers` · `pin_memory` · `persistent_workers` · `prefetch_factor` · nghẽn CPU vs GPU vs I/O** ·
> Pretext tasks (jigsaw / rotation / inpainting / colorization) · contrastive (SimCLR/MoCo/BYOL/DINO) ·
> MAE/SimMIM · **Siamese + contrastive loss + triplet loss** · **MGC (Mahalanobis Gradient Compatibility)** ·
> **CNN học độ tương thích cạnh** · **best-buddies** · Kruskal-like assembly · **Hungarian** ·
> ràng buộc vòng · beam search trên hoán vị ·
> U-Net / FPN decoder tự viết · Dice/Tversky/Lovász · watershed tách instance · `approxPolyDP` ·
> PaDiM/PatchCore · hiệu chỉnh ngưỡng không nhãn ·
> **cache đặc trưng encoder đóng băng** 🔴 · **cascade / early-exit** 🔴 · gradient checkpointing ·
> lượng tử hoá int8 · distillation

**💻 Cài đặt**
0. **`06_seg_match_losses.py` — dice · tversky · contrastive · triplet** (~45' code) — `pytest bai_tap/test_all.py -q`.
   `contrastive`/`triplet` là nền của Siamese đo tương thích cạnh ở mục dưới.
1. **Giải VOAI 2025 CK Tác vụ 2 (ghép ảnh 3×5)** trên Colab, bấm giờ 3 tiếng.
   **Bắt buộc** có thành phần học được (CNN Siamese dự đoán cặp kề) — đây là bài luyện §1.8.
2. Viết **`main.py` inference-only** cho cả hai bài đã giải (tuần 3 + tuần 5),
   **đo và ép xuống dưới 20 phút**. Nếu vượt: áp dụng cache đặc trưng / cascade / rút gọn model.
3. Cài U-Net decoder trên encoder torchvision, gõ tay.

**✅ Nghiệm thu**
- [ ] `06_seg_match_losses` xanh; `triplet` dùng được trực tiếp cho Siamese cạnh.
- [ ] 📓 **Gõ lại toàn bộ sổ assert (§4C) từ trí nhớ**, không nhìn — sai mục nào thì học lại mục đó.
- [ ] Bài jigsaw có PPA > 0 và **chứng minh được thành phần ML thực sự tham gia quyết định**.
- [ ] Hai `main.py` đều chạy dưới 20 phút, sinh đúng `submission.csv`.
- [ ] Tính được không gian hoán vị 15 mảnh và giải thích chiến lược cắt tỉa.

---

### 🗓️ TUẦN 6 · 13–19/10 · ML LỒNG GHÉP + RÁP ĐỘI + TỔNG DUYỆT 1

**📖 Lý thuyết (10h)** — Tầng 4 (4h) + Tầng 2 phân loại văn bản (4h) + **finetune_lr: LLRD** (1h) + Tầng 5 (1h)
- ESL Ch.9/10/15 · LightGBM & XGBoost paper · sklearn `calibration`
- Guo (2017) temperature scaling

**🔧 Kỹ thuật phải nắm tuần này**
> **encoder pretrained tiếng Việt: PhoBERT (CẦN tách từ VnCoreNLP) · XLM-R · ViSoBERT (KHÔNG cần)** ·
> **TF-IDF `char_wb` + LogisticRegression — lưới an toàn 10 phút, ~92% điểm transformer (§Tầng 2)** ·
> LightGBM/XGBoost/CatBoost và tham số quan trọng · target encoding OOF · temperature scaling ·
> Platt/isotonic · **hill-climbing chọn tập con model** · rank averaging · stacking 2 tầng ·
> permutation & null importance · Optuna TPE + ASHA · uncertainty weighting đa nhiệm

**💻 Cài đặt + vận hành**
0. **`06_finetune_lr.py` — LLRD · gradual unfreezing · freeze BN** (~45' code) — `pytest bai_tap/test_all.py -q`.
   🫙 **Đây là module đầu tiên bị cắt nếu tuần vỡ** (§4B.3) — tuần này đã có TỔNG DUYỆT 1.
1. Chốt **phân vai 3 người / 2 máy / 1 DeepSeek** (§7). Dựng bảng theo dõi chung, giao thức hàng đợi LLM.
2. **TỔNG DUYỆT 1** — mô phỏng 100%: một ngày trọn vẹn 8h30–14h30, **2 tác vụ**,
   chỉ Colab/Kaggle, **chỉ DeepSeek 2k ngữ cảnh**, không mở code cũ, có `Final/` + báo cáo kỹ thuật.
   Đề: **SOLOAI 2025** (cả 2 tác vụ) — đề mẫu sát nhất mà bạn chưa giải trọn vẹn theo đúng luật.
3. **Post-mortem 90 phút** → `MOCK1.md`: cái gì vỡ, mất bao nhiêu phút ở đâu.

**✅ Nghiệm thu**
- [ ] `06_finetune_lr` xanh **hoặc** đã chủ động bỏ và ghi lý do vào `MOCK1.md`.
- [ ] 📓 **Gõ lại toàn bộ sổ assert (§4C) từ trí nhớ**, không nhìn — sai mục nào thì học lại mục đó.
- [ ] Cả 2 tác vụ có submission hợp lệ + `Final/` chạy dưới 20 phút + báo cáo kỹ thuật.
- [ ] `MOCK1.md` liệt kê ≥5 sự cố cụ thể kèm thời gian mất.

---

### 🗓️ TUẦN 7 · 20–26/10 · VÁ LỖ HỔNG + TỔNG DUYỆT 2 + ĐÓNG BĂNG

**📖 Lý thuyết (6h)** — ôn Tầng 2 (3h) + Tầng 5 (3h). Chỉ đọc lại phần `MOCK1.md` chỉ ra là yếu. **Không chủ đề mới.**

**🔧 Kỹ thuật**: không thêm mới. Củng cố nhóm 🔴 trong bảng tổng hợp Phần 4.

**💻 Cài đặt + vận hành**
1. Nửa đầu tuần: sửa đúng những gì `MOCK1.md` chỉ ra, **không làm gì khác**.
   (Kinh nghiệm: 80% thời gian mất nằm ở cài môi trường, lỗi định dạng CSV, hết VRAM, Colab ngắt,
   và đọc sót một ràng buộc trong đề.)
2. **Đóng băng thư viện chiến đấu** (§6) — sau tuần này không sửa nữa.
3. **TỔNG DUYỆT 2**: đề **VOAI 2025 CK** (cả 2 tác vụ), khó hơn, thêm một "khúc cua" cố ý.

**✅ Nghiệm thu**
- [ ] 📓 **Gõ lại toàn bộ sổ assert (§4C) từ trí nhớ**, không nhìn — sai mục nào thì học lại mục đó.
- [ ] Submission hợp lệ đầu tiên cho **cả 2 tác vụ trong vòng 60 phút**.
- [ ] Không lặp lại bất kỳ sự cố nào từ `MOCK1.md`.

---

### 🗓️ TUẦN 8 · 27–30/10 · NGHI THỨC HOÁ & GIẢM TẢI

> **Không kỹ thuật mới. Không thí nghiệm mới.** Tuần này chỉ để chắc chắn không mắc lỗi ngớ ngẩn.

- **T3 27/10 – T4 28/10** (≤2h/ngày): gõ lại toàn bộ thư viện chiến đấu từ trí nhớ, mỗi module 1 lần ·
  diễn tập `check_submission.py` cho 3 dạng đề khác nhau trong 10 phút ·
  diễn tập dựng `Final/` + báo cáo kỹ thuật, **bấm giờ, phải xong trong 25 phút**.
- **T5 29/10**: hậu cần (§9.1). Kiểm tra tài khoản, quota, đăng nhập sẵn. **Không code.**
- **T6 30/10**: **nghỉ hoàn toàn, ngủ đủ.** 6 tiếng ra quyết định liên tục là bài kiểm tra thể lực nhận thức;
  đội mệt sẽ mắc lỗi định dạng.
- **T7 31/10**: thi.

---

### 📌 Ghi chú: nếu cả 3 người cùng ôn

Tuần 1–2 **cả đội học chung** (nền tảng bắt buộc). Từ tuần 3 chuyên môn hoá:

| Người | Tuần 3–5 đào sâu | Vai ngày thi |
|---|---|---|
| **A** | Tầng 2 — NLP/Dịch máy | Máy 1 — Tác vụ NLP |
| **B** | Tầng 3 — CV/Video/Jigsaw | Máy 2 — Tác vụ CV |
| **C** | Tầng 0 + 4 + 5 (đánh giá, ghép mô hình, tối ưu suy luận) + **luyện hỏi LLM 2k** | Tham mưu · LLM · đóng gói |

Cả ba vẫn phải đọc **Tầng 0** và **Tầng 1** đầy đủ — đó là ngôn ngữ chung của đội.

---

## PHẦN 6 — THƯ VIỆN CHIẾN ĐẤU (thuộc lòng, gõ được không cần tra)

Mục tiêu **không phải là có file**, mà là **có trong đầu** — vì có thể không được mang code vào (§11.4),
và vì DeepSeek 2k ngữ cảnh không thể viết hộ bạn một module dài.

| # | Module | Đích | Dùng cho |
|---|---|---|---|
| 1 | `seed_everything()` đầy đủ (random/np/torch/cuda/PYTHONHASHSEED/worker_init_fn) | 2' | **Bắt buộc theo quy chế** |
| 2 | Vòng lặp train PyTorch (AMP, scheduler, early stop, best-ckpt, log thời gian) | 12' | Cả 2 tác vụ |
| 3 | **`main.py` inference-only ≤20 phút** (tự tìm data, nạp weights, ghi CSV) | 10' | **Bắt buộc theo đề** |
| 4 | `check_submission.py` (khớp ID, tên cột, giá trị hợp lệ, không trùng/thiếu) | 6' | Trước **mọi** lượt nộp |
| 5 | **TF-IDF `char_wb` + LogisticRegression** | **8'** | NLP — lưới an toàn (~92%, xem §Tầng 2) |
| 6 | Chuẩn hoá tiếng Việt (NFC, dấu, teencode, ký tự lặp, che ký tự) | 8' | NLP |
| 7 | **Transformer seq2seq Pre-LN + tied embeddings** | 20' | **Dịch máy** |
| 8 | **Beam search + length penalty** | 10' | Dịch máy |
| 9 | **Checkpoint averaging + ensemble decoding** | 8' | Dịch máy — tăng BLEU rẻ nhất |
| 10 | SentencePiece/BPE train + **BPE-dropout** | 6' | Dịch máy |
| 11 | **IBM Model 1 + word alignment** | 15' | Phương án SMT khi dữ liệu cực ít |
| 12 | Fine-tune encoder đa nhiệm + **FGM** | 15' | NLP phân loại |
| 13 | `Dataset` ảnh/video + augmentation + **TSN sparse sampling** | 10' | CV video |
| 14 | **TSM (temporal shift)** cắm vào ResNet2D | 10' | CV video |
| 15 | **Siamese/contrastive học độ tương thích cạnh** | 12' | CV jigsaw/so khớp |
| 16 | U-Net decoder trên encoder torchvision + Dice/Focal | 14' | CV segmentation |
| 17 | PaDiM/PatchCore tối giản + Mahalanobis | 15' | CV anomaly |
| 18 | Metrics numpy: macro-F1, BA, BLEU4, AP50 | 12' | Hàm mục tiêu offline |
| 19 | **Bootstrap SE** + quét ngưỡng tối ưu trên OOF | 8' | Quyết định tin hay không tin |
| 20 | **Hill-climbing chọn trọng số ensemble** trên OOF | 8' | Ghép mô hình |
| 21 | **Cache đặc trưng encoder đóng băng** | 6' | Cứu cánh khi GPU yếu |
| 22 | `StratifiedGroupKFold` + adversarial validation | 8' | Thiết kế CV |

**Cách luyện:** mỗi module gõ lại **3 lần** trong 8 tuần (tuần học → tuần 6 ôn → tuần 8 kiểm tra).
Bấm giờ. Quá đích 2 lần liên tiếp = chưa thuộc.

---

## PHẦN 7 — PHÂN VAI: 3 NGƯỜI · 2 MÁY · 1 DEEPSEEK

### 7.1 Nguyên tắc thiết kế

**Nút cổ chai không phải bàn phím, mà là (a) phiên GPU và (b) phiên LLM.**
Với 2 tác vụ và 2 máy, cách chia hiển nhiên là 1 người/1 tác vụ. Câu hỏi thật là:
**người thứ 3 làm gì cho có ích?**

Trả lời: người thứ 3 **không phải lập trình viên dự bị** — họ là **tham mưu trưởng**, và đây là vai
có ROI cao nhất trong đội, vì ba lý do:
1. **1 phiên DeepSeek không thể để 3 người dùng loạn xạ.** Phải có một người sở hữu và xếp hàng đợi.
2. **Hai sản phẩm bắt buộc — `Final/main.py` và báo cáo kỹ thuật — tốn 30–45 phút** mà A/B
   không thể vừa train vừa làm.
3. **Đọc kỹ đề là việc ROI cao nhất và dễ bị bỏ qua nhất khi vội.**

### 7.2 Bảng phân vai

| | **Người A — Máy 1** | **Người B — Máy 2** | **Người C — không máy cố định** |
|---|---|---|---|
| **Sở hữu** | Tác vụ NLP | Tác vụ CV | **Phiên DeepSeek** · bảng theo dõi · đồng hồ · `Final/` · báo cáo |
| T+0:00–0:20 | Đọc kỹ đề NLP, lướt đề CV | Đọc kỹ đề CV, lướt đề NLP | **Đọc kỹ CẢ HAI**, điền Bảng hợp đồng (§9.2) |
| T+0:20–1:00 | Baseline NLP → nộp lượt 1 | Baseline CV → nộp lượt 1 | Gác cổng 2 lượt nộp đầu · mở phiên LLM |
| T+1:00–3:30 | Cải tiến NLP | Cải tiến CV | Xếp hàng LLM · ghi bảng · thiết kế thuật toán **trên giấy** khi A/B chờ train |
| T+3:30–4:30 | Ghép mô hình, chốt cấu hình | Ghép mô hình, chốt cấu hình | **Dựng `Final/` + viết báo cáo** (mượn máy lúc A/B đang train) |
| T+4:30 | 🧊 **ĐÓNG BĂNG** | 🧊 **ĐÓNG BĂNG** | Chạy lại sạch · đo `main.py` < 20 phút |
| T+5:00–6:00 | Inference private | Inference private | Kiểm tra 5 lượt private · giữ lượt an toàn nhất |

### 7.3 Ba giao thức bắt buộc

**(1) Giao thức mượn máy.** Khi A hoặc B khởi động một lần train **> 8 phút**, họ **nhường bàn phím cho C**
trong thời gian đó. C dùng để gõ `main.py`, chạy `check_submission.py`, đóng gói zip.
**Không ai được sửa code của người khác.**

**(2) Giao thức hàng đợi LLM.** A/B **không bao giờ tự mở DeepSeek**. Họ ghi câu hỏi vào file chung
`hoi.md` (mỗi câu ≤ 40 từ). C nén xuống prompt chuẩn (§8), hỏi, dán kết quả vào `dap.md`.
C được quyền **từ chối** câu hỏi mà tự nghĩ nhanh hơn.

**(3) Giao thức nộp bài.** **Không lượt nộp nào rời máy mà chưa qua checklist của C** (§9.4).
Mỗi lượt phải được ghi vào bảng theo dõi **trước** khi nộp, không phải sau.

### 7.4 ⚡ Tối ưu ít người biết: 2 máy ≠ 2 GPU

**Phiên GPU gắn với *tài khoản*, không gắn với *máy*.** Nếu BTC cấp **3 tài khoản**
(FPT Smart Cloud / Colab), bạn có thể chạy **3 job huấn luyện song song trên 2 máy**:
mở profile trình duyệt thứ hai trên Máy 1, đăng nhập tài khoản của C, giao cho nó một job train dài.

→ **Xác nhận số tài khoản được cấp ngay ở buổi practice hôm trước.** Nếu được 3, đội bạn có
**+50% năng lực tính toán** so với đội chỉ dùng 2. Đây là lợi thế miễn phí và hầu như không ai khai thác.

---

## PHẦN 8 — CHIẾN THUẬT DÙNG DEEPSEEK 2.000 TOKEN

### 8.1 Hiểu đúng ràng buộc

- **2.000 token ngữ cảnh ≈ 1.400–1.600 từ tiếng Việt ≈ 70–90 dòng code — tính CẢ câu hỏi lẫn câu trả lời.**
- **Token tổng vô hạn** → chiến lược đúng là **rất nhiều phiên ngắn**, không phải một phiên dài.
- `deepseek-r1-distill-qwen-32b` là **mô hình suy luận**: nó tự sinh chuỗi `<think>` dài.
  **Nếu không ép ngắn, chuỗi suy luận ăn hết 2.000 token và bạn nhận về câu trả lời bị cắt giữa chừng.**
  Đây là cái bẫy số 1 và phần lớn đội sẽ dính.
- Ước tính thực tế: mỗi lượt hỏi–đáp mất 1,5–3 phút → **trong 5 tiếng chỉ dùng được ~40–60 câu hỏi có ích.**
  Đó là tài nguyên khan hiếm, phải phân phối.

### 8.2 Mẫu prompt bắt buộc

Mở **mọi** phiên bằng dòng này:

```
Trả lời cực ngắn. Không giải thích. Không suy luận từng bước. Chỉ xuất code Python.
<một câu hỏi duy nhất, ≤ 40 từ>
```

Nếu vẫn bị "nghĩ" dài, thêm: `Bắt đầu trả lời ngay bằng ```python`.

### 8.3 Dùng cho gì / không dùng cho gì

| ✅ **Dùng** | ❌ **Không dùng** |
|---|---|
| Nhớ lại signature API (`torch.nn.functional.ctc_loss` nhận gì?) | Thiết kế kiến trúc / chọn hướng giải |
| Viết một hàm ≤ 30 dòng, mô tả rõ input/output | Debug pipeline dài |
| Công thức toán, tên tham số, hằng số mặc định | Dán notebook hoặc traceback đầy đủ |
| Sửa lỗi cú pháp trên **≤ 10 dòng** code | Hỏi "code tôi sai ở đâu" kèm 200 dòng |
| Nhắc lại thuật toán kinh điển (beam search, Hungarian) | Việc bạn tự nghĩ ra nhanh hơn 2 phút |

**Khi có lỗi:** chỉ dán **dòng lỗi cuối + 3 dòng code liên quan**. Không bao giờ dán cả traceback.

### 8.4 Luyện tập (bắt đầu từ tuần 3)

1. **Tự áp trần 2.000 token** mỗi khi hỏi bất kỳ LLM nào trong lúc ôn. Đếm bằng `tiktoken` hoặc ước lượng từ.
2. **Chuẩn bị trước 20 câu hỏi mẫu** hay gặp nhất, viết sẵn dạng ≤ 40 từ, lưu vào `prompts.md`.
   Ví dụ: *"Viết beam search cho decoder PyTorch, có length penalty. Chỉ code."*
3. **Ghi lại câu trả lời tốt** vào `dap.md` để tái dùng — **không bao giờ hỏi lại cùng một câu**.
4. Trong hai buổi tổng duyệt, **Người C phải là người duy nhất chạm vào LLM.** Đó là kỹ năng riêng
   và cần luyện như mọi kỹ năng khác.

> **Quan trọng:** kế hoạch 8 tuần này tồn tại chính là để **giảm số câu bạn cần hỏi**.
> Mỗi module trong thư viện chiến đấu (§6) mà bạn thuộc là một câu hỏi không phải tiêu.

---

## PHẦN 9 — CHIẾN LƯỢC 6 TIẾNG & NGÀY THI

### 9.1 Ba quy tắc bất di bất dịch

1. **Quy tắc 1 giờ:** đến **T+1:00**, **cả 2 tác vụ** phải có submission hợp lệ.
   ⚠️ Lưu ý khác vòng trường: với công thức `(S−Min)/(Max−Min)`, baseline tầm thường cho **~0 điểm**,
   nên lượt nộp đầu chỉ để **kiểm tra định dạng**, không còn neo điểm. Vẫn phải làm — sai định dạng
   phát hiện lúc T+5:30 là thảm hoạ.
2. **Quy tắc trọng số:** đọc **công thức chấm trước** khi đọc mô tả bài toán. Dồn lực theo trọng số.
   (VOAI CK: BLEU = 0.8. Làm phân loại hoàn hảo mà dịch kém là thua.)
3. **Quy tắc đóng băng T+4:30:** cấm huấn luyện model mới. Chỉ đóng gói, chạy lại sạch, dựng `Final/`.

### 9.2 Bảng hợp đồng — Người C điền trong 20 phút đầu, cho **mỗi** tác vụ

```
Tác vụ:              [1 / 2]        Lĩnh vực: [NLP / CV]
Nhiệm vụ con A:      ..........     Trọng số: .....
Nhiệm vụ con B:      ..........     Trọng số: .....
Metric & công thức:  ..........     Min/Max của BTC có nêu không? .....
Ràng buộc mô hình:   pretrained cho phép = ..... (đọc file download_model.py nếu có)
Hợp đồng nộp:        cột = [...]  nhãn hợp lệ = [...]  số dòng = .....
Tên file:            [team]_[task]_pub.zip  chứa  ....._output.csv  + .ipynb
Số lượt nộp:         public ..... / private .....
Ngân sách inference: private có ..... mẫu → main.py phải xử lý ..... mẫu/giây
Khúc cua của đề:     ................................................
Baseline nộp được trong 15 phút: ................................
```

### 9.3 Dòng thời gian

| Thời điểm | A (NLP) | B (CV) | C (tham mưu) |
|---|---|---|---|
| **7:30** | Đăng nhập, đo GPU được cấp, `pip install` sẵn, **xác nhận số tài khoản** (§7.4) | ← | ← |
| **T+0:00–0:20** | Đọc đề NLP | Đọc đề CV | **Điền Bảng hợp đồng cả 2 tác vụ** |
| **T+0:20–1:00** | Bộ ghi submission **trước** model → nộp lượt 1 | ← | Gác cổng · mở phiên LLM |
| **T+1:00–2:30** | Baseline mạnh-rẻ (TF-IDF / SMT / khung đơn) | Baseline mạnh-rẻ | Bảng theo dõi · hàng đợi LLM |
| **T+2:30–3:30** | **Một** ý tưởng đòn bẩy cao nhất | ← | Thiết kế trên giấy · bắt đầu báo cáo |
| **T+3:30–4:30** | Đa dạng hoá + ensemble + hiệu chỉnh | ← | **Dựng `Final/`**, đo thời gian |
| **T+4:30–5:00** | 🧊 Chạy lại sạch notebook từ trên xuống | ← | Đóng gói · kiểm `main.py` < 20 phút |
| **T+5:00–6:00** | Private: **chỉ inference**. Lượt 1 = an toàn nhất | ← | Kiểm 5 lượt private · chốt bài cuối |

### 9.4 Checklist 30 giây trước **mỗi** lượt nộp (Người C chạy)

- [ ] Đúng số dòng, đúng tập ID (không thiếu / thừa / trùng)?
- [ ] Đúng tên cột, đúng thứ tự?  Giá trị chỉ trong tập nhãn hợp lệ?
- [ ] UTF-8, dấu phẩy, **không có cột index** (`to_csv(..., index=False)`)?
- [ ] Zip có **cả `output.csv` và `.ipynb`**?
- [ ] **Notebook chạy được tuần tự từ trên xuống** và sinh đúng file này?
- [ ] Đúng quy ước tên file? < 100MB?
- [ ] Đã ghi vào bảng theo dõi **trước** khi nộp?

### 9.5 Nguyên tắc ra quyết định (rút từ chính bài học của bạn)

- **Tin OOF, không tin bảng public.** OOF 51.663 dòng (SE ±0.003) đã dự báo đúng private **2 lần liên tiếp**;
  public 3.340 dòng (SE ±0.0116) lệch tới 0.02. Trước khi tin một cải thiện: **nó có lớn hơn 2×SE không?**
- **Cải tiến đơn lẻ hầu như luôn thất bại; đa dạng hoá mới thắng.** 9/10 cấu hình đơn lẻ của bạn nằm gọn
  trong nhiễu; chỉ ensemble 4 nhánh khác bản chất mới nhảy +2.5.
  → Kẹt thì **thêm nhánh khác bản chất**, đừng tinh chỉnh.
- **Đừng search trên hàm mục tiêu hỏng.** Proxy sai hướng 2 lần → vứt proxy, không chạy random search.
- ⚠️ **Ràng buộc mới:** mọi ensemble phải **chạy lọt 20 phút** trong `main.py`. Nếu không, dùng
  **cascade / early-exit** hoặc **cache đặc trưng** thay vì cắt nhánh.

### 9.6 Hậu cần (chuẩn bị T5 29/10)
- Thẻ sinh viên/CCCD · giấy trắng · bút (C cần **nhiều giấy** — thiết kế trên giấy là việc chính).
- Tài khoản BTC cấp: đăng nhập thử **ở buổi practice hôm trước**, không để đến sáng thi.
- ≥2 tài khoản Google dự phòng · Kaggle đã xác minh SĐT (bắt buộc để có GPU) · kiểm tra quota.
- Sạc, chuột, ổ cắm, nước, đồ ăn nhẹ.

### 9.7 Gói tái lập + báo cáo kỹ thuật (mẫu — C dựng từ T+3:30)

```markdown
# [Tên đội] — Tác vụ N — Báo cáo kỹ thuật
1. Tuân thủ: [chỉ dùng dữ liệu BTC ✓] [seed cố định ✓] [không sửa tay ✓]
   [không dùng test để train ✓] [pretrained: ..... nằm trong danh sách cho phép ✓]
   [có tham số học từ dữ liệu: ..... ✓]      ← điều khoản §1.8
2. Dữ liệu: file nào, checksum, không biến đổi ngoài pipeline
3. Phương pháp: kiến trúc, đặc trưng, loss, cách chọn ngưỡng (3–5 câu + 1 sơ đồ ASCII)
4. Đánh giá: sơ đồ CV, điểm OOF, sai số chuẩn bootstrap
5. Tái lập: `cd Final/Tac_vu_N && python main.py` → `submission.csv`, chạy hết ..... phút (< 20)
6. Đã thử và loại bỏ: ..... (phần này thể hiện chiều sâu — là điểm đánh giá định tính)
```

> Quy chế chấm theo *"bảng xếp hạng tự động **và các tiêu chí của cuộc thi**"*.
> **Báo cáo kỹ thuật là điểm, không phải thủ tục.** Bảng "đã thử và loại bỏ" trong `cv/RESULTS.md`
> của bạn chính xác là thứ ban giám khảo muốn thấy — giữ nguyên phong cách đó.

---

## PHẦN 10 — SỔ RỦI RO

| Rủi ro | Xác suất | Tác động | Phòng ngừa |
|---|---|---|---|
| **`main.py` vượt 20 phút** | **Cao** | Bài không hợp lệ | Đo ngân sách inference **từ khi chọn kiến trúc**; cascade / cache đặc trưng |
| Notebook không chạy tuần tự được → bài không hợp lệ | **Cao** | Mất trắng lượt nộp | Restart & Run All **trước mỗi lượt quan trọng** |
| Colab/FPT Cloud ngắt kết nối, hết quota | **Cao** | Mất tiến độ 1 tác vụ | Checkpoint mỗi epoch xuống Drive; tài khoản dự phòng |
| Bỏ trắng 1 tác vụ | Trung bình | −50% điểm cuối | **Quy tắc 1 giờ** (§9.1) |
| Câu trả lời LLM bị cắt do `<think>` ăn hết 2k | **Cao** | Mất thời gian, bực bội | Mẫu prompt §8.2 — luyện từ tuần 3 |
| Bài kiểu "thuật toán" bị loại vì thiếu ML | Trung bình | Mất trắng tác vụ | §1.8 — luôn nhúng thành phần học được, ghi rõ trong báo cáo |
| Sai định dạng CSV / thiếu `.ipynb` trong zip | Trung bình | Lượt nộp = 0 | Checklist §9.4, C gác cổng |
| Không tái tạo được khi hậu kiểm | Trung bình | Bị loại | Seed + lưu weights + chạy lại sạch ở T+4:30 |
| Hết VRAM (quen H100 80GB) | **Cao** | Mất 20–40 phút | Tuần 1 đo trước; mặc định batch nhỏ + gradient accumulation + AMP |
| Đuổi theo nhiễu bảng public | Trung bình | Chọn sai bài cuối | Quy tắc 2×SE; tin OOF |
| Dùng pseudo-label trên test → **vi phạm quy chế** | Thấp | Bị loại | *"Không được dùng test dưới bất kỳ hình thức nào để huấn luyện"* |

---

## PHẦN 11 — CÂU HỎI GỬI BTC NGAY TUẦN NÀY (`olpvietnam@vaip.vn`)

1. **Norm_score:** *"Nếu > 10 thì quy về 10"* — là **10** hay **100**? Trần điểm thực tế là bao nhiêu?
   Vòng miền 2026 dùng công thức `(S−Min)/(Max−Min)` như 2025 hay `SCORE/MAX` như vòng trường 2026?
2. Đề vòng miền có **2 tác vụ** (1 NLP + 1 CV) như 4 kỳ trước, hay tách thành 3 bài theo 3 lĩnh vực?
3. **Pretrained**: có danh sách model được phép như `download_model.py` của SOLOAI 2025 không?
   Cụ thể — **HuggingFace encoder tiếng Việt (PhoBERT/ViSoBERT/XLM-R) và mBART/NLLB có được dùng không?**
4. Thí sinh có được **mang mã nguồn/thư viện tiện ích tự viết sẵn** (GitHub/Drive) vào phòng thi không?
5. **Bộ trích keypoint dựng sẵn** (MediaPipe, OpenPose) có bị coi là "model/dữ liệu ngoài" không?
   *(quyết định hướng giải cho bài video/cử chỉ)*
6. Máy tính do BTC cấp hay **mang laptop cá nhân**? Đội 3 người được cấp **mấy máy, mấy tài khoản GPU**?
7. Cấu hình GPU trên FPT Smart Cloud (loại GPU, VRAM)? Có giới hạn giờ chạy không?
8. Xác nhận **5h public + 1h private**, **20 lượt public / 5 lượt private mỗi tác vụ**,
   và yêu cầu `Final/main.py ≤ 20 phút` có còn áp dụng không.
9. Truy cập DeepSeek: xác nhận **2.000 token ngữ cảnh/phiên**, không giới hạn số phiên?
10. **Quy định về thư viện:** đề vòng trường 2026 nhắc *"tuân thủ quy định về thư viện và tài nguyên
    của kỳ thi"* nhưng **quy định này chưa được công bố**. Xin cho biết: có danh sách thư viện
    được phép/bị cấm không? **PyTorch Lightning, HuggingFace Trainer, fastai, timm, Keras** có được dùng không?
    Bộ kit có kèm `requirements.txt` cố định môi trường không?
11. **Trọng số pretrained:** `timm.create_model(pretrained=True)` và `AutoModel.from_pretrained()`
    tải trọng số từ internet — có được phép không, hay chỉ được dùng model trong danh sách BTC phát
    (như `download_model.py` của SOLOAI 2025)? Môi trường thi **có internet mở** hay chỉ whitelist?

---

## PHẦN 12 — THEO DÕI & BẢNG TỔNG KẾT

**Kiểm tra mỗi Chủ nhật, 10 phút:**
- [Thông báo OlpAI](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/th%C3%B4ng-b%C3%A1o-olpai)
- [**Môi trường – Đề thi**](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/m%C3%B4i-tr%C6%B0%E1%BB%9Dng-%C4%91%E1%BB%81-thi) ← *"sẽ công bố tháng 9/2026"* — **trang quan trọng nhất**
- [Quy chế – Kế hoạch](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/quy-ch%E1%BA%BF-k%E1%BA%BF-ho%E1%BA%A1ch)

**Khi có đề mẫu / luật mới công bố: dừng lịch ôn, đọc kỹ, cập nhật tài liệu này trước.**

**Ngân sách: 4h/ngày = 28h/tuần** (10 đọc · 12 cài đặt · 2 sổ assert · 3 thí nghiệm · 1 dự phòng).
Tổng khoá **~208 giờ**. Số học đã kiểm: 68h đọc = 68h phân bổ (§4B.2).

| Tuần | Ngày | 📖 Đọc | Kỹ thuật trọng tâm | Sản phẩm |
|---|---|---|---|---|
| 1 | 07–14/09 | Tầng 0 (10h) + 🔴🔴 **§5 torch cơ bản (2h)** + d2l/BoT (1h) = **13h** | 🔴🔴 **tensor · broadcasting · autograd · nn.Module** · metrics, CV design, bootstrap · 🔴 **phân tích lỗi · trần Bayes · đọc baseline BTC** | 🔴 `00_torch_basics.py` · `train_loop.py` < 15' · `env_report.md` · `error_analysis.py` · `bayes_ceiling.py` |
| 2 | 15–21/09 | T1 phần A (5h) + T2 đầu (6h) = **11h** | Transformer, attention, BPE, beam search | Transformer tự viết, **BLEU > 40** · 📓 khởi tạo sổ assert |
| 3 | 22–28/09 | T2 giữa (7h) + Tầng 1: weight-avg · grad-tricks · **FGM · R-Drop** (3h) 🔴 | **Back-translation, BPE-dropout, ckpt averaging, SMT · FGM +0,0121 · R-Drop +0,0047** | **Giải VOAI CK T1 (Ba Na)** + bảng so sánh 5 hướng |
| 4 | 29/09–05/10 | T3 video (9h) + focal loss (1h) | **CHỌN 1: TSM *hoặc* ST-GCN** (§Tầng 3 quy tắc cắt), Bag of Tricks | **Giải SOLOAI T2 (ký hiệu)** + bảng 3 kiến trúc |
| 5 | 06–12/10 | T3 so khớp (7h) + T5 (2h) + seg/match losses (1h) | **Siamese cạnh + Hungarian**, cache đặc trưng, cascade | **Giải VOAI CK T2** + 2 `main.py` < 20' |
| 6 | 13–19/10 | T4 (4h) + T2 phân loại (4h) + **LLRD** (1h) + T5 (1h) | LightGBM, hill-climbing, temperature scaling, LLRD | Phân vai + **TỔNG DUYỆT 1** (SOLOAI) → `MOCK1.md` |
| 7 | 20–26/10 | ôn T2 (3h) + T5 (3h) = **6h** | củng cố nhóm 🔴 | **TỔNG DUYỆT 2** (VOAI CK) → `MOCK2.md`, đóng băng |
| 8 | 27–30/10 | — *(~10h, giảm tải)* | nghi thức | Mẫu báo cáo, hậu cần, **ngủ đủ** |
| — | **31/10** | — | — | **2 submission + `Final/` + báo cáo** |

> 📓 Từ Tuần 3, mỗi tuần **gõ lại toàn bộ sổ assert (§4C) từ trí nhớ 1 lần**. Đó là thứ duy nhất
> còn dùng được ở **giờ thứ 6**, khi DeepSeek bị khoá và private test vừa mở.

---

### Lời cuối

Bạn đã có thứ khó dạy nhất: **kỷ luật thực nghiệm** — đo sai số bằng bootstrap, từ chối tin bảng public,
phát hiện hàm mục tiêu hỏng rồi dừng lại. Đó là tư duy của người đi giải cao.

Thứ bạn thiếu là thứ **dễ dạy hơn nhiều** và nằm gọn trong 8 tuần:
**dịch máy** (tuần 2–3, xác suất ra cao nhất), **CV video và suy luận không gian** (tuần 4–5),
**trí nhớ cơ bắp PyTorch** để không phụ thuộc LLM, và **kỷ luật ngân sách 20 phút inference**.

Hai tuần quyết định là **tuần 3 (dịch máy)** và **tuần 5 (jigsaw + tối ưu suy luận)**.
Nếu phải cắt gì, đừng cắt hai tuần đó.
