# SỔ TAY PHƯƠNG PHÁP LUẬN THI ĐẤU
### Giải thích sâu 4 điều bạn đã làm đúng ở vòng trường — và cách áp dụng có chủ đích

> Tài liệu đi kèm `KE_HOACH_OLPAI26.md` (Phần 2).
> Nguồn số liệu: `work/REPRODUCE.md` (NLP) và `cv/RESULTS.md` (CV).

**Vì sao phần này quan trọng hơn kỹ thuật:** kỹ thuật (Transformer, TSM, PatchCore) là thứ ai cũng
tra được. Cái tách top 10% khỏi phần còn lại là **biết cải tiến nào là thật, cải tiến nào là nhiễu** —
vì trong 6 tiếng bạn chỉ ra được ~10–15 quyết định, và đi sai hướng 3 quyết định là hết giờ.

Bốn điều bạn làm đúng thực ra là **bốn nguyên lý có nền toán**. Bạn đã làm đúng bằng trực giác.
Tài liệu này biến trực giác thành **quy trình lặp lại được**, cộng thêm chỗ có thể làm tốt hơn.

---

## NGUYÊN LÝ 1 — ĐỘ PHÂN GIẢI CỦA THƯỚC ĐO
### *Mọi điểm số là biến ngẫu nhiên. Đo sai số trước, rồi mới tin.*

### 1.1 Bạn đã làm gì

| Tập | Cỡ mẫu | SE (bootstrap) |
|---|---|---|
| Public LB (NLP) | 3.340 | **±0.0116** |
| OOF (NLP) | 51.663 | **±0.003** |
| Public LB (CV) | 480 ảnh | **±1,3 điểm** |

Rồi kết luận: *"các mức 0.700–0.703 không phân biệt được về mặt thống kê"* → chọn model theo OOF.

### 1.2 Vì sao con số của bạn đúng — kiểm chứng lại

Sai số chuẩn tỉ lệ nghịch với `√n`:

```
SE_OOF / SE_public  ≈  √(3.340 / 51.663)  =  1/3,93
Thực đo:            0,003 / 0,0116        =  1/3,87   ✓
```

Khớp gần như hoàn hảo. **Bootstrap của bạn chạy đúng.** Đây là phép kiểm tra nên làm mỗi lần
bootstrap: nếu tỉ lệ SE không tuân luật `1/√n`, bạn đã cài sai (thường là quên resample theo *mẫu*
mà lại resample theo *fold*).

### 1.3 Cách tính — nhắc lại để tuần 1 cài lại

```python
def bootstrap_se(y_true, y_pred, metric_fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)          # lấy lại có hoàn lại
        scores[b] = metric_fn(y_true[idx], y_pred[idx])
    return scores.std(ddof=1)
```

Với metric phi tuyến (macro-F1, BA, AP50, BLEU) **bắt buộc dùng bootstrap** — không có công thức đóng.
Với accuracy thì có: `SE = √(p(1−p)/n)`.

### 1.4 ⬆️ NÂNG CẤP: bạn dùng SE biên, nên dùng **bootstrap ghép cặp**

Đây là chỗ bạn có thể làm tốt hơn ở vòng miền.

Khi so **hai model trên cùng một tập test**, hai vector dự đoán **tương quan mạnh** (chúng sai ở
những mẫu khó giống nhau). SE của *hiệu số* nhỏ hơn nhiều so với `√2 × SE` biên:

```python
def paired_bootstrap(y, pred_A, pred_B, metric_fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    d = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)          # CÙNG idx cho cả A và B  ← mấu chốt
        d[b] = metric_fn(y[idx], pred_A[idx]) - metric_fn(y[idx], pred_B[idx])
    return d.mean(), d.std(ddof=1), (d > 0).mean()   # hiệu, SE hiệu, P(A>B)
```

**Vì sao quan trọng:** SE biên ±0.0116 nói *"không phân biệt được 0.700 và 0.703"*.
Nhưng bootstrap ghép cặp thường cho SE hiệu chỉ **1/2 → 1/3** con số đó → bạn **phân biệt được**
những cải thiện mà SE biên tuyên bố là vô nghĩa. Với 20 lượt nộp hữu hạn, đây là chênh lệch lớn.

Đại lượng đáng dùng nhất là **`P(A > B)`** — xác suất A thật sự tốt hơn B. Quy tắc:
- `P > 0.95` → tin, đổi sang A
- `0.6 < P < 0.95` → chưa đủ, giữ cả hai để ghép mô hình
- `P < 0.6` → coi như bằng nhau, chọn cái **rẻ hơn khi inference** (nhớ trần 20 phút)

Với phân loại còn có **kiểm định McNemar** (rẻ hơn, chỉ đếm mẫu A đúng-B sai và ngược lại).

### 1.5 ⚠️ TOÁN QUAN TRỌNG NHẤT: chọn "tốt nhất trên public" bị thổi phồng bao nhiêu

Mỗi lượt nộp là **một lần truy vấn tập test**. Sau `k` lượt, chọn cái điểm cao nhất thì
kỳ vọng thiên lệch lên trên xấp xỉ:

```
bias  ≈  SE × √(2 ln k)
```

Thay số của chính bạn:

| Trường hợp | k | √(2 ln k) | SE | **Thiên lệch** | Khoảng dao động thật |
|---|---|---|---|---|---|
| NLP public | 20 | 2,45 | 0,0116 | **+0,028** | 0.702 ↔ 0.703 (0,001) |
| CV public | 11 | 2,19 | 1,3 điểm | **+2,85 điểm** | 76,3 ↔ 77,9 (1,6 điểm) |

**Đọc bảng này cho kỹ.** Ở cả hai bài, **thiên lệch do chọn lọc lớn hơn toàn bộ khoảng dao động
giữa các cấu hình.** Nghĩa là:

> Đỉnh `p = 0.42 → 77.9` trong `cv/RESULTS.md` **gần như chắc chắn là nhiễu**, không phải đỉnh thật.
> Bạn quét `p ∈ {0.34, 0.42, 0.50, 0.58}` và thấy hình chuông — nhưng một chuỗi 4 số ngẫu nhiên
> với SE 1,3 cũng tạo ra hình chuông y hệt.

Bạn đã kết luận đúng bằng trực giác (*"chọn cấu hình tốt nhất theo public là chọn theo nhiễu"*).
Giờ bạn có **con số** để nói điều đó với đồng đội lúc 2 giờ chiều khi cả đội đang muốn tin vào đỉnh giả.

### 1.6 Quy trình áp dụng ngày thi

1. **T+1:00** — ngay khi có OOF đầu tiên: chạy `bootstrap_se`, **viết SE lên giấy dán màn hình**.
   Con số này chi phối mọi quyết định 5 tiếng còn lại.
2. Trước khi tiêu một lượt nộp, hỏi: *"offline nó hơn bản tốt nhất bao nhiêu SE ghép cặp?"*
   Dưới 2 SE → **không nộp**, để dành lượt.
3. Cuối giờ, chọn bài nộp cuối theo **OOF**, không theo public. Ghi lý do vào báo cáo kỹ thuật.
4. Nếu OOF và public mâu thuẫn: tin tập **lớn hơn**, trừ khi nghi có rò rỉ (→ Nguyên lý 3).

---

## NGUYÊN LÝ 2 — BẬC THANG ĐỘ TIN CẬY CỦA HÀM MỤC TIÊU
### *Biết mình đang đứng bậc nào, trước khi quyết định có được phép search hay không.*

### 2.1 Bạn đã gặp gì

> *"`AUC_proxy` (test vs holdout-normal) dự đoán **sai hướng 4/4 lần**. Synthetic anomaly tự tạo thì
> bão hoà (AUROC 0,957 trong khi AUC thật ~0,86) vì lỡ tạo ra dễ hơn defect thật.
> ⇒ KHÔNG chạy random/beam search: search trên hàm mục tiêu hỏng chỉ giúp overfit nhanh hơn."*

Đây là quyết định **xuất sắc nhất** trong cả hai tài liệu, và là thứ đa số thí sinh làm ngược lại
(thấy điểm không lên thì tăng cường search).

### 2.2 Bậc thang — xác định bậc của mình trong 20 phút đầu

| Bậc | Hàm mục tiêu | Độ tin | Được phép làm gì |
|---|---|---|---|
| **1** | **OOF trên train có nhãn, chia fold đúng nhóm** | ★★★★★ | Tinh chỉnh siêu tham số, search, chọn model tự tin |
| **2** | Holdout cùng phân phối, đủ lớn | ★★★★ | Như bậc 1 nhưng cẩn thận với SE |
| **3** | Bảng public | ★★ | Chỉ dùng để *hiệu chuẩn*, tối đa 20 truy vấn, nhớ §1.5 |
| **4** | **Proxy tổng hợp / tự nghĩ ra** | ★☆ | **Phải kiểm định trước khi dùng.** Chưa kiểm định = cấm search |
| **5** | Trực giác chưa kiểm chứng | ☆ | Chỉ dùng cho các nước đi "gần như luôn dương" (§2.5) |

**Cấu trúc bài toán quyết định bậc bạn được đứng — không phải kỹ năng của bạn:**

- **Bài NLP vòng trường**: train 48.092 dòng **có nhãn** → bậc 1. Nên bạn iterate tự tin, OOF dự báo
  đúng private 2 lần.
- **Bài CV vòng trường**: train **chỉ có ảnh normal**, không có một nhãn anomaly nào → **không tồn tại
  bậc 1 hay 2**. Bị đẩy xuống bậc 4, và bậc 4 gãy.

> **Đây là bài học tổng quát hoá được:** đọc đề xong, câu hỏi thứ ba (sau metric và trọng số) phải là
> ***"tôi đang ở bậc mấy?"***. Nếu bậc 4–5, **kế hoạch 6 tiếng phải khác hoàn toàn**: ít thí nghiệm,
> nhiều nước đi an toàn, dựa vào tiên nghiệm thay vì đo đạc.

### 2.3 Vì sao proxy gãy — ba cơ chế

1. **Bão hoà (saturation).** Anomaly tổng hợp của bạn dễ hơn defect thật → AUROC 0,957 sát trần →
   không còn khả năng *xếp hạng* model. Một thước đo chạm trần không phân biệt được gì nữa.
2. **Đo nhầm đại lượng.** `AUC(test vs holdout-normal)` đo *"tập test có khác normal không"* —
   khác vì **bất cứ lý do gì**: nén JPEG, ánh sáng, thời điểm chụp. Không phải đo *"ảnh nào bất thường"*.
3. **Lệch phân phối.** Phân phối anomaly bạn tự sinh ≠ phân phối defect thật. Tối ưu trên cái thứ nhất
   là **Goodhart's law**: thước đo trở thành mục tiêu thì nó thôi là thước đo tốt.

### 2.4 Giao thức kiểm định proxy — làm trước khi tin

**Đừng hỏi "proxy này có hợp lý không". Hỏi "proxy này xếp hạng có đúng không".**

```
1. Lấy k ≥ 5 cấu hình đã biết điểm thật (từ các lượt nộp public đã tiêu).
2. Tính điểm proxy cho đúng k cấu hình đó.
3. Spearman ρ(proxy, thật).
       ρ > 0,7   → dùng được, nhưng vẫn kiểm lại sau mỗi 5 lượt
   0,3 < ρ < 0,7 → chỉ dùng để LOẠI cấu hình rất tệ, không dùng để chọn cấu hình tốt nhất
       ρ < 0,3   → VỨT. Không search. Chuyển sang §2.5.
```

**Đây là khoản đầu tư hợp lý:** tiêu 4–5 lượt nộp trong 90 phút đầu **chỉ để hiệu chuẩn proxy**.
Nghe lãng phí, nhưng nếu ρ > 0.7 bạn mua được quyền search trong 3 tiếng còn lại bằng 5 lượt.
Nếu ρ < 0.3, bạn tiết kiệm được 3 tiếng đi sai hướng. Bạn đã trả giá này rồi — 4/4 lần sai —
chỉ là trả một cách bị động thay vì chủ động.

### 2.5 Khi không có hàm mục tiêu đáng tin: dùng tiên nghiệm

Không đo được thì **đừng đoán** — hãy chơi những nước **gần như luôn dương** theo kinh nghiệm cộng đồng,
không cần proxy xác nhận:

| Nước đi | Vì sao gần như luôn dương | Rủi ro |
|---|---|---|
| **Ensemble nhiều nhánh khác bản chất** | Giảm phương sai, xem Nguyên lý 4 | Chi phí inference |
| **TTA (lật ngang, đa tỉ lệ)** | Giảm phương sai, không đổi bias | ~0 |
| **Checkpoint averaging / EMA / SWA** | Làm phẳng cực tiểu | ~0 |
| **Train lâu hơn + augmentation mạnh hơn** | Giảm overfit khi dữ liệu ít | Thời gian |
| **Dùng nhiều fold hơn cho ensemble** | Vừa giảm phương sai vừa dùng hết dữ liệu | Thời gian |
| **Độ phân giải cao hơn** (CV) | Thường dương... | ⚠️ **bạn đã đo: KHÔNG dương** ở bài đó |

Dòng cuối là ví dụ tốt: bạn thử 768/770/1022 và **đo được là không giúp**. Đó là lý do vẫn phải đo —
tiên nghiệm chỉ là thứ tự ưu tiên, không phải chân lý.

Chính bạn đã đi đúng đường này: khi proxy gãy, bạn không search — bạn **ensemble**. Và +2,5 điểm.

---

## NGUYÊN LÝ 3 — RÒ RỈ & THIẾT KẾ CHIA FOLD
### *Một sơ đồ CV sai làm hỏng toàn bộ 5 tiếng còn lại, và nó im lặng.*

### 3.1 Bạn đã gặp gì

> *"Tập train ghép cặp (ORIGINAL + một bản nhiễu của cùng comment), nếu không nhóm đúng thì
> CV bị rò rỉ và **cao giả ~0,04**."*

Bạn phục hồi nhóm bằng **so khớp nearest-neighbour char n-gram** (`src/build_groups.py`),
rồi dùng `StratifiedGroupKFold`.

### 3.2 Cơ chế — vì sao rò rỉ này giết chết mọi thứ

Bài R-ViHSD sinh dữ liệu bằng cách lấy comment gốc rồi tạo bản nhiễu (teencode, bỏ dấu…).
Nếu chia fold ngẫu nhiên:

```
Fold train:  "thằng này ngu vl"        (ORIGINAL, nhãn HATE)
Fold val:    "thg nay ngu vl"          (TEENCODE, cùng comment, cùng nhãn HATE)
```

Model không cần *học khái niệm hate speech* — nó chỉ cần **nhớ nội dung** rồi khớp mờ.
Điểm val vọt lên, nhưng năng lực tổng quát hoá **không hề tăng**.

**Mức độ nghiêm trọng, đo bằng chính SE của bạn:** rò rỉ thổi phồng 0,04 trong khi SE của OOF
là 0,003 → **hơn 13 lần sai số chuẩn**. Nếu không phát hiện, bạn sẽ:
- chọn siêu tham số tối ưu cho việc *ghi nhớ*, không phải *tổng quát hoá*
- tin rằng mình đang ở 0,75 trong khi thật ra 0,71
- và **mọi so sánh A/B trong 5 tiếng đều vô nghĩa**, vì cả A lẫn B đều được thưởng vì ghi nhớ

### 3.3 Bảng phân loại rò rỉ — quét đủ 6 loại trong 20 phút đầu

| # | Loại | Dấu hiệu trong đề | Cách chữa |
|---|---|---|---|
| 1 | **Trùng lặp / gần trùng** | Dữ liệu sinh bằng augmentation, có cột `noise_type`/`variant`, số mẫu là bội của số nguồn | Nhóm theo mẫu gốc → `GroupKFold` |
| 2 | **Cùng nguồn** | Nhiều ảnh/khung hình từ cùng bệnh nhân, cùng video, cùng phiên chụp, cùng người ký hiệu | Nhóm theo nguồn |
| 3 | **Thời gian** | Có cột ngày/giờ, hoặc test là giai đoạn sau | `TimeSeriesSplit`, không shuffle |
| 4 | **Rò rỉ nhãn qua đặc trưng** | Target encoding, thống kê theo nhóm tính trên cả tập | Tính **out-of-fold** |
| 5 | **Tiền xử lý** | Fit scaler / TF-IDF / PCA / normalize trên train+val | Fit **bên trong** từng fold |
| 6 | **Augmentation** | Bản tăng cường rơi khác fold với bản gốc | Nhóm theo ảnh gốc |

> **Với đề OlpAI cụ thể:** loại **1** và **2** là nguy hiểm nhất.
> - Bài ngôn ngữ ký hiệu (SOLOAI T2): nhiều khung hình từ **cùng một video**, và nhiều video từ
>   **cùng một người ký hiệu**. Chia fold ngẫu nhiên theo *khung hình* là rò rỉ nặng.
>   → Phải nhóm theo **video**, tốt hơn nữa là theo **người thực hiện**.
> - Bài ghép ảnh (VOAI CK T2): nhiều mảnh từ **cùng một ảnh gốc** → nhóm theo ảnh.
> - Bài dịch máy: câu trùng lặp hoặc gần trùng trong corpus → khử trùng lặp trước khi chia.

### 3.4 Bốn phép thử phát hiện rò rỉ — chạy trong 10 phút

**(a) Phép thử khoảng cách (rẻ nhất, luôn làm).**
Nếu `CV − public > 3 × SE` một cách nhất quán → nghi rò rỉ.
*Của bạn: 0,04 lệch với SE 0,003 = 13 SE. Cờ đỏ rực.*

**(b) Quét trùng lặp.**
```python
# văn bản: TF-IDF char 3-5gram + cosine, hoặc MinHash/LSH khi dữ liệu lớn
# ảnh:     perceptual hash (pHash) hoặc kNN trên embedding
# so số cụm với số dòng — lệch nhiều nghĩa là phải nhóm
```
*Đây chính là `src/build_groups.py` của bạn. Đóng gói lại thành module dùng chung cho cả CV lẫn NLP.*

**(c) Adversarial validation.** Train classifier phân biệt train vs test.
- AUC ≈ 0,5 → cùng phân phối, yên tâm
- AUC → 1,0 → lệch phân phối; **đặc trưng quan trọng nhất của classifier đó chỉ thẳng vào chỗ rò rỉ**

**(d) Phép thử xáo nhãn.** Xáo ngẫu nhiên nhãn rồi train. Điểm CV **phải** rơi về mức ngẫu nhiên.
Nếu vẫn cao → có rò rỉ hoặc có bug trong pipeline đánh giá.

### 3.5 ⬆️ NÂNG CẤP: nhóm theo *lớp phân cấp*

Bạn nhóm theo comment gốc — đúng. Nhưng nhiều bài có **nhiều tầng nhóm** cùng lúc:

```
người ký hiệu  ⊃  video  ⊃  khung hình
ảnh gốc        ⊃  mảnh ghép
người viết     ⊃  comment gốc  ⊃  biến thể nhiễu
```

**Nhóm theo tầng CAO NHẤT mà tập test cũng tách theo tầng đó.** Cách xác định: đọc đề xem test
được xây thế nào. Nếu private test dùng **người ký hiệu chưa từng thấy**, mà bạn chỉ nhóm theo video,
CV của bạn vẫn lạc quan giả — chỉ là ít hơn.

Đây là câu hỏi nên đưa vào **Bảng hợp đồng** (Phần 9.2 của kế hoạch):
> *"Tập private khác tập train ở chiều nào? Nhóm CV của tôi có tách theo đúng chiều đó không?"*

---

## NGUYÊN LÝ 4 — ĐA DẠNG HOÁ THẮNG TINH CHỈNH
### *Sàn của ensemble do tương quan quyết định, không phải do số lượng model.*

### 4.1 Bạn đã gặp gì

```
Cấu hình đơn lẻ:  74,2 · 76,0 · 76,3 · 76,5 · 77,1 · 77,5 · 77,5 · 77,9   ← 9/10 nằm trong nhiễu
Ensemble 4 nhánh: 80,4                                                     ← +2,5, bước nhảy thật
```

Và bạn ghi đúng bản chất: *"**giảm phương sai, không phải model tốt hơn**"*.

### 4.2 Công thức — vì sao tinh chỉnh vô ích còn đa dạng hoá thì không

Với `M` model, phương sai sai số `σ²`, **tương quan trung bình từng cặp `ρ`**:

```
Var(ensemble)  =  σ² · [ ρ  +  (1 − ρ)/M ]
```

Cho `M → ∞`: `Var → ρσ²`. **Sàn do `ρ` đặt ra, không phải `M`.**

| ρ | M=2 | M=4 | M=6 | M=∞ | Ý nghĩa |
|---|---|---|---|---|---|
| 0,95 | 0,987 | 0,981 | 0,979 | 0,975 | *"đổi độ phân giải, đổi seed"* → **vô ích** |
| 0,70 | 0,922 | 0,880 | 0,866 | 0,837 | cùng họ kiến trúc, khác cấu hình |
| 0,50 | 0,866 | 0,791 | 0,764 | 0,707 | khác họ kiến trúc |
| 0,30 | 0,806 | 0,689 | 0,648 | 0,548 | khác **biểu diễn đầu vào** |

*(bảng ghi hệ số nhân của **độ lệch chuẩn** = `√[ρ + (1−ρ)/M]`)*

**Đọc bảng theo hàng, không theo cột.** Đi từ M=2 → M=6 ở ρ=0,95 chỉ giảm 0,8%.
Đi từ ρ=0,95 → ρ=0,50 ở cùng M=4 giảm **19%**. **Giảm ρ đáng giá hơn tăng M cả một bậc.**

### 4.3 Số liệu của bạn khớp hoàn hảo với lý thuyết

Nhìn lại `cv/RESULTS.md`:

| Lượt | Thay đổi | Kết quả | ρ với bản gốc |
|---|---|---|---|
| 6 | dinov2b-518 → **dinov2b-770** | 77,9 → 77,5 | **rất cao** — cùng model, chỉ đổi độ phân giải |
| 10 | → **dinov2l** đa tầng | 77,5 | **rất cao** — cùng họ DINOv2 |
| 12 | ensemble 6 (+ViT-Giant) | *chưa có* | — |
| **11** | **ensemble 4 nhánh khác bản chất** | **80,4** | **thấp** ← đây |

Lượt 6 và 10 thất bại **không phải vì model tệ**, mà vì chúng **quá giống** cái đã có → `ρ ≈ 0,95` →
theo bảng trên, gộp vào chỉ giảm ~2% phương sai, chìm nghỉm trong SE 1,3 điểm.

**Dự đoán từ lý thuyết:** lượt 12 (ensemble 6, thêm ViT-Giant và dinov2b-ml) **sẽ cải thiện rất ít**
so với lượt 11 — vì hai nhánh thêm vào lại cùng họ ViT, `ρ` cao. Đi từ M=4→6 ở ρ=0,7 chỉ được 1,6%.
*Nếu bạn còn log điểm lượt 12, kiểm chứng lại — đó là bài kiểm tra tốt cho mô hình tư duy này.*

### 4.4 Thang nguồn đa dạng — xếp theo mức giảm ρ

| Hạng | Nguồn đa dạng | Giảm ρ | Ví dụ cho đề OlpAI |
|---|---|---|---|
| 🥇 | **Khác biểu diễn đầu vào** | **Nhiều nhất** | Ký hiệu: **pixel vs keypoint/khung xương** · Jigsaw: **màu biên vs gradient biên vs kết cấu** · NLP: **char n-gram vs subword vs từ** |
| 🥈 | **Khác họ kiến trúc** | Nhiều | CNN vs ViT vs **GBDT trên đặc trưng thủ công** |
| 🥉 | **Khác hàm mục tiêu** | Vừa | CE vs focal vs contrastive/triplet |
| 4 | Khác cách chia dữ liệu | Vừa-ít | fold khác, augmentation khác |
| 5 | Khác seed | **Ít nhất** | gần như miễn phí nhưng yếu |

> **Quy tắc hành động:** khi kẹt, đừng hỏi *"model nào mạnh hơn"* — hỏi
> ***"cách nhìn dữ liệu nào mà tôi CHƯA dùng"***. Một nhánh yếu hơn nhưng nhìn dữ liệu khác hẳn
> đóng góp nhiều hơn một nhánh mạnh hơn nhưng nhìn giống hệt.

### 4.5 Giới hạn: ensemble giảm **phương sai**, không giảm **thiên lệch**

Nếu mọi nhánh cùng sai một cách có hệ thống (ví dụ đều thua ở `category_03` — bạn ghi là khó nhất),
ensemble **không chữa được**. Dấu hiệu: sai số của các nhánh tương quan mạnh **trên đúng những mẫu khó**.

Chẩn đoán: dựng ma trận `sai số × nhánh`, tìm các mẫu **mọi nhánh đều sai**.
Nếu chiếm > 10% → vấn đề là thiên lệch, phải đổi **cách tiếp cận**, không phải thêm nhánh.

### 4.6 ⚠️ Ràng buộc 2026 làm thay đổi tất cả: `main.py ≤ 20 phút`

Lối đánh "ensemble 4–6 nhánh nặng" của bạn **không lọt qua trần 20 phút**. Bốn cách giữ được đa dạng
mà vẫn nằm trong ngân sách — theo thứ tự nên thử:

**(1) 🔴 Ensemble chia sẻ encoder — mẹo mạnh nhất cho tình huống của bạn.**
```
ảnh/văn bản ──► encoder ĐÓNG BĂNG (chạy MỘT lần, cache đặc trưng)
                      ├──► head A (linear)
                      ├──► head B (kNN / Mahalanobis)
                      ├──► head C (GBDT trên chính vector đó)
                      └──► head D (MLP nhỏ)
```
Chi phí inference ≈ **một** encoder. Các head rẻ đến mức miễn phí, và vì chúng **khác họ thuật toán**
nên `ρ` giữa chúng thấp hơn bạn tưởng nhiều.

**(2) TTA thay cho nhiều model.** Lật ngang + đa tỉ lệ trên **một** model = 3–4 "nhánh" với chi phí
tuyến tính nhỏ, `ρ` trung bình.

**(3) Cascade / thoát sớm.** Model nhẹ xử lý phần lớn mẫu dễ; chỉ mẫu gần ngưỡng mới gọi model nặng.
Nếu 80% mẫu dễ, chi phí trung bình giảm ~3 lần.

**(4) Chưng cất (distillation)** ensemble về một model. Đúng về lý thuyết nhưng **tốn thời gian train**
— thường không lọt vào 6 tiếng. Để dành cho chung kết.

### 4.7 Cách ghép: leo đồi trên OOF

Trung bình đều là mặc định an toàn. Nhưng tốt hơn là **leo đồi tham lam có hoàn lại**:

```python
def hill_climb(oof_preds, y, metric_fn, n_iter=50):
    """oof_preds: list các mảng xác suất OOF. Cho phép chọn lại cùng model
       → tự nhiên sinh ra trọng số nguyên."""
    chosen, cur = [], None
    for _ in range(n_iter):
        best_j, best_s = None, -np.inf
        for j, p in enumerate(oof_preds):
            cand = p if cur is None else (cur*len(chosen) + p)/(len(chosen)+1)
            s = metric_fn(y, cand)
            if s > best_s: best_s, best_j, best_cand = s, j, cand
        if chosen and best_s <= metric_fn(y, cur): break   # dừng khi hết cải thiện
        chosen.append(best_j); cur = best_cand
    return chosen, cur
```

Ổn định hơn Nelder–Mead khi số mẫu OOF nhỏ, và **tự động loại nhánh vô dụng**.
Với seq2seq, dùng **ensemble decoding** (trung bình log-prob **tại mỗi bước giải mã**),
mạnh hơn nhiều so với ghép đầu ra cuối.

---

## NGUYÊN LÝ 5 — LƯỢT NỘP LÀ THIẾT BỊ ĐO, KHÔNG CHỈ LÀ BÀI LÀM
### *Điều thứ năm bạn đã làm nhưng chưa đặt tên.*

### 5.1 Bạn đã làm gì

> Lượt 1 · *"toàn nhãn 0"* · **50,0** ·
> *"Mốc neo: TPR=0, TNR=1 ⇒ BA=0,5 chính xác. Xác nhận format + thang điểm"*

Bạn **dự đoán trước** điểm bằng lý thuyết, rồi dùng lượt nộp để **kiểm chứng ba thứ cùng lúc**:
hợp đồng định dạng đúng · công thức chấm đúng như hiểu · thang điểm hiển thị đúng như nghĩ.

Đó là dùng lượt nộp như **thiết bị đo**, không phải như một lần thử vận may. Rất ít đội làm.

### 5.2 Tổng quát hoá: lượt nộp thăm dò (probe)

Với một số metric, có thể **giải ngược** ra thông tin về tập test từ điểm số:

| Nộp gì | Metric | Suy ra được |
|---|---|---|
| Toàn nhãn `c` | Accuracy | Tỉ lệ lớp `c` trong test |
| Toàn nhãn `c` | Macro-F1 | Ràng buộc về tỉ lệ các lớp |
| Toàn nhãn 0 | Balanced Accuracy | **Luôn 0,5** → chỉ kiểm định dạng, không cho thông tin |
| Bản dịch rỗng | BLEU | Độ dài tham chiếu qua brevity penalty |

*Vì sao BA cho 0,5 bất kể tỉ lệ anomaly: `BA = (TPR + TNR)/2` không chứa prevalence — chính là
**Bài học #3** trong `cv/RESULTS.md` của bạn. Bạn đã hiểu điều này rồi.*

### 5.3 ⚠️ Ranh giới tuân thủ — đọc kỹ trước khi dùng

Quy chế: *"Không được phép sử dụng tập test dưới bất kỳ hình thức nào để huấn luyện mô hình."*

- ✅ **Được**: dùng điểm public để chọn siêu tham số / ngưỡng — đó là mục đích của 20 lượt nộp,
  BTC chủ động cho phản hồi.
- ✅ **Được**: nộp baseline tầm thường để kiểm định dạng.
- ⚠️ **Vùng xám**: giải ngược tỉ lệ lớp từ điểm rồi cứng hoá vào ngưỡng. Không phải "huấn luyện"
  theo nghĩa gradient, nhưng **là** dùng thông tin từ test. **Nếu làm, phải ghi rõ trong báo cáo kỹ thuật.**
- ❌ **Cấm**: pseudo-label trên test rồi đưa vào tập train. Đây là vi phạm rõ ràng.
- ❌ **Cấm**: gán nhãn thủ công từng mẫu test.

> **Khuyến nghị:** ở vòng miền, dùng probe **chỉ để kiểm định dạng và kiểm chứng hiểu đúng công thức**.
> Đừng giải ngược tỉ lệ lớp — lợi ích nhỏ, rủi ro bị loại lớn, và **ban giám khảo có chấm định tính**.

### 5.4 ⚠️ Cảnh báo lớn cho vòng miền: mốc neo của bạn sẽ khác

Ở vòng trường, công thức là `SCORE / MAX × 100%` → "toàn nhãn 0" cho **50,0**, một con số đẹp để neo.

Ở SOLOAI/VOAI 2025, công thức là `(S − Min)/(Max − Min) × 100` với `Min` = **model đơn giản của BTC**.
Baseline tầm thường của bạn **kém hơn model đơn giản của BTC** → `Norm_score ≈ 0` hoặc **âm** (bị chặn về 0).

**Hệ quả:** lượt nộp đầu vẫn **bắt buộc** (kiểm định dạng — phát hiện sai format lúc T+5:30 là thảm hoạ),
nhưng **đừng hoảng khi thấy 0 điểm**. Cả đội phải biết trước điều này, nếu không sẽ có người
tưởng pipeline hỏng và đập đi làm lại lúc T+1:00.

---

## ÁP DỤNG: QUY TRÌNH 6 TIẾNG DÙNG CẢ 5 NGUYÊN LÝ

| Thời điểm | Việc | Nguyên lý |
|---|---|---|
| **T+0:00–0:20** | Đọc đề. Trích metric + trọng số. **Hỏi: "tôi ở bậc mấy?"** (§2.2). **Quét 6 loại rò rỉ** (§3.3). Xác định chiều mà private khác train. | 2, 3 |
| **T+0:20–0:50** | Dựng sơ đồ CV **đúng nhóm** trước khi train bất cứ gì. Nộp lượt 1 kiểm định dạng — **biết trước là ~0 điểm**. | 3, 5 |
| **T+0:50–1:10** | Có OOF đầu tiên → **chạy `bootstrap_se` ngay**, viết SE lên giấy dán màn hình. Chạy phép thử khoảng cách CV vs public. | 1, 3 |
| **T+1:10–2:30** | Nếu bậc 1–2: iterate tự tin. Nếu bậc 4–5: **kiểm định proxy bằng ρ trước** (§2.4), hoặc bỏ luôn việc search và chuyển sang nước đi tiên nghiệm. | 2 |
| **T+2:30–3:30** | Mỗi thay đổi: so bằng **bootstrap ghép cặp**, xem `P(A>B)`. `P < 0,95` → **giữ cả hai để ghép**, đừng thay thế. | 1, 4 |
| **T+3:30–4:30** | Ghép mô hình. Hỏi *"cách nhìn dữ liệu nào tôi chưa dùng"* (§4.4). Ưu tiên **ensemble chia sẻ encoder** để lọt trần 20 phút. Leo đồi trên OOF. | 4 |
| **T+4:30–5:00** | 🧊 Đóng băng. Chọn bài cuối **theo OOF**, không theo public — nhớ thiên lệch `SE·√(2 ln k)` (§1.5). Ghi lý do vào báo cáo. | 1 |
| **T+5:00–6:00** | Private: chỉ inference. Lượt 1 = cấu hình **OOF cao nhất**, không phải public cao nhất. | 1 |

---

## CHECKLIST — DÁN LÊN MÀN HÌNH NGÀY THI

**Trước khi train bất cứ gì:**
- [ ] Tôi đang ở **bậc mấy** trên thang độ tin cậy hàm mục tiêu?
- [ ] Dữ liệu có **cấu trúc nhóm** không? (biến thể / cùng video / cùng người / cùng ảnh gốc)
- [ ] Sơ đồ CV của tôi có tách theo **đúng chiều mà private test tách** không?

**Sau OOF đầu tiên:**
- [ ] SE bootstrap = **______** ← viết ra giấy, dán màn hình
- [ ] `CV − public` có > 3×SE không? (nếu có → nghi rò rỉ, dừng lại kiểm tra)

**Trước mỗi lượt nộp:**
- [ ] Offline hơn bản tốt nhất bao nhiêu **SE ghép cặp**? `P(A>B)` = ?
- [ ] Nếu `P < 0,95`: **giữ cả hai để ghép**, không thay thế, không tiêu lượt nộp.

**Khi kẹt (điểm không lên 45 phút):**
- [ ] Proxy của tôi có `ρ > 0,7` không? Chưa kiểm định → **cấm search**.
- [ ] **Cách nhìn dữ liệu nào tôi chưa dùng?** (biểu diễn ≫ kiến trúc ≫ loss ≫ fold ≫ seed)
- [ ] Có phải vấn đề **thiên lệch** không? (mọi nhánh cùng sai một chỗ → đổi cách tiếp cận)

**Chọn bài cuối:**
- [ ] Chọn theo **OOF**, không theo public. Thiên lệch best-of-k = `SE × √(2 ln k)` = **______**
- [ ] Ensemble có chạy lọt **20 phút** trong `main.py` không?

---

## TÓM TẮT MỘT TRANG

| # | Nguyên lý | Câu hỏi lõi | Sai lầm phổ biến bạn đã tránh |
|---|---|---|---|
| 1 | **Độ phân giải thước đo** | *Cải thiện này lớn hơn mấy SE?* | Tin vào chênh lệch 0,001 trên 3.340 mẫu |
| 2 | **Bậc độ tin cậy** | *Tôi được phép search không?* | Random search trên proxy đã sai hướng 4/4 lần |
| 3 | **Rò rỉ & chia fold** | *Dữ liệu có cấu trúc nhóm gì?* | CV cao giả 0,04 = 13 SE |
| 4 | **Đa dạng hoá** | *Cách nhìn nào tôi chưa dùng?* | Thêm model to hơn cùng họ (ρ≈0,95, vô ích) |
| 5 | **Lượt nộp là thiết bị đo** | *Lượt này đo được gì?* | Nộp bừa để "xem sao" |

**Ba nâng cấp cho vòng miền:**
1. **Bootstrap ghép cặp** thay SE biên → phân biệt được cải thiện nhỏ mà SE biên tuyên bố vô nghĩa (§1.4).
2. **Kiểm định proxy bằng Spearman ρ** trước khi tin, thay vì phát hiện nó hỏng sau 4 lần thất bại (§2.4).
3. **Ensemble chia sẻ encoder** thay ensemble nhiều model nặng → giữ đa dạng dưới trần 20 phút (§4.6).
