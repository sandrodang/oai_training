# TUẦN 1 · 05–12/09/2026
## NỀN TẢNG: ĐÁNH GIÁ + PYTORCH + ĐO MÔI TRƯỜNG THẬT

> Thuộc `KE_HOACH_OLPAI26.md` Phần 5. Lý thuyết: **Tầng 0 toàn bộ + khởi động Tầng 1**.

---

## 🎯 Mục tiêu tuần

Cuối tuần bạn phải làm được **3 việc**, không cần tra tài liệu, không cần LLM:

1. **Gõ vòng lặp huấn luyện PyTorch từ số 0 trong < 15 phút**, chạy đúng ngay lần đầu.
2. **Tự cài mọi metric bằng numpy** (macro-F1, Balanced Accuracy, BLEU-4, AP50) và **bootstrap SE**.
3. **Biết chính xác** một GPU Colab/Kaggle làm được gì trong 30 phút — bằng số đo thật, không phỏng đoán.

**Vì sao tuần này quan trọng:** mọi quyết định trong 6 tiếng thi đều dựa trên *"cải thiện này có thật không?"*.
Không có thước đo riêng đáng tin, bạn sẽ đuổi theo nhiễu suốt 6 tiếng — đúng như bạn đã suýt làm ở
`cv/RESULTS.md` (9/10 cấu hình nằm gọn trong sai số).

---

## 📅 LỊCH 8 NGÀY (~16 giờ — xem *Ngân sách thật* bên dưới)

| Ngày | | Nội dung | Giờ | Sản phẩm |
|---|---|---|---|---|
| **N1** | T7 05/09 | 🔴 **ĐO MÔI TRƯỜNG** Colab + Kaggle (`notebooks/env_probe.ipynb`) | 2h | `env_report.md` |
| **N2** | CN 06/09 | Đọc `TAI_LIEU.md` §1 (80') · **BT 01** metrics | 2,5h | `01_metrics.py` xanh |
| **N3** | T2 07/09 | Đọc §2 BLEU (90') · **BT 02** BLEU 🔴 | 3h | `02_bleu.py` xanh |
| **N4** | T3 08/09 | Đọc §3 sai số (60') · **BT 03** bootstrap | 2h | `03_bootstrap.py` xanh |
| **N5** | T4 09/09 | Đọc §4 rò rỉ (90') · **BT 04** chia fold | 2,5h | `04_cv_split.py` xanh |
| **N6** | T5 10/09 | Đọc §5 PyTorch nền (100') | 1,75h | ghi chú + `TU_KIEM_TRA` |
| **N7** | T6 11/09 | **BT 06** `train_loop.py` — được tra tài liệu | 2h | vòng lặp chạy được |
| **N8** | T7 12/09 | **BT 05** ngưỡng · Nghiệm thu · trả lời 8 câu | 2h | `NGHIEM_THU.md` |

### ⚠️ Vì sao ĐO MÔI TRƯỜNG được đưa lên N1 (bản đầu để ở N7 — sai)

1. **Nó chứa việc chờ.** Xác minh SĐT Kaggle, tải dữ liệu, cài package — tắc thì phải biết
   **ngày 1**, không phải ngày 7.
2. **Nó định nghĩa lại mọi thứ phía sau.** Nếu T4 mất 10 phút/epoch thì chiến lược
   "8 epoch × 5 fold × 3 model" là **25 giờ** — bất khả thi. Phải biết điều đó **trước khi**
   học các kỹ thuật giả định có nhiều compute.
3. **Nó nhẹ về trí óc**, hợp ngày đầu, chạy song song được với việc đọc.

### 💰 Ngân sách thật (không tô hồng)

| Khoản | Giờ |
|---|---|
| Đọc tài liệu (§1–§5, đã cắt 3 mục) | 7,0h |
| BT 01 metrics | 1,0h |
| BT 02 BLEU 🔴 *(thực tế 1,5–2h ở lần đầu — đừng tin con số 75')* | 1,75h |
| BT 03 bootstrap | 0,85h |
| BT 04 chia fold *(phần core; ⭐ tự cài splitter +45' là tuỳ chọn)* | 1,0h |
| BT 05 ngưỡng | 0,85h |
| BT 06 train loop *(lần đầu, được tra tài liệu)* | 1,75h |
| Đo môi trường | 2,0h |
| **TỔNG** | **~16,2h** |

**Nếu chỉ có 15h:** bỏ ⭐ (tự cài `stratified_group_kfold` — được phép gọi sklearn)
và rút §4 xuống 60'. Đừng cắt BT 02 hay N1.

### ✂️ Đã cắt khỏi bản đầu (5h) — chỉ cắt phần đọc không chuyển thành làm được

| Cắt | Giờ | Lý do |
|---|---|---|
| Dwork — *reusable holdout* | 30' | Lý thuyết đẹp nhưng không dùng được trong 6 tiếng thi. BT 03 dạy đủ |
| Bài báo AdamW | 20' | Cần **dùng** AdamW, không cần dẫn lại công thức. d2l Ch.12 đủ |
| ESL Ch.7 · 50' → 20' | 30' | Bootstrap học bằng cách **gõ BT 03** nhanh hơn đọc sách |
| AP50 (đọc §1.4 + phần trong BT 01) | 40' | Metric detection — xác suất ra thấp hơn F1/BLEU nhiều → **hoãn sang Tuần 5** |
| **DRILL gõ từ trí nhớ** | **3h** | ⏸ **HOÃN** đến khi BTC trả lời *"có được mang code vào phòng thi không"* (KE_HOACH §11.4) |

## 📂 Cấu trúc thư mục

```
tuan01/
├── README.md            ← bạn đang đọc
├── TAI_LIEU.md          ← ĐỌC GÌ, ở đâu, bao lâu, câu hỏi tự kiểm tra
├── bai_tap/             ← khung code có sẵn docstring, BẠN điền vào
│   ├── 01_metrics.py        macro-F1, Balanced Accuracy, AP50
│   ├── 02_bleu.py           BLEU-4 + brevity penalty + smoothing
│   ├── 03_bootstrap.py      bootstrap SE + bootstrap GHÉP CẶP
│   ├── 04_cv_split.py       StratifiedGroupKFold + phát hiện rò rỉ
│   ├── 05_threshold.py      tối ưu ngưỡng cho macro-F1 / BA
│   ├── 06_train_loop.py     vòng lặp huấn luyện đầy đủ
│   └── test_all.py          bộ chấm tự động (sklearn làm oracle)
├── dap_an/              ← ⛔ CHỈ MỞ SAU KHI ĐÃ TỰ LÀM
├── notebooks/
│   └── env_probe.ipynb  ← chạy trên Colab & Kaggle để đo GPU thật
├── drill/
│   ├── DRILL.md         ← luật chơi bài gõ lại từ trí nhớ
│   └── KET_QUA.md       ← bảng ghi thời gian của bạn
└── NGHIEM_THU.md        ← checklist cuối tuần
```

---

## ▶️ BẮT ĐẦU NGAY (5 phút)

```bash
cd /home/namdp36/oai/tuan01

# 1. Kiểm tra bộ chấm chạy được (sẽ ĐỎ hết — đúng như mong đợi)
python3 -m pytest bai_tap/test_all.py -q

# 2. N1 KHÔNG phải code — mở notebook đo môi trường, tải lên Colab rồi Kaggle
#    (việc chờ: xác minh SĐT Kaggle có thể mất cả ngày, làm sớm)
xdg-open notebooks/env_probe.ipynb
```

Dùng **`python3` hệ thống** (đã có torch 2.6 + sklearn + pytest), không phải `cv/.venv`.

Bộ chấm dùng **scikit-learn làm chuẩn đối chiếu**. Bạn tự cài bằng numpy, test so khớp với sklearn
đến 1e-9. Đây là cách duy nhất để chắc chắn bạn *hiểu* metric chứ không chỉ *gọi được hàm*.

---

## ⚖️ LUẬT CHƠI TUẦN NÀY

1. **Không copy từ sklearn/internet.** Mục tiêu là hiểu, không phải là có code chạy.
   Bạn đã biết gọi `f1_score`; cái bạn cần là biết nó **sai số bao nhiêu trên 480 mẫu**.
2. **Chưa cấm LLM ở tuần 1** (bắt đầu cấm từ tuần 3). Với `06_train_loop.py` nên tự viết
   trước 45 phút rồi mới hỏi — bạn sẽ phải gõ lại nó từ trí nhớ ở tuần 6/8.
3. **Mở `dap_an/` chỉ sau khi đã thử ≥ 45 phút** cho mỗi bài.
4. **Ghi lại thời gian thật** vào `drill/KET_QUA.md`. Con số đó cho bạn biết mình đang ở đâu.

---

## ✅ Nghiệm thu cuối tuần → `NGHIEM_THU.md`

- [ ] `pytest bai_tap/test_all.py` **32 xanh, 1 skip**
- [ ] `train_loop.py` chạy được (bài gõ-từ-trí-nhớ đã **hoãn** — xem mục ✂️ ở trên)
- [ ] Có `env_report.md` với **≥ 8 con số đo thật** trên Colab và Kaggle
- [ ] Trả lời được 8 câu trong `TAI_LIEU.md § TỰ KIỂM TRA`
- [ ] Tính tay BLEU-4 một cặp câu, khớp với code của mình
