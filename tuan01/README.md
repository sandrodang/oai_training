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

## 📅 LỊCH 8 NGÀY · **4h/ngày = 28h** (ngân sách bản 3, xem kế hoạch §4B.1)

| Ngày | | Nội dung | Giờ | Sản phẩm |
|---|---|---|---|---|
| **N1** | T7 05/09 | 🔴 **ĐO MÔI TRƯỜNG** Colab + Kaggle (2h) · Đọc [§1 đánh giá phân loại](TAI_LIEU.md) (2h) | 4h | `env_report.md` |
| **N2** | CN 06/09 | **BT 01** metrics (2h) · Đọc §2 BLEU (1,5h) · khởi động BT 02 (30') | 4h | `01_metrics.py` xanh |
| **N3** | T2 07/09 | **BT 02** BLEU 🔴 (2,5h) · Đọc §3 sai số + bias–variance (1,5h) | 4h | `02_bleu.py` xanh |
| **N4** | T3 08/09 | **BT 03** bootstrap (1,5h) · Đọc §4 rò rỉ (1,5h) · khởi động BT 04 (1h) | 4h | `03_bootstrap.py` xanh |
| **N5** | T4 09/09 | **BT 04** chia fold (2h) · **BT 05** ngưỡng (2h) | 4h | `04`, `05` xanh |
| **N6** | T5 10/09 | Đọc §5 PyTorch nền (1,7h) · **BT 06** `train_loop.py` (2,3h) | 4h | vòng lặp chạy được |
| **N7** | T6 11/09 | 🔴🔴 Đọc [§6 vòng lặp cải tiến](TAI_LIEU.md) (3h) · **BT 07** `error_analysis` (1h) | 4h | `07` xanh |
| **N8** | T7 12/09 | **BT 08** `bayes_ceiling` (1,5h) · 🔴 đọc `task1_nlp_fpt26/` như baseline BTC (45') · nghiệm thu + 12 câu (1,75h) | 4h | `NGHIEM_THU.md` |

**Tổng 32h trên 8 ngày** — 28h nội dung + 4h đệm. Nếu tuần bị bóp, xem **thứ tự hy sinh** ở cuối.

### ⚠️ Vì sao ĐO MÔI TRƯỜNG ở N1 (bản đầu để ở N7 — sai)

1. **Nó chứa việc chờ.** Xác minh SĐT Kaggle, tải dữ liệu, cài package — tắc thì phải biết
   **ngày 1**, không phải ngày 7.
2. **Nó định nghĩa lại mọi thứ phía sau.** Nếu T4 mất 10 phút/epoch thì chiến lược
   "8 epoch × 5 fold × 3 model" là **25 giờ** — bất khả thi. Phải biết **trước khi**
   học các kỹ thuật giả định có nhiều compute.
3. **Nó nhẹ về trí óc**, hợp ngày đầu, chạy song song được với việc đọc.

### ⚠️ Vì sao §6 và BT 07–08 ở CUỐI tuần chứ không đầu

§1–§4 dạy **ĐO** xem một cải thiện có thật không. §6 dạy **SINH RA giả thuyết nên cải thiện gì**.
Phải biết đo trước, vì mọi giả thuyết §6 sinh ra đều phải qua cổng 2×SE của §3.
Nhưng **đừng để nó rớt khỏi tuần** — đây là nửa vòng lặp mà bản kế hoạch đầu thiếu hoàn toàn.

### 💰 Ngân sách thật (không tô hồng)

| Khoản | Giờ |
|---|---|
| 📖 Đọc §1–§6 | **11,3h** |
| Đo môi trường | 2,0h |
| BT 01 metrics | 1,0h |
| BT 02 BLEU 🔴 *(thực tế 1,5–2h lần đầu — đừng tin con số 75')* | 1,75h |
| BT 03 bootstrap | 0,85h |
| BT 04 chia fold *(⭐ tự cài splitter +45' là tuỳ chọn)* | 1,0h |
| BT 05 ngưỡng | 0,85h |
| BT 06 train loop *(lần đầu, được tra tài liệu)* | 1,75h |
| BT 07 error_analysis 🔴 | 1,5h |
| BT 08 bayes_ceiling 🔴 | 1,5h |
| Đọc baseline BTC + nghiệm thu | 2,5h |
| 🫙 Đệm | 4,0h |
| **TỔNG** | **~30h / 32h** |

**Thứ tự hy sinh nếu vỡ:** đệm → ⭐ tự cài splitter → §5 (100'→60') → BT 05.
**KHÔNG BAO GIỜ CẮT:** BT 02 · N1 đo môi trường · §6 + BT 07 + BT 08.

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
│   ├── 07_error_analysis.py  🔴 TẦNG 0 §c — nhầm đối xứng (trần) vs một chiều (lệch prior)
│   ├── 08_bayes_ceiling.py   🔴 TẦNG 0 §c — đo TRẦN trước khi tối ưu
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

- [ ] `pytest bai_tap/test_all.py` **41 xanh, 1 skip**
- [ ] `train_loop.py` chạy được (bài gõ-từ-trí-nhớ đã **hoãn** — xem mục ✂️ ở trên)
- [ ] Có `env_report.md` với **≥ 8 con số đo thật** trên Colab và Kaggle
- [ ] Trả lời được 8 câu trong `TAI_LIEU.md § TỰ KIỂM TRA`
- [ ] Tính tay BLEU-4 một cặp câu, khớp với code của mình
