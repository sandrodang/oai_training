# OLPAI'26 — Ôn luyện Vòng loại Khu vực miền Bắc

Tài liệu ôn tập của đội cho **Olympic AI Sinh viên 2026**, vòng loại khu vực
**31/10/2026** tại Học viện Công nghệ Bưu chính Viễn thông (Hà Nội).

---

## 📁 Nội dung

| Tệp / thư mục | Mô tả |
|---|---|
| **[KE_HOACH_OLPAI26.md](KE_HOACH_OLPAI26.md)** | Kế hoạch 8 tuần đầy đủ: luật thi, chẩn đoán năng lực, chương trình lý thuyết 6 tầng, lộ trình từng tuần, phân vai ngày thi, sổ rủi ro |
| **[PHUONG_PHAP_LUAN.md](PHUONG_PHAP_LUAN.md)** | Sổ tay phương pháp luận thi đấu — 5 nguyên lý quyết định "cải thiện này có thật không" |
| **[tuan01/](tuan01/)** | Gói học Tuần 1: tài liệu đọc, 6 bài tập có bộ chấm tự động, notebook đo môi trường |
| **[tuan02/](tuan02/)** | Gói học Tuần 2: Transformer & seq2seq tự viết, 5 bài tập + 26 test, corpus song ngữ Kơtu→Việt |
| **[de_tham_khao/](de_tham_khao/)** | 5 đề thi PDF: SOLOAI 2025, VOAI 2025 CK, 2 đề chính thức vòng trường 2026, đề thi thử 2026 |
| **[task1_nlp_fpt26/](task1_nlp_fpt26/)** | 🗄 Mã nguồn bài **Tác vụ 1 — NLP** vòng trường 2026 (R-ViHSD: hate speech + noise type). Private 0.720 |
| **[task2_cv_fpt26/](task2_cv_fpt26/)** | 🗄 Mã nguồn bài **Tác vụ 2 — CV** vòng trường 2026 (image anomaly detection, 6 category). Public 80.4 |

---

## 🚀 Bắt đầu

```bash
git clone https://github.com/sandrodang/oai_training.git
cd oai_training

# Yêu cầu: python3 với torch, scikit-learn, pytest
python3 -c "import torch, sklearn, pytest; print('OK')"

# Chạy bộ chấm — sẽ ĐỎ hết, đó là đúng: bạn phải tự điền code
python3 -m pytest tuan01/bai_tap/test_all.py -q
python3 -m pytest tuan02/bai_tap/test_all.py -q -m "not slow"
```

**Đọc theo thứ tự:**
1. `KE_HOACH_OLPAI26.md` — Phần 0 → Phần 1 (luật thi, đọc kỹ §1.3 công thức điểm và §1.4 ràng buộc 20 phút)
2. `PHUONG_PHAP_LUAN.md` — 5 nguyên lý
3. `tuan01/README.md` — lịch 8 ngày, rồi bắt đầu N1

---

## 🗄 Hai thư mục bài cũ — dùng để làm gì

`task1_nlp_fpt26/` và `task2_cv_fpt26/` là **mã nguồn bài làm vòng trường 2026** của đội, tách ra từ
thư mục làm việc gốc (77GB). **Chỉ gồm mã nguồn, log, notebook, tài liệu và file nộp** —
không có trọng số, checkpoint hay dữ liệu BTC.

Giá trị của chúng **không phải để tái sử dụng** — vòng miền đổi luật khá nhiều
(xem `KE_HOACH_OLPAI26.md` Phần 0). Giá trị là để **đọc lại phương pháp**:

- `task1_nlp_fpt26/REPRODUCE.md` — mẫu hồ sơ tái lập + tuyên bố tuân thủ. **Dùng lại cấu trúc này
  cho báo cáo kỹ thuật ngày thi** (quy chế bắt buộc nộp).
- `task2_cv_fpt26/RESULTS.md` — nhật ký từng lượt nộp kèm kết luận. Bảng *"đã thử và loại bỏ"*
  chính là thứ ban giám khảo muốn thấy ở phần chấm định tính.
- `task1_nlp_fpt26/src/build_groups.py` — kỹ thuật phục hồi nhóm chống rò rỉ (Tuần 1 BT 04 cài lại).
- `task1_nlp_fpt26/src/tune_prior.py`, `task2_cv_fpt26/src/calibrate.py` — hiệu chỉnh ngưỡng.
  ⚠️ Đọc kèm cảnh báo ở `tuan01/bai_tap/05_threshold.py`: tune ngưỡng **làm tệ đi** task hate.

---

## ⚠️ Quy tắc kho này

- **KHÔNG commit dữ liệu thi của BTC**, trọng số mô hình, hay checkpoint.
  `.gitignore` đã chặn sẵn `cv/`, `work/`, `ThiChinhThucData/`, `*.pt`, `*.zip`…
  Quy chế ghi rõ *"chỉ được sử dụng dữ liệu do Ban giám khảo cung cấp"*; phát tán lại
  dữ liệu thi là rủi ro kỷ luật.
- **`tuan01/dap_an/` là đáp án.** Chỉ mở sau khi đã tự làm ≥ 45 phút mỗi bài.
- Ghi kết quả đo môi trường của **máy bạn** vào `tuan01/env_report.md` (mỗi người một bản,
  vì GPU được cấp khác nhau).

---

## 📊 Trạng thái

| Tuần | Nội dung | Trạng thái |
|---|---|---|
| 1 · 05–12/09 | Nền tảng: đánh giá, PyTorch, đo môi trường | ✅ tài liệu sẵn sàng |
| 2 · 13–19/09 | Transformer & seq2seq tự viết | ✅ tài liệu sẵn sàng |
| 3 · 20–26/09 | Dịch máy ít tài nguyên 🔴 | ⏳ |
| 4 · 27/09–03/10 | CV nền + video/hành động | ⏳ |
| 5 · 04–10/10 | Tự giám sát, jigsaw, tối ưu suy luận | ⏳ |
| 6 · 11–17/10 | Ráp đội + Tổng duyệt 1 | ⏳ |
| 7 · 18–24/10 | Vá lỗ hổng + Tổng duyệt 2 | ⏳ |
| 8 · 25–30/10 | Nghi thức, giảm tải | ⏳ |

**Câu hỏi đang chờ BTC trả lời:** xem `KE_HOACH_OLPAI26.md` Phần 11 (11 câu).
Một số câu ảnh hưởng trực tiếp đến kế hoạch — nhất là §11.1 (công thức điểm),
§11.3 (pretrained), §11.4 (được mang code vào phòng thi không).
