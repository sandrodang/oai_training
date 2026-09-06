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

---

## 🚀 Bắt đầu

```bash
git clone https://github.com/sandrodang/oai_training.git
cd oai_training

# Yêu cầu: python3 với torch, scikit-learn, pytest
python3 -c "import torch, sklearn, pytest; print('OK')"

# Chạy bộ chấm Tuần 1 — sẽ ĐỎ hết, đó là đúng: bạn phải tự điền code
python3 -m pytest tuan01/bai_tap/test_all.py -q
```

**Đọc theo thứ tự:**
1. `KE_HOACH_OLPAI26.md` — Phần 0 → Phần 1 (luật thi, đọc kỹ §1.3 công thức điểm và §1.4 ràng buộc 20 phút)
2. `PHUONG_PHAP_LUAN.md` — 5 nguyên lý
3. `tuan01/README.md` — lịch 8 ngày, rồi bắt đầu N1

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
| 2 · 13–19/09 | Transformer & seq2seq tự viết | ⏳ chưa dựng |
| 3 · 20–26/09 | Dịch máy ít tài nguyên 🔴 | ⏳ |
| 4 · 27/09–03/10 | CV nền + video/hành động | ⏳ |
| 5 · 04–10/10 | Tự giám sát, jigsaw, tối ưu suy luận | ⏳ |
| 6 · 11–17/10 | Ráp đội + Tổng duyệt 1 | ⏳ |
| 7 · 18–24/10 | Vá lỗ hổng + Tổng duyệt 2 | ⏳ |
| 8 · 25–30/10 | Nghi thức, giảm tải | ⏳ |

**Câu hỏi đang chờ BTC trả lời:** xem `KE_HOACH_OLPAI26.md` Phần 11 (11 câu).
Một số câu ảnh hưởng trực tiếp đến kế hoạch — nhất là §11.1 (công thức điểm),
§11.3 (pretrained), §11.4 (được mang code vào phòng thi không).
