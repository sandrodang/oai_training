# ✅ NGHIỆM THU TUẦN 1 — tự chấm T7 12/09

## A · Bài tập code

```bash
cd /home/namdp36/oai/tuan01
python3 -m pytest bai_tap/test_all.py -q
```

- [ ] **49 xanh, 1 skip** (test AP50 bị skip có chủ đích — hoãn sang Tuần 5)
- [ ] 🔴🔴 `00_torch_basics` xanh 8/8 — và nói được **vì sao** mỗi hàm tồn tại
      (mỗi hàm là một chỗ bản sai vẫn chạy và vẫn ra số)
- [ ] 🔴 `07_error_analysis` — chỉ đúng cặp nào là **trần**, cặp nào là **lệch prior**
- [ ] 🔴 `08_bayes_ceiling` — chạy trên `work/data/training_set.csv` ra **TEENCODE 78,5%**,
      năm nhãn còn lại **< 10%** ⇒ chỉ TEENCODE có trần
- [ ] 🔴 Đọc `task1_nlp_fpt26/` **như thể là baseline BTC** trong 20 phút, không chạy code:
      ghi ra kiến trúc · augmentation · số epoch · cách chia val
- [ ] `01_metrics.py` — 5 test (+1 skip)
- [ ] `02_bleu.py` — 7 test
- [ ] `03_bootstrap.py` — 5 test
- [ ] `04_cv_split.py` — 6 test
- [ ] `05_threshold.py` — 4 test 🔴 *(gồm 2 test giao thức half-fit/half-eval)*
- [ ] `06_train_loop.py` — 5 test

## B · Đo môi trường (làm ở **N1**) → `env_report.md`

- [ ] Chạy `notebooks/env_probe.ipynb` **trên cả Colab và Kaggle**
- [ ] `env_report.md` có **≥ 8 con số đo thật**
- [ ] Trả lời được: *"Trong 6 tiếng tôi chạy được tối đa mấy fold × mấy epoch?"*
- [ ] Biết Kaggle còn bao nhiêu **giờ GPU trong tuần**
- [ ] Đã thử **2 tài khoản Colab trên cùng máy** (xác nhận mẹo *2 máy ≠ 2 GPU*)

## C · Lý thuyết — 8 câu trong `TAI_LIEU.md § TỰ KIỂM TRA`

Viết tay, không nhìn tài liệu:

- [ ] 1. macro / micro / weighted khác nhau thế nào? Vì sao OlpAI chọn macro?
- [ ] 2. Vì sao "toàn nhãn 0" cho BA = 0,5 bất kể tỉ lệ lớp?
- [ ] 3. Vì sao lượt đó được **50 điểm** ở vòng trường nhưng **~0** ở công thức 2025?
- [ ] 4. Brevity penalty phạt bản dịch ngắn ra sao?
- [ ] 5. SE = 0,0116 → cải thiện 0,004 có đáng tin không?
- [ ] 6. Thiên lệch best-of-20 = ? (viết công thức)
- [ ] 7. Ba tầng nhóm bài ngôn ngữ ký hiệu? Nên GroupKFold theo tầng nào?
- [ ] 9. **Bổ sung:** tune ngưỡng ở R-ViHSD làm task hate tệ đi 0.0021. Vì sao?
       Phép kiểm nào lẽ ra phải chạy trước khi tin vào mức tăng của task noise?
- [ ] 8. `model.eval()` đổi BatchNorm ra sao? Vì sao LayerNorm không đổi?

## D · Bài tính tay

- [ ] Tính BLEU-4 cho cặp câu dưới bằng **giấy bút**, rồi so với `corpus_bleu` của mình:
  ```
  hyp = the cat sat on the mat today
  ref = the cat sat on the mat yesterday
  ```
  *(đáp số: p1=6/7, p2=5/6, p3=4/5, p4=3/4, BP=1 → BLEU = 100·(3/7)^(1/4) = 80,91)*

---

## 🚦 Kết luận

- **≥ 90% mục xanh** → sang Tuần 2 (Transformer & seq2seq) đúng lịch.
- **60–90%** → sang tuần 2 nhưng **kéo phần thiếu sang làm cùng**.
- **< 60%** → ⚠️ dừng lại. Tuần 1 là nền của cả 7 tuần sau; nợ ở đây sẽ trả đắt ở tuần 3 (dịch máy).

**Tuần 2 bắt đầu bằng:** [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — đọc code từng dòng, 3 giờ.


---

## F · Ghi chú về những gì ĐÃ HOÃN

| Hoãn | Sang đâu | Điều kiện |
|---|---|---|
| `average_precision` / AP50 | **Tuần 5** | khi làm segmentation |
| Tự cài `stratified_group_kfold` ⭐ | tuỳ chọn | chỉ làm nếu dư giờ; phòng thi gọi sklearn |
| **DRILL gõ từ trí nhớ** (3h) | **Tuần 6 & 8** | sau khi BTC trả lời §11.4 |
| Dwork *reusable holdout* · bài báo AdamW | — | bỏ hẳn, không mất gì |
