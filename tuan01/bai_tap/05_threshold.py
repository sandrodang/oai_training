"""BT 05 — Tối ưu ngưỡng: KHI NÀO ĐÁNG LÀM, KHI NÀO PHÍ THỜI GIAN.  ⏱ ~50' · Ngày N7

⚠️ ĐỌC KỸ PHẦN NÀY TRƯỚC KHI CODE — nó quan trọng hơn bản thân đoạn code.

F1 và Balanced Accuracy không tối ưu tại ngưỡng 0.5, nên tune ngưỡng NGHE có vẻ
là món ăn điểm miễn phí. Số đo thật từ bài R-ViHSD vòng trường, giao thức
half-fit / half-eval, chạy lại trên **4 mô hình** (nguồn: work/src/tune_bias.py):

    mô hình                       hate held-out    noise held-out
    ─────────────────────────────────────────────────────────────
    visobert_ep    (fold RÒ RỈ)     −0.0021          +0.0211
    visobert_clean (fold sạch)      +0.0012          +0.0244
    visobert_fgm   (fold sạch)      −0.0008          +0.0171
    visobert_cons  (fold sạch)      −0.0014          +0.0095

⚠️ Dòng đầu đến từ lần chạy có RÒ RỈ FOLD (xem BT 04) nên không dùng một mình
   được. Ba dòng dưới mới là số sạch.

BÀI HỌC ĐÚNG — đọc theo cột, không đọc theo ô:

  1. HATE: bốn lần đo ra −0.0021 / +0.0012 / −0.0008 / −0.0014.
     Đổi dấu, tất cả đều bé. Kết luận KHÔNG phải "tune làm hại" mà là
     **"tune KHÔNG LÀM GÌ CẢ"**. Một mẫu đơn lẻ không đủ để kết luận dấu.

  2. NOISE: bốn lần đều DƯƠNG (+0.0095 .. +0.0244), kể cả trên fold sạch.
     Bất đối xứng này bền vững -> nó có CƠ CHẾ, không phải may rủi.

  3. Cơ chế: noise có lớp TEENCODE với precision 0.21 / recall 0.41 — lệch nặng,
     bị dự đoán thừa gấp đôi. Dịch ngưỡng ở đó ăn tiền thật.
     Hate đã dùng class_weight='balanced' nên các lớp đã nằm gần đỉnh F1 rồi
     (OFFENSIVE: P=0.6075 / R=0.5642, gần bằng nhau) -> không còn gì để lấy.

  4. => QUY TẮC DÙNG ĐƯỢC TRONG PHÒNG THI:
     **Xem |precision − recall| từng lớp TRƯỚC. Lệch nhiều thì tune. Cân rồi thì bỏ qua.**
     Hàm worth_tuning() dưới đây biến quy tắc này thành code — chạy nó trước,
     đừng tune mù rồi mới xem.

  5. Và luôn: kiểm bằng HELD-OUT (tuning_gain_holdout). Con số đo trên chính
     dữ liệu đã fit BAO GIỜ CŨNG lạc quan — 8/8 lần đo thật, không ngoại lệ.

  6. Hỏi thêm: mình đang tối ưu theo PRIOR NÀO? Ở vòng trường, bias tune theo
     prior TRAIN đẩy ORIGINAL lên 34% (thật ~21%) và làm mất điểm. Tune theo
     prior TEST ước lượng mới đúng.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Threshold
"""
import numpy as np


def best_threshold_binary(y_true, scores, metric_fn, n_grid=None):
    """Quét ngưỡng, trả (ngưỡng_tốt_nhất, điểm_tốt_nhất).

    Ứng viên = các giá trị score DUY NHẤT (np.unique) — quét lưới đều là lãng phí.
    n_grid không None và có quá nhiều ứng viên -> rút gọn bằng np.quantile.
    Nhớ thêm ±inf để cho phép dự đoán toàn 0 / toàn 1.

    ⚠️ Điểm trả về là điểm TRÊN CHÍNH DỮ LIỆU ĐÃ DÙNG ĐỂ FIT -> lạc quan.
       Muốn biết nó có thật không, dùng tuning_gain_holdout() bên dưới.
    """
    raise NotImplementedError


def tuning_gain_holdout(y_true, scores, metric_fn, default_thr=0.5,
                        n_repeat=20, seed=0):
    """🔴 HÀM QUAN TRỌNG NHẤT FILE NÀY — giao thức half-fit / half-eval.

    Trả (gain_fit_TB, gain_eval_TB).

    Mỗi vòng lặp trong n_repeat:
        1. hoán vị ngẫu nhiên chỉ số, chia đôi thành FIT và EVAL
        2. tìm ngưỡng tốt nhất TRÊN NỬA FIT  (dùng best_threshold_binary)
        3. gain_fit  = điểm(FIT,  ngưỡng đó) − điểm(FIT,  default_thr)
           gain_eval = điểm(EVAL, ngưỡng đó) − điểm(EVAL, default_thr)
    Trả trung bình của hai đại lượng trên qua n_repeat lần.

    CÁCH ĐỌC KẾT QUẢ:
        gain_eval ≈ gain_fit  -> tune thật sự tổng quát hoá, dùng được
        gain_eval << gain_fit -> phần lớn là fit nhiễu, ĐỪNG dùng
        gain_eval < 0         -> tune LÀM HẠI, giữ ngưỡng mặc định
        |gain_eval| rất bé    -> tune vô hại nhưng VÔ ÍCH, đừng tốn thời gian

    ĐO THẬT trên dữ liệu tổng hợp (8 seed), để bạn biết kỳ vọng thế nào:

        kịch bản                    gain_fit          gain_eval
        ─────────────────────────────────────────────────────────────
        nhiễu thuần, n=1000         +0.0039           −0.0089   ← LÀM HẠI
        tín hiệu thật, n=1500       +0.0150..+0.0311  −0.0044..+0.0105
        tín hiệu thật, n=3000       +0.0146..+0.0224  −0.0015..+0.0179

    Hai điều rút ra, và điều thứ hai mới là quan trọng:
      1. Ngay cả khi CÓ tín hiệu thật, gain_eval vẫn có thể ÂM nếu thiếu mẫu.
         Với 480 ảnh (bài CV vòng trường) hay 3.340 mẫu public, bạn thường
         KHÔNG đủ mẫu để tune ngưỡng một cách đáng tin.
      2. Điều DUY NHẤT luôn đúng qua mọi seed: gain_eval < gain_fit.
         Nghĩa là con số bạn thấy khi tune luôn LẠC QUAN. Không có ngoại lệ.

    Trong phòng thi, chạy phép này TRƯỚC KHI tin bất kỳ mức tăng nào từ tuning.
    """
    raise NotImplementedError


def best_class_bias(y_true, proba, metric_fn, n_iter=30, lo=-3.0, hi=3.0, n_grid=25, seed=0):
    """Đa lớp: tìm vector bias cộng vào LOG-xác suất, dự đoán = argmax(log p + b).
    Trả (bias, điểm_tốt_nhất).

    Tìm kiếm theo TOẠ ĐỘ: lặp qua từng lớp, quét lưới cho lớp đó, giữ giá trị tốt
    nhất, sang lớp tiếp theo. Dừng sớm khi một vòng không cải thiện.

    ⚠️ Cùng cảnh báo: đây là hiệu chỉnh PRIOR. Nó chỉ đúng nếu prior của tập bạn
    fit trùng prior của tập bạn sẽ bị chấm. Ở vòng trường bạn dùng bias prior
    0.25 rồi 0.21 — và cả hai cho private 0.718, tức khác biệt nằm trong nhiễu.
    Nhớ np.clip proba trước khi lấy log.
    """
    raise NotImplementedError


def worth_tuning(y_true, y_pred, n_cls):
    """🔴 CHẠY HÀM NÀY TRƯỚC KHI TUNE. Trả mảng |precision − recall| mỗi lớp.

    Ngưỡng chỉ đáng dịch khi precision và recall của một lớp LỆCH NHAU.
    Lệch lớn  -> mô hình đang dự đoán lớp đó quá nhiều (P thấp, R cao) hoặc
                 quá ít (P cao, R thấp) -> dịch ngưỡng lấy lại được điểm.
    Lệch ~0   -> lớp đó đã ở gần điểm tối ưu F1 -> tune không còn gì để lấy.

    Số thật từ vòng trường (dùng để tự kiểm tra trực giác của bạn):
        noise/TEENCODE   P=0.2100  R=0.4112  -> gap 0.2012   -> tune ĐƯỢC +0.02
        hate/OFFENSIVE   P=0.6075  R=0.5642  -> gap 0.0433   -> tune ~0

    Gợi ý: precision_recall_fscore_support(..., average=None) trả về mảng theo lớp;
    nhưng ở đây TỰ TÍNH bằng numpy để hiểu (BT 01 đã có confusion_matrix_).
    """
    raise NotImplementedError
