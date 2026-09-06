"""BT 04 — Chia fold theo nhóm & phát hiện rò rỉ.  ⏱ thực tế ~60' (core) · Ngày N5   🔴

Ở vòng trường, chia fold sai làm CV cao giả 0.04 = 13 lần SE.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k CVSplit
"""
import numpy as np
from collections import defaultdict


def has_group_leak(train_idx, val_idx, groups):
    """True nếu có nhóm nào xuất hiện ở CẢ train lẫn val. Dùng set intersection."""
    raise NotImplementedError


def stratified_group_kfold(y, groups, n_splits=5, seed=0):
    """Trả list [(train_idx, val_idx)] × n_splits.

    ⭐ HÀM NÀY ĐƯỢC PHÉP GỌI sklearn.model_selection.StratifiedGroupKFold.
       Trong phòng thi bạn SẼ gọi sklearn — thứ sklearn không cho bạn là
       (a) biết rằng mình CẦN nó, và (b) phục hồi được `groups`.
       Đó là hai hàm còn lại trong file này, và chúng mới là phần bắt buộc.

    Tự cài từ đầu là BÀI NÂNG CAO (⭐, +45'). Chỉ làm nếu còn dư giờ.
    Thuật toán nếu bạn muốn thử:

    Hai ràng buộc phải thoả ĐỒNG THỜI:
      (a) mỗi nhóm nằm TRỌN trong đúng một fold  -> chống rò rỉ
      (b) phân phối lớp mỗi fold gần phân phối toàn cục -> stratified

    Thuật toán tham lam (giống sklearn.StratifiedGroupKFold):
      1. đếm nhãn theo từng nhóm: cnt[g] = vector độ dài n_classes
      2. xếp nhóm theo -std(cnt[g])  (nhóm 'lệch' nhất xử lý trước)
      3. với mỗi nhóm, thử đặt vào từng fold, chọn fold làm chi phí nhỏ nhất:
             cost = mean_c( std_f( fold_cnt[f, c] / tổng_số_mẫu_lớp_c ) )
         Chuẩn hoá theo TỔNG số mẫu mỗi lớp khiến mỗi fold hướng tới việc giữ
         1/n_splits của MỖI lớp -> tự động cân bằng cả kích thước fold.
    """
    raise NotImplementedError


def recover_groups_by_text(texts, threshold=0.5, ngram_range=(3, 5)):
    """Phục hồi 'văn bản gốc' từ các biến thể nhiễu -> trả mảng id nhóm.

    Đây CHÍNH LÀ kỹ thuật src/build_groups.py bạn đã dùng ở vòng trường.
    Được phép dùng sklearn ở riêng hàm này.

      1. TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range)
      2. TF-IDF đã chuẩn hoá L2 -> S = X @ X.T chính là ma trận cosine
      3. nối union-find mọi cặp có S[i,j] >= threshold
      4. đánh số lại nhãn cụm từ 0

    Tổng quát hoá cho CV: thay TF-IDF bằng perceptual hash hoặc kNN trên embedding.
    """
    raise NotImplementedError


def adversarial_validation_auc(X_train, X_test, seed=0):
    """Train classifier phân biệt train vs test, trả AUC cross-validated.

        AUC ≈ 0.5  -> cùng phân phối, yên tâm
        AUC -> 1.0 -> lệch phân phối hoặc rò rỉ; đặc trưng quan trọng nhất
                      của classifier đó CHỈ THẲNG vào chỗ rò rỉ

    Được dùng sklearn: LogisticRegression + cross_val_predict(method='predict_proba')
    + roc_auc_score.
    """
    raise NotImplementedError
