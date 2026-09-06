"""Đáp án tham khảo — BT 04: Chia fold theo nhóm + phát hiện rò rỉ."""
import numpy as np
from collections import defaultdict


def has_group_leak(train_idx, val_idx, groups):
    groups = np.asarray(groups)
    return len(set(groups[train_idx]) & set(groups[val_idx])) > 0


def stratified_group_kfold(y, groups, n_splits=5, seed=0):
    """Mỗi nhóm nằm trọn trong ĐÚNG MỘT fold, đồng thời cân bằng phân phối lớp.
    Thuật toán tham lam: xếp nhóm theo độ 'lệch' giảm dần, gán vào fold làm
    phân phối lớp toàn cục lệch ít nhất (giống sklearn.StratifiedGroupKFold)."""
    y = np.asarray(y); groups = np.asarray(groups)
    n_classes = int(y.max()) + 1

    # đếm nhãn theo từng nhóm
    cnt = defaultdict(lambda: np.zeros(n_classes, dtype=np.int64))
    for lab, g in zip(y, groups):
        cnt[g][lab] += 1

    y_cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    y_cnt[y_cnt == 0] = 1.0                      # tranh chia 0

    rng = np.random.default_rng(seed)
    g_keys = list(cnt.keys())
    rng.shuffle(g_keys)
    # nhom "lech" nhat xu ly truoc -> con lai de can bang (giong sklearn)
    g_keys.sort(key=lambda g: -np.std(cnt[g]))

    fold_cnt = np.zeros((n_splits, n_classes), dtype=np.float64)
    g_to_fold = {}
    for g in g_keys:
        best_f, best_cost = None, None
        for f in range(n_splits):
            fold_cnt[f] += cnt[g]
            # chuan hoa theo TONG so mau moi lop -> moi fold nen giu 1/n_splits moi lop.
            # Chuan hoa nay dong thoi can bang KICH THUOC fold.
            cost = np.std(fold_cnt / y_cnt, axis=0).mean()
            fold_cnt[f] -= cnt[g]
            if best_cost is None or cost < best_cost:
                best_cost, best_f = cost, f
        fold_cnt[best_f] += cnt[g]
        g_to_fold[g] = best_f

    fold_of_row = np.array([g_to_fold[g] for g in groups])
    splits = []
    for f in range(n_splits):
        val = np.where(fold_of_row == f)[0]
        tr = np.where(fold_of_row != f)[0]
        splits.append((tr, val))
    return splits


def recover_groups_by_text(texts, threshold=0.7, ngram_range=(3, 5)):
    """Phục hồi 'comment gốc' từ các biến thể nhiễu — đúng kỹ thuật build_groups.py.
    TF-IDF char n-gram -> cosine -> nối thành phần liên thông khi sim >= threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range).fit_transform(texts)
    S = (X @ X.T).toarray()          # đã chuẩn hoá L2 -> tích vô hướng = cosine
    n = len(texts)

    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                union(i, j)

    roots = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


def adversarial_validation_auc(X_train, X_test, seed=0):
    """AUC ~0.5 -> cùng phân phối. AUC -> 1.0 -> lệch phân phối / rò rỉ."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = np.vstack([np.asarray(X_train), np.asarray(X_test)])
    y = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
