"""Bộ chấm tự động Tuần 1.

    python3 -m pytest bai_tap/test_all.py -x -q          # chấm bài CỦA BẠN (bai_tap/)
    SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q  # chấm ĐÁP ÁN (để đối chiếu)

Oracle: scikit-learn cho metrics phân loại; giá trị BLEU tính tay (xem docstring từng test).
"""
import os, sys, math, warnings, importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC = Path(__file__).parent.parent / ("dap_an" if os.environ.get("SOLUTION") else "bai_tap")


def load(name):
    p = SRC / f"{name}.py"
    if not p.exists():
        pytest.skip(f"chưa có {p}")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError:
        pytest.skip(f"{name}: chưa cài đặt")
    return mod


# ══════════════════════════════════════════════ BT 01 — METRICS
class TestMetrics:
    def setup_method(self):
        self.m = load("01_metrics")
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 4, 500)
        self.p = np.where(rng.random(500) < 0.65, self.y, rng.integers(0, 4, 500))

    def test_confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        assert np.array_equal(self.m.confusion_matrix_(self.y, self.p, 4),
                              confusion_matrix(self.y, self.p, labels=range(4)))

    def test_precision_recall_f1(self):
        from sklearn.metrics import precision_recall_fscore_support as prf
        ep, er, ef, _ = prf(self.y, self.p, labels=range(4), zero_division=0)
        gp, gr, gf = self.m.precision_recall_f1(self.y, self.p, 4)
        assert np.allclose(gp, ep, atol=1e-9)
        assert np.allclose(gr, er, atol=1e-9)
        assert np.allclose(gf, ef, atol=1e-9)

    def test_macro_f1(self):
        from sklearn.metrics import f1_score
        assert abs(self.m.macro_f1(self.y, self.p, 4)
                   - f1_score(self.y, self.p, average="macro", zero_division=0)) < 1e-9

    def test_balanced_accuracy(self):
        from sklearn.metrics import balanced_accuracy_score
        assert abs(self.m.balanced_accuracy(self.y, self.p, 4)
                   - balanced_accuracy_score(self.y, self.p)) < 1e-9

    def test_ba_all_zeros_is_half(self):
        """BÀI HỌC CỐT LÕI: BA không chứa prevalence.
        Nộp toàn nhãn 0 -> TPR=0, TNR=1 -> BA=0.5, bất kể tỉ lệ lớp."""
        for pos_rate in (0.05, 0.3, 0.5, 0.9):
            n = 2000
            rng = np.random.default_rng(1)
            y = (rng.random(n) < pos_rate).astype(int)
            assert abs(self.m.balanced_accuracy(y, np.zeros(n, int), 2) - 0.5) < 1e-12

    @pytest.mark.skip(reason="AP50 hoãn sang Tuần 5 — bỏ để giữ ngân sách giờ Tuần 1")
    def test_average_precision(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        s = rng.random(300) * 0.5 + y * 0.4
        assert abs(self.m.average_precision(y, s) - average_precision_score(y, s)) < 1e-9


# ══════════════════════════════════════════════ BT 02 — BLEU
class TestBLEU:
    def setup_method(self):
        self.b = load("02_bleu")

    def test_perfect_match_is_100(self):
        s = "the cat sat on the mat"
        assert abs(self.b.corpus_bleu([s], [s]) - 100.0) < 1e-6

    def test_hand_computed_value(self):
        """Tính tay — chuỗi log co gọn (telescoping):
            hyp = the cat sat on the mat today       (7 token)
            ref = the cat sat on the mat yesterday   (7 token)
            p1=6/7  p2=5/6  p3=4/5  p4=3/4   BP=1  (c=r=7)
            BLEU = exp((ln6-ln7 + ln5-ln6 + ln4-ln5 + ln3-ln4)/4)
                 = exp((ln3-ln7)/4) = (3/7)^(1/4) = 0.8091067...
        """
        got = self.b.corpus_bleu(["the cat sat on the mat today"],
                                 ["the cat sat on the mat yesterday"])
        assert abs(got - 100.0 * (3 / 7) ** 0.25) < 1e-6

    def test_brevity_penalty(self):
        """hyp ngắn nhưng mọi n-gram đều khớp -> p1..p4 = 1, chỉ còn BP phạt.
            c=4, r=7 -> BP = exp(1 - 7/4) = exp(-0.75) -> BLEU = 47.2367...
        """
        got = self.b.corpus_bleu(["the cat sat on"], ["the cat sat on the mat today"])
        assert abs(got - 100.0 * math.exp(-0.75)) < 1e-6

    def test_clipping_blocks_repetition(self):
        """Lặp một từ đúng nhiều lần KHÔNG ăn được điểm nhờ clipping."""
        assert self.b.corpus_bleu(["the the the the"], ["the cat sat on"]) < 1.0

    def test_no_crash_on_zero_ngram(self):
        assert self.b.corpus_bleu(["x y z w"], ["a b c d"]) < 1.0

    def test_brevity_penalty_fn(self):
        assert abs(self.b.brevity_penalty(10, 5) - 1.0) < 1e-12
        assert abs(self.b.brevity_penalty(5, 10) - math.exp(1 - 2.0)) < 1e-12

    def test_modified_precision_clipping(self):
        num, den = self.b.modified_precision("the the the the".split(),
                                             "the cat sat on".split(), 1)
        assert (num, den) == (1, 4), "numerator phải bị CẮT TRẦN về 1"


# ══════════════════════════════════════════════ BT 03 — BOOTSTRAP
class TestBootstrap:
    def setup_method(self):
        self.b = load("03_bootstrap")
        self.acc = lambda y, p: float((y == p).mean())

    def test_se_matches_closed_form(self):
        """Với accuracy: SE = sqrt(p(1-p)/n)."""
        rng = np.random.default_rng(0)
        n, p_true = 4000, 0.75
        y = np.ones(n, int)
        pred = (rng.random(n) < p_true).astype(int)
        se = self.b.bootstrap_se(y, pred, self.acc, B=600, seed=0)
        assert abs(se - math.sqrt(p_true * (1 - p_true) / n)) < 0.004

    def test_se_scales_as_inv_sqrt_n(self):
        """Luật 1/sqrt(n) — phép kiểm tra bạn nên chạy MỌI LẦN bootstrap."""
        rng = np.random.default_rng(1)
        ses = []
        for n in (1000, 4000):
            y = np.ones(n, int)
            pred = (rng.random(n) < 0.7).astype(int)
            ses.append(self.b.bootstrap_se(y, pred, self.acc, B=600, seed=0))
        assert 1.7 < ses[0] / ses[1] < 2.3, "gấp 4 lần mẫu phải giảm SE ~2 lần"

    def test_paired_se_smaller_than_marginal(self):
        """Hai model tương quan mạnh -> SE của HIỆU nhỏ hơn nhiều so với SE biên."""
        rng = np.random.default_rng(2)
        n = 3000
        y = rng.integers(0, 2, n)
        base = np.where(rng.random(n) < 0.75, y, 1 - y)
        a = base.copy(); b = base.copy()
        flip = rng.choice(n, 60, replace=False)
        a[flip] = y[flip]                      # A đúng thêm 60 mẫu
        se_a = self.b.bootstrap_se(y, a, self.acc, B=500, seed=0)
        _, se_d, p_better = self.b.paired_bootstrap(y, a, b, self.acc, B=500, seed=0)
        assert se_d < se_a, "bootstrap ghép cặp phải cho SE nhỏ hơn SE biên"
        assert p_better > 0.99, "A hơn B rõ ràng -> P(A>B) phải gần 1"

    def test_selection_bias(self):
        """SE * sqrt(2 ln k). Số của chính bạn: k=20, SE=0.0116 -> ~0.0284."""
        assert abs(self.b.selection_bias(0.0116, 20) - 0.0116 * math.sqrt(2 * math.log(20))) < 1e-12
        assert abs(self.b.selection_bias(0.0116, 20) - 0.0284) < 0.001
        assert self.b.selection_bias(1.0, 1) == 0.0

    def test_se_scaling_check(self):
        """Số thật của bạn: 0.0116 ở n=3340 -> ~0.003 ở n=51663."""
        assert abs(self.b.se_scaling_check(0.0116, 3340, 51663) - 0.00295) < 5e-4


# ══════════════════════════════════════════════ BT 04 — CHIA FOLD
class TestCVSplit:
    def setup_method(self):
        self.c = load("04_cv_split")
        rng = np.random.default_rng(0)
        self.groups = np.repeat(np.arange(120), 3)      # 120 nhóm, 3 dòng/nhóm
        self.y = np.repeat(rng.integers(0, 3, 120), 3)  # nhãn cố định trong nhóm

    def test_no_group_leak(self):
        for tr, va in self.c.stratified_group_kfold(self.y, self.groups, 5, seed=0):
            assert not self.c.has_group_leak(tr, va, self.groups)

    def test_partition_is_exact(self):
        splits = self.c.stratified_group_kfold(self.y, self.groups, 5, seed=0)
        assert len(splits) == 5
        seen = np.concatenate([va for _, va in splits])
        assert np.array_equal(np.sort(seen), np.arange(len(self.y))), "val phải phủ đúng 1 lần"
        for tr, va in splits:
            assert len(set(tr) & set(va)) == 0

    def test_stratification_reasonable(self):
        splits = self.c.stratified_group_kfold(self.y, self.groups, 5, seed=0)
        glob = np.bincount(self.y, minlength=3) / len(self.y)
        for _, va in splits:
            frac = np.bincount(self.y[va], minlength=3) / len(va)
            assert np.abs(frac - glob).max() < 0.15, "phân phối lớp mỗi fold phải gần toàn cục"

    def test_has_group_leak_detects(self):
        g = np.array([0, 0, 1, 1, 2, 2])
        assert self.c.has_group_leak(np.array([0, 2]), np.array([1, 3]), g)
        assert not self.c.has_group_leak(np.array([0, 1]), np.array([2, 3]), g)

    def test_recover_groups(self):
        """Tái hiện bài toán R-ViHSD: mỗi comment gốc có vài biến thể nhiễu."""
        base = ["hom nay troi dep qua di thoi",
                "thang nay noi chuyen rat vo duyen",
                "mai di an lau khong ban oi"]
        texts, truth = [], []
        for i, b in enumerate(base):
            for v in (b, b.upper(), b + " !!!", b.replace("a", "aa", 1)):
                texts.append(v); truth.append(i)
        got = self.c.recover_groups_by_text(texts, threshold=0.5)
        # so bằng ma trận đồng-cụm (bất biến với việc đổi tên nhãn cụm)
        got = np.asarray(got); truth = np.asarray(truth)
        assert np.array_equal(got[:, None] == got[None, :],
                              truth[:, None] == truth[None, :])

    def test_adversarial_validation(self):
        rng = np.random.default_rng(0)
        same_a, same_b = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
        shifted = rng.normal(loc=3.0, size=(300, 5))
        assert abs(self.c.adversarial_validation_auc(same_a, same_b) - 0.5) < 0.12
        assert self.c.adversarial_validation_auc(same_a, shifted) > 0.95


# ══════════════════════════════════════════════ BT 05 — NGƯỠNG
class TestThreshold:
    def setup_method(self):
        self.t = load("05_threshold")
        from sklearn.metrics import f1_score
        self.f1 = lambda y, p: f1_score(y, p, average="macro", zero_division=0)

    def test_beats_default_half(self):
        """Lớp lệch -> ngưỡng tối ưu KHÔNG phải 0.5."""
        rng = np.random.default_rng(0)
        n = 4000
        y = (rng.random(n) < 0.12).astype(int)          # chỉ 12% dương
        s = np.clip(rng.normal(0.25 + 0.35 * y, 0.18), 0, 1)
        thr, best = self.t.best_threshold_binary(y, s, self.f1)
        assert best >= self.f1(y, (s >= 0.5).astype(int)) - 1e-12
        assert best > self.f1(y, (s >= 0.5).astype(int)), "phải TỐT HƠN ngưỡng mặc định"

    def test_fit_half_gain_is_always_inflated(self):
        """Tín hiệu THẬT, n=3000. Điều DUY NHẤT luôn đúng: mức tăng đo trên nửa FIT
        bao giờ cũng lớn hơn mức tăng thật trên nửa EVAL.
        (Kiểm chứng qua 8 seed × 2 kịch bản — không có ngoại lệ.)"""
        rng = np.random.default_rng(0)
        n = 3000
        y = (rng.random(n) < 0.12).astype(int)
        s = np.clip(rng.normal(0.25 + 0.35 * y, 0.18), 0, 1)
        gf, ge = self.t.tuning_gain_holdout(y, s, self.f1, n_repeat=12, seed=0)
        assert gf > 0, "trên nửa FIT, tune luôn >= 0 theo định nghĩa"
        assert ge < gf, "mức tăng PHẢI co lại khi sang dữ liệu chưa thấy"

    def test_worth_tuning_flags_the_imbalanced_class(self):
        """🔴 Quy tắc phòng thi: chỉ tune ngưỡng khi |precision − recall| của lớp nào đó LỚN.

        Dựng lại đúng hình dạng dữ liệu R-ViHSD: lớp 0 chiếm đa số, lớp 2 hiếm và
        bị dự đoán THỪA (rò rỉ từ lớp 0 sang) -> precision sập, recall cao.
        Đó chính là TEENCODE thật: P=0.21 / R=0.41, và tune ở đó ăn +0.02.

        Lưu ý đa lớp là tổng-bằng-không: đẩy mẫu VÀO một lớp cũng làm lệch các lớp
        khác. Nên phải để lớp đông chịu phần rò rỉ (mẫu số lớn -> ảnh hưởng loãng)."""
        rng = np.random.default_rng(0)
        n = 6000
        y = rng.choice([0, 1, 2], n, p=[0.70, 0.22, 0.08])
        p = y.copy()
        sym = (rng.random(n) < 0.10) & (y != 2)          # nhầm đối xứng 0<->1
        p[sym] = 1 - p[sym]
        leak = (y == 0) & (rng.random(n) < 0.12)         # lớp 0 rò sang lớp 2
        p[leak] = 2
        gap = np.asarray(self.t.worth_tuning(y, p, 3))
        assert gap.shape == (3,)
        assert gap.argmax() == 2, f"phải chỉ ra lớp 2 (bị đoán thừa), nhận {np.round(gap,4)}"
        assert gap[2] > 2 * max(gap[0], gap[1]), \
            f"lớp lệch phải rõ hơn HẲN các lớp cân, nhận {np.round(gap,4)}"

    def test_holdout_exposes_noise_fitting(self):
        """🔴 BÀI HỌC CỐT LÕI: score thuần nhiễu vẫn cho gain_fit > 0.

        Đo thật: gain_fit = +0.0039  nhưng  gain_eval = −0.0089
        -> tune ngưỡng LÀM HẠI. Chỉ nhìn gain_fit thì bạn tin nhầm.

        Ở vòng trường, task hate (85% điểm) đo 4 lần cho −0.0021 / +0.0012 /
        −0.0008 / −0.0014: đổi dấu, đều bé -> tune KHÔNG LÀM GÌ CẢ. Còn task
        noise đo 4 lần đều dương (+0.0095..+0.0244). Xem worth_tuning()."""
        rng = np.random.default_rng(1)
        n = 1000
        y = (rng.random(n) < 0.5).astype(int)
        s = rng.random(n)                      # KHÔNG mang thông tin gì về y
        gf, ge = self.t.tuning_gain_holdout(y, s, self.f1, n_repeat=8, seed=0)
        assert gf > 0, "tune trên nửa FIT luôn tạo ra mức tăng ẢO"
        assert ge < gf / 2, "mức tăng ảo đó phải biến mất trên nửa EVAL"

    def test_class_bias_improves(self):
        rng = np.random.default_rng(1)
        n, k = 3000, 3
        y = rng.choice(k, n, p=[0.7, 0.2, 0.1])
        logits = rng.normal(size=(n, k)); logits[np.arange(n), y] += 1.1
        proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        base = self.f1(y, proba.argmax(1))
        bias, best = self.t.best_class_bias(y, proba, self.f1, n_iter=6)
        assert best >= base - 1e-12 and len(bias) == k


# ══════════════════════════════════════════════ BT 06 — VÒNG LẶP TRAIN
torch = pytest.importorskip("torch")
import torch.nn as nn
from torch.utils.data import TensorDataset


def _toy(n=512, d=16, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, d, generator=g) * 3.0
    y = torch.randint(0, k, (n,), generator=g)
    X = centers[y] + torch.randn(n, d, generator=g)
    return TensorDataset(X, y)


class TestTrainLoop:
    def setup_method(self):
        self.t = load("06_train_loop")

    def test_seed_everything_is_reproducible(self):
        import random
        outs = []
        for _ in range(2):
            self.t.seed_everything(123)
            outs.append((random.random(), float(np.random.rand()), float(torch.rand(1))))
        assert outs[0] == outs[1], "cùng seed phải cho cùng kết quả ở CẢ BA thư viện"

    def test_optimizer_excludes_bias_and_norm_from_decay(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        opt = self.t.build_optimizer(model, lr=1e-3, weight_decay=0.01)
        assert len(opt.param_groups) == 2, "phải có 2 nhóm: decay và no-decay"
        wds = sorted(g["weight_decay"] for g in opt.param_groups)
        assert wds[0] == 0.0 and wds[1] == 0.01
        no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0.0][0]
        # 2 bias Linear + weight&bias của BatchNorm = 4 tensor 1 chiều
        assert all(p.ndim <= 1 for p in no_decay["params"])
        assert len(no_decay["params"]) == 4

    def test_scheduler_warms_up_then_decays(self):
        """Khảo sát RIÊNG đường cong LR: bước scheduler mà KHÔNG bước optimizer.

        PyTorch sẽ cảnh báo "lr_scheduler.step() before optimizer.step()" — ở đây
        cảnh báo đó ĐÚNG và là chủ ý, nên ta tắt nó có khai báo. Trong vòng lặp
        huấn luyện thật thì KHÔNG được làm vậy: xem chú thích về GradScaler trong
        06_train_loop.py."""
        model = nn.Linear(4, 2)
        opt = self.t.build_optimizer(model, lr=1.0, weight_decay=0.0)
        sched = self.t.cosine_warmup(opt, num_warmup=10, num_total=100)
        lrs = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
            for _ in range(100):
                lrs.append(opt.param_groups[0]["lr"]); sched.step()
        assert lrs[0] < lrs[5] < lrs[10], "giai đoạn warmup phải TĂNG"
        assert lrs[10] > lrs[50] > lrs[95], "sau warmup phải GIẢM theo cosine"
        assert abs(max(lrs) - 1.0) < 1e-6, "đỉnh phải bằng lr đặt vào"

    def test_train_reduces_loss(self, tmp_path):
        self.t.seed_everything(0)
        tr = self.t.make_loader(_toy(512, seed=0), 64, True, num_workers=0)
        va = self.t.make_loader(_toy(256, seed=1), 64, False, num_workers=0)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
        model, hist = self.t.train(model, tr, va, epochs=6, lr=1e-2, patience=5,
                                   amp=False, device="cpu",
                                   ckpt=str(tmp_path / "b.pt"), log_every=0)
        assert len(hist) >= 3
        for key in ("epoch", "train_loss", "val_loss", "val_acc", "sec"):
            assert key in hist[0], f"history phải ghi '{key}'"
        assert hist[-1]["train_loss"] < hist[0]["train_loss"], "loss phải giảm"
        assert hist[-1]["val_acc"] > 0.8, "bài toán 3 cụm tách rời -> acc phải > 0.8"
        assert (tmp_path / "b.pt").exists(), "phải lưu checkpoint tốt nhất"

    def test_eval_mode_restored(self, tmp_path):
        """Bẫy kinh điển: quên model.train() sau khi evaluate -> BatchNorm đóng băng."""
        self.t.seed_everything(0)
        tr = self.t.make_loader(_toy(256, seed=0), 64, True, num_workers=0)
        va = self.t.make_loader(_toy(128, seed=1), 64, False, num_workers=0)
        model = nn.Sequential(nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
        model, _ = self.t.train(model, tr, va, epochs=3, amp=False, device="cpu",
                                ckpt=str(tmp_path / "c.pt"), log_every=0)
        bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)][0]
        assert bn.running_mean.abs().sum() > 0, "BatchNorm phải đã cập nhật running stats"
