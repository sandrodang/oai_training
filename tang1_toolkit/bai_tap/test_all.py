"""Bộ chấm — Tầng 1 Toolkit (bộ đồ nghề huấn luyện).

    python3 -m pytest bai_tap/test_all.py -q
    SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q
"""
import os, importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F

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


def tiny():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))


# ═══════════════════════════════ 01 — TRUNG BÌNH TRỌNG SỐ  → dùng ở TUẦN 3
class TestWeightAveraging:
    def setup_method(self):
        self.w = load("01_weight_averaging")

    def test_checkpoint_average_is_arithmetic_mean(self):
        sds = []
        for v in (1.0, 3.0, 5.0):
            m = tiny()
            with torch.no_grad():
                m[0].weight.fill_(v)
            sds.append({k: t.clone() for k, t in m.state_dict().items()})
        avg = self.w.average_state_dicts(sds)
        assert torch.allclose(avg["0.weight"], torch.full_like(avg["0.weight"], 3.0)), \
            "phải là trung bình cộng (1+3+5)/3 = 3"

    def test_integer_buffers_take_last_not_mean(self):
        """num_batches_tracked của BatchNorm là số NGUYÊN — trung bình nó là vô nghĩa."""
        sds = []
        for v in (10, 20, 30):
            m = tiny(); m.state_dict()["1.num_batches_tracked"].fill_(v)
            sds.append({k: t.clone() for k, t in m.state_dict().items()})
        avg = self.w.average_state_dicts(sds)
        assert avg["1.num_batches_tracked"].item() == 30, "tensor nguyên phải lấy bản CUỐI"

    def test_ema_moves_then_restores_exactly(self):
        m = tiny()
        ema = self.w.EMA(m, decay=0.9)
        w0 = m[0].weight.data.clone()
        with torch.no_grad():
            m[0].weight.add_(1.0)
        ema.update(m)
        ema.apply_to(m)
        assert not torch.allclose(m[0].weight.data, w0 + 1.0), "EMA phải làm trọng số KHÁC bản hiện tại"
        ema.restore(m)
        assert torch.allclose(m[0].weight.data, w0 + 1.0), "restore phải trả về CHÍNH XÁC bản cũ"

    def test_swa_equals_running_mean(self):
        m = tiny()
        swa = self.w.SWA(m)
        for v in (2.0, 4.0, 6.0):
            with torch.no_grad():
                m[0].weight.fill_(v)
            swa.update(m)
        sd = swa.state_dict(m)
        assert torch.allclose(sd["0.weight"], torch.full_like(sd["0.weight"], 4.0)), \
            "trung bình chạy (2+4+6)/3 = 4"

    def test_update_bn_is_noop_without_batchnorm(self):
        m = nn.Sequential(nn.Linear(4, 2))
        assert self.w.update_bn(m, [], device="cpu") is m


# ═══════════════════════════════ 02 — MẸO GRADIENT  → dùng ở TUẦN 3
class TestGradTricks:
    def setup_method(self):
        self.g = load("02_grad_tricks")

    def test_accumulate_steps(self):
        assert self.g.accumulate_steps(64, 16) == 4
        with pytest.raises(AssertionError):
            self.g.accumulate_steps(64, 15)

    def test_accumulation_equals_full_batch(self):
        """🔴 PHÉP KIỂM QUAN TRỌNG NHẤT: gradient tích luỹ PHẢI khớp gradient batch lớn.
        Sai lệch = bạn quên chia loss cho số micro-batch -> vô tình tăng LR N lần."""
        torch.manual_seed(0)
        net = nn.Linear(4, 2)
        x, y = torch.randn(8, 4), torch.randn(8, 2)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)   # lr=0 -> trọng số không đổi
        opt.zero_grad(); F.mse_loss(net(x), y).backward()
        g_full = net.weight.grad.clone()
        net.zero_grad()
        self.g.train_step_accum(net, [(x[:4], y[:4]), (x[4:], y[4:])],
                                F.mse_loss, opt, clip=None)
        assert torch.allclose(g_full, net.weight.grad, atol=1e-6), \
            "gradient tích luỹ lệch gradient batch lớn — nhiều khả năng quên chia /n"

    def test_accumulation_over_four_microbatches(self):
        torch.manual_seed(1)
        net = nn.Linear(6, 3)
        x, y = torch.randn(12, 6), torch.randn(12, 3)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        opt.zero_grad(); F.mse_loss(net(x), y).backward()
        g_full = net.weight.grad.clone(); net.zero_grad()
        micro = [(x[i:i + 3], y[i:i + 3]) for i in range(0, 12, 3)]
        self.g.train_step_accum(net, micro, F.mse_loss, opt, clip=None)
        assert torch.allclose(g_full, net.weight.grad, atol=1e-6)


# ═══════════════════════════════ 03 — FINE-TUNE ENCODER  → dùng ở TUẦN 6
class TestFinetuneLR:
    def setup_method(self):
        self.f = load("03_finetune_lr")

    def test_no_decay_group_holds_only_1d_params(self):
        gs = self.f.param_groups_no_decay(tiny(), weight_decay=0.01)
        assert len(gs) == 2
        nd = [g for g in gs if g["weight_decay"] == 0.0][0]
        assert all(p.ndim <= 1 for p in nd["params"])
        assert len(nd["params"]) == 4, "2 bias Linear + weight&bias BatchNorm"

    def test_llrd_increases_with_depth(self):
        layers = [nn.Linear(8, 8) for _ in range(4)]
        head = nn.Linear(8, 2)
        gs = self.f.llrd_param_groups(layers, head, base_lr=2e-5, decay=0.5)
        lrs = [g["lr"] for g in gs]
        assert min(lrs) < max(lrs), "tầng thấp phải có LR NHỎ HƠN tầng cao"
        assert abs(max(lrs) - 2e-5) < 1e-12, "head phải giữ base_lr"
        assert abs(min(lrs) - 2e-5 * 0.5 ** 3) < 1e-12, "tầng thấp nhất = base * decay^(L-1)"

    def test_gradual_unfreeze_is_monotonic(self):
        sch = self.f.gradual_unfreeze_schedule(6, 4)
        assert sch[0] == 0 and sch[-1] == 6, "bắt đầu chỉ head, cuối mở hết"
        assert all(a <= b for a, b in zip(sch, sch[1:])), "phải không giảm"

    def test_freeze_batchnorm(self):
        m = tiny()
        n = self.f.freeze_batchnorm(m)
        bn = m[1]
        assert n == 1 and not bn.training
        assert all(not p.requires_grad for p in bn.parameters())


# ═══════════════════════════════ 04 — ĐỐI KHÁNG  → dùng ở TUẦN 6
class _EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.emb(x).mean(1))


class TestAdversarial:
    def setup_method(self):
        self.a = load("04_adversarial")

    def test_fgm_perturbs_then_restores_exactly(self):
        torch.manual_seed(0)
        m = _EmbNet()
        F.cross_entropy(m(torch.randint(0, 10, (8, 5))), torch.randint(0, 2, (8,))).backward()
        w0 = m.emb.weight.data.clone()
        fgm = self.a.FGM(m, "emb", eps=1.0)
        fgm.attack()
        assert not torch.allclose(m.emb.weight.data, w0), "attack phải ĐỔI embedding"
        fgm.restore()
        assert torch.equal(m.emb.weight.data, w0), "restore phải khôi phục CHÍNH XÁC"

    def test_fgm_only_touches_embedding(self):
        torch.manual_seed(0)
        m = _EmbNet()
        F.cross_entropy(m(torch.randint(0, 10, (8, 5))), torch.randint(0, 2, (8,))).backward()
        fc0 = m.fc.weight.data.clone()
        fgm = self.a.FGM(m, "emb", eps=1.0); fgm.attack()
        assert torch.equal(m.fc.weight.data, fc0), "chỉ nhiễu embedding, không đụng lớp khác"
        fgm.restore()

    def test_fgm_step_accumulates_gradient(self):
        """Gradient sau FGM phải là TỔNG của gradient sạch + gradient đối kháng."""
        torch.manual_seed(0)
        x, y = torch.randint(0, 10, (8, 5)), torch.randint(0, 2, (8,))
        m1 = _EmbNet(); opt1 = torch.optim.SGD(m1.parameters(), lr=0.0)
        opt1.zero_grad(); F.cross_entropy(m1(x), y).backward()
        g_clean = m1.fc.weight.grad.clone()
        torch.manual_seed(0)
        m2 = _EmbNet(); opt2 = torch.optim.SGD(m2.parameters(), lr=0.0)
        self.a.fgm_train_step(m2, x, y, F.cross_entropy, opt2, "emb", eps=1.0)
        assert not torch.allclose(m2.fc.weight.grad, g_clean, atol=1e-7), \
            "phải CỘNG DỒN gradient đối kháng, không phải ghi đè"


# ═══════════════════════════════ 05 — LOSS  → TUẦN 4 (focal) & TUẦN 5 (dice/contrastive)
class TestLosses:
    def setup_method(self):
        self.l = load("05_losses")
        torch.manual_seed(0)
        self.lg, self.y = torch.randn(64, 3), torch.randint(0, 3, (64,))
        self.sl = torch.randn(4, 1, 16, 16)
        self.tg = (torch.rand(4, 1, 16, 16) > 0.8).float()

    def test_focal_gamma0_equals_cross_entropy(self):
        assert torch.allclose(self.l.focal_loss(self.lg, self.y, gamma=0.0),
                              F.cross_entropy(self.lg, self.y), atol=1e-6), \
            "gamma=0 PHẢI quy về cross-entropy — nếu không, công thức sai"

    def test_focal_downweights_easy_examples(self):
        assert self.l.focal_loss(self.lg, self.y, gamma=2.0).item() < \
               F.cross_entropy(self.lg, self.y).item()

    def test_dice_zero_on_perfect_prediction(self):
        perf = self.tg * 20.0 - (1 - self.tg) * 20.0
        assert self.l.dice_loss(perf, self.tg).item() < 1e-4
        assert self.l.dice_loss(-perf, self.tg).item() > 0.9, "đoán ngược -> loss gần 1"

    def test_tversky_reduces_to_dice(self):
        """alpha=beta=0.5 quy về Dice — kiểm với eps NHỎ.
        Với eps=1.0 hai công thức lệch ~4e-3 vì eps vào mẫu theo hệ số khác nhau."""
        d = self.l.dice_loss(self.sl, self.tg, eps=1e-6)
        t = self.l.tversky_loss(self.sl, self.tg, 0.5, 0.5, eps=1e-6)
        assert abs(d.item() - t.item()) < 1e-5

    def test_tversky_beta_favours_recall(self):
        """beta > alpha phạt FN nặng hơn -> loss cao hơn khi model BỎ SÓT."""
        under = self.tg * 20.0 - 20.0            # dự đoán gần như toàn 0 -> nhiều FN
        hi_fn = self.l.tversky_loss(under, self.tg, alpha=0.1, beta=0.9)
        hi_fp = self.l.tversky_loss(under, self.tg, alpha=0.9, beta=0.1)
        assert hi_fn > hi_fp, "beta cao phải phạt bỏ sót nặng hơn"

    def test_contrastive_pulls_and_pushes(self):
        a = torch.zeros(4, 8); b = torch.zeros(4, 8)
        same = self.l.contrastive_loss(a, b, torch.ones(4), margin=1.0)
        assert same.item() < 1e-6, "cùng cặp + khoảng cách 0 -> loss 0"
        diff = self.l.contrastive_loss(a, b, torch.zeros(4), margin=1.0)
        # dung sai 1e-4 chu khong phai 1e-6: F.pairwise_distance cong eps=1e-6 noi bo
        # nen khoang cach khong bao gio bang 0 tuyet doi.
        assert abs(diff.item() - 1.0) < 1e-4, "khác cặp + khoảng cách 0 -> loss = margin^2"

    def test_triplet_zero_when_well_separated(self):
        a = torch.zeros(4, 8); p = torch.zeros(4, 8)
        n = torch.full((4, 8), 10.0)
        assert self.l.triplet_loss(a, p, n, margin=1.0).item() < 1e-6


# ═══════════════════════════════ 06 — CONSISTENCY & ĐA NHIỆM  → dùng ở TUẦN 6
class TestConsistencyMultitask:
    def setup_method(self):
        self.c = load("06_consistency_multitask")

    def test_rdrop_equals_ce_when_logits_identical(self):
        """Hai lần forward giống hệt -> KL = 0 -> R-Drop quy về CE."""
        torch.manual_seed(0)
        lg = torch.randn(16, 3); y = torch.randint(0, 3, (16,))
        assert abs(self.c.rdrop_loss(lg, lg.clone(), y, alpha=1.0).item()
                   - F.cross_entropy(lg, y).item()) < 1e-5

    def test_rdrop_penalises_disagreement(self):
        torch.manual_seed(0)
        l1, l2 = torch.randn(16, 3), torch.randn(16, 3)
        y = torch.randint(0, 3, (16,))
        both = self.c.rdrop_loss(l1, l2, y, alpha=1.0).item()
        ce = 0.5 * (F.cross_entropy(l1, y) + F.cross_entropy(l2, y)).item()
        assert both > ce, "bất đồng giữa hai lần forward phải bị PHẠT"

    def test_consistency_detaches_clean_branch(self):
        """Nhánh sạch phải .detach() — nếu không, model 'gian lận' bằng cách kéo
        dự đoán sạch về phía nhiễu thay vì ngược lại."""
        clean = torch.randn(8, 3, requires_grad=True)
        noisy = torch.randn(8, 3, requires_grad=True)
        self.c.consistency_loss(clean, noisy).backward()
        assert clean.grad is None or torch.allclose(clean.grad, torch.zeros_like(clean.grad)), \
            "gradient KHÔNG được chảy ngược vào nhánh sạch"

    def test_uncertainty_weighting_learns(self):
        uw = self.c.UncertaintyWeighting(2)
        assert len(list(uw.parameters())) == 1, "log_var phải là nn.Parameter"
        opt = torch.optim.SGD(uw.parameters(), lr=0.5)
        for _ in range(60):
            opt.zero_grad()
            uw([torch.tensor(0.05), torch.tensor(3.0)]).backward()
            opt.step()
        w = uw.weights()
        assert w[0] > w[1], "nhiệm vụ loss THẤP (ít nhiễu) phải được trọng số CAO hơn"
