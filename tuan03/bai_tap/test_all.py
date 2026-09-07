"""Bộ chấm tự động Tuần 3 — phần Tầng 1 (bộ đồ nghề học sâu).

    python3 -m pytest bai_tap/test_all.py -q            # chấm bài CỦA BẠN
    SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q # chấm ĐÁP ÁN
"""
import os, importlib.util
from pathlib import Path
import numpy as np
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



class TestWeightAveraging:
    def setup_method(self):
        self.w = load("06_weight_averaging")

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
        self.g = load("07_grad_tricks")

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


class _EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.emb(x).mean(1))


class TestAdversarial:
    def setup_method(self):
        self.a = load("08_adversarial")

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


class TestConsistencyMultitask:
    def setup_method(self):
        self.c = load("09_consistency_multitask")

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


# ══════════════════════════════════════════ M07 — CHUẨN HOÁ · KHỞI TẠO · GRADIENT (Tuần 2)


# ══════════════════════════════ BT 10 — TRANSFORMER HIỆN ĐẠI (RMSNorm·RoPE·GQA·SwiGLU)
class TestModernTransformer:
    def setup_method(self):
        self.m = load("10_modern_transformer")

    def test_rmsnorm_matches_formula_and_is_not_layernorm(self):
        """Trừ mean -> thành LayerNorm: vẫn chạy, vẫn hội tụ, chỉ là sai thứ mình tưởng."""
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16) + 5.0           # mean lệch hẳn khỏi 0
        got = self.m.RMSNorm(16)(x)
        exp = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(got, exp, atol=1e-5), "phải đúng công thức RMS"
        assert not torch.allclose(got, nn.LayerNorm(16)(x), atol=1e-3), \
            "RMSNorm KHÔNG được trừ mean — đang trùng LayerNorm"
        n = self.m.RMSNorm(16)
        assert len(list(n.parameters())) == 1, "chỉ một tham số học được (w), KHÔNG có bias"

    def test_rope_makes_dot_product_depend_only_on_relative_distance(self):
        """🔴 Đây LÀ cả điểm của RoPE, và là test duy nhất bắt được quy ước ghép cặp sai."""
        torch.manual_seed(0)
        cos, sin = self.m.build_rope_cache(64, 8)
        q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

        def dot(mi, ni):
            qa = self.m.apply_rope(q, cos[mi:mi + 1], sin[mi:mi + 1])
            ka = self.m.apply_rope(k, cos[ni:ni + 1], sin[ni:ni + 1])
            return (qa * ka).sum().item()

        assert abs(dot(5, 3) - dot(12, 10)) < 1e-4, \
            "cùng khoảng cách m-n=2 phải cho cùng tích vô hướng -> quy ước ghép cặp sai"
        assert abs(dot(5, 3) - dot(5, 4)) > 1e-3, \
            "khác khoảng cách phải cho khác kết quả -> RoPE chưa mã hoá gì cả"

    def test_rope_cache_shape_and_rejects_odd_dim(self):
        cos, sin = self.m.build_rope_cache(10, 6)
        assert cos.shape == (10, 3) and sin.shape == (10, 3)
        assert torch.allclose(cos[0], torch.ones(3)) and torch.allclose(sin[0], torch.zeros(3)), \
            "vị trí 0 -> góc 0 -> cos=1, sin=0"
        with pytest.raises(ValueError):
            self.m.build_rope_cache(10, 7)

    def test_repeat_kv_groups_contiguously(self):
        """x.repeat(1, n, 1, 1) xếp XEN KẼ và không báo lỗi — bẫy âm thầm."""
        x = torch.arange(2.0).view(1, 2, 1, 1)          # head 0 = 0, head 1 = 1
        out = self.m.repeat_kv(x, 3).flatten()
        assert out.tolist() == [0, 0, 0, 1, 1, 1], \
            f"phải lặp liền khối [0,0,0,1,1,1]; đang {out.tolist()} (dùng .repeat -> xen kẽ)"

    def test_gqa_equals_mha_when_kv_expanded(self):
        torch.manual_seed(0)
        q = torch.randn(1, 4, 3, 8)
        k1, v1 = torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 8)
        o_gqa, _ = self.m.grouped_query_attention(q, k1, v1)
        o_mha, _ = self.m.grouped_query_attention(
            q, k1.expand(1, 4, 3, 8).contiguous(), v1.expand(1, 4, 3, 8).contiguous())
        assert o_gqa.shape == (1, 4, 3, 8)
        assert torch.allclose(o_gqa, o_mha, atol=1e-6), \
            "GQA với Hkv=1 phải trùng MHA khi K,V được nhân bản sẵn"
        with pytest.raises(ValueError):     # Hq=4 không chia hết Hkv=3
            self.m.grouped_query_attention(q, torch.randn(1, 3, 3, 8), torch.randn(1, 3, 3, 8))

    def test_gqa_respects_mask(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
        mask = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
        _, attn = self.m.grouped_query_attention(q, k, v, mask)
        assert torch.allclose(attn.triu(1), torch.zeros_like(attn.triu(1)), atol=1e-6)

    def test_swiglu_gates_the_right_branch(self):
        """Đảo nhánh vẫn chạy, vẫn học — chỉ khác hàm. Kiểm bằng trọng số dựng tay."""
        f = self.m.FFN_SwiGLU(4, 4)
        assert all(l.bias is None for l in (f.gate, f.up, f.down)), "ba Linear đều bias=False"
        with torch.no_grad():
            f.gate.weight.copy_(torch.eye(4)); f.up.weight.copy_(torch.eye(4) * 2)
            f.down.weight.copy_(torch.eye(4))
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        exp = F.silu(x) * (2 * x)
        assert torch.allclose(f(x), exp, atol=1e-5), \
            "nhánh đi qua SiLU phải là GATE; đang đảo nhánh"

