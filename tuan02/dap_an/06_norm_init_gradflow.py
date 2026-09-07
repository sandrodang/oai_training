"""M07 — CHUẨN HOÁ · KHỞI TẠO · DÒNG GRADIENT.  Tuần 2 · ~1,5h

Đây là lớp giải thích nằm DƯỚI bài Transformer của tuần 2. Không có nó thì
"Pre-LN ổn định hơn" chỉ là câu học thuộc; có nó, đó là số bạn tự đo.
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────── a. LayerNorm vs BatchNorm
class LayerNormScratch(nn.Module):
    """Chuẩn hoá theo chiều CUỐI (feature), từng mẫu ĐỘC LẬP với nhau."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.w + self.b


@torch.no_grad()
def batch_sensitivity(norm, x, seed=0):
    """Giữ nguyên mẫu 0, thay TOÀN BỘ các mẫu khác bằng nhiễu. Đo |Δ| tại mẫu 0.

    LayerNorm -> ~0 (mỗi mẫu tự chuẩn hoá).
    BatchNorm -> LỚN (thống kê batch bị các mẫu khác kéo đi).
    Đây chính là lý do Transformer dùng LayerNorm: batch NLP có padding và
    độ dài thay đổi, nên thống kê theo batch bị ô nhiễm bởi ô đệm.
    """
    g = torch.Generator().manual_seed(seed)
    norm.train()                      # BatchNorm chỉ lộ vấn đề ở chế độ train
    y1 = norm(x)[0].clone()
    x2 = x.clone()
    x2[1:] = torch.randn(x2[1:].shape, generator=g) * 3.0 + 2.0
    y2 = norm(x2)[0]
    return (y1 - y2).abs().max().item()


# ─────────────────────────────────────────── b. Khởi tạo
def activation_variance(depth=20, d=256, init="he", seed=0):
    """Phương sai kích hoạt sau mỗi tầng Linear+ReLU. Trả list độ dài `depth`."""
    torch.manual_seed(seed)
    x = torch.randn(512, d)
    out = []
    for _ in range(depth):
        lin = nn.Linear(d, d, bias=False)
        if init == "he":       nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        elif init == "xavier": nn.init.xavier_normal_(lin.weight)
        elif init == "naive":  nn.init.normal_(lin.weight, std=1.0)
        else: raise ValueError(init)
        with torch.no_grad():
            x = torch.relu(lin(x))
        out.append(x.var().item())
    return out


# ─────────────────────────────────────────── c. Pre-LN vs Post-LN
class _Block(nn.Module):
    def __init__(self, d, mode):
        super().__init__()
        self.mode = mode
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        if self.mode == "pre":
            return x + self.ff(self.ln(x))        # đường identity SẠCH
        return self.ln(x + self.ff(x))            # LN nằm TRÊN đường residual


def grad_norm_first_layer(mode="pre", n_layers=12, d=64, seed=0):
    """Dựng stack n_layers block, backward một lần, trả norm gradient tầng ĐẦU.

    Post-LN: mỗi bước lùi phải đi qua thêm một LayerNorm -> gradient tới tầng đáy
    bị suy giảm. Đó là lý do Post-LN cần warmup dài; Pre-LN thì không.
    """
    torch.manual_seed(seed)
    net = nn.Sequential(*[_Block(d, mode) for _ in range(n_layers)])
    x = torch.randn(32, d)
    net(x).pow(2).mean().backward()
    return net[0].ff[0].weight.grad.norm().item()
