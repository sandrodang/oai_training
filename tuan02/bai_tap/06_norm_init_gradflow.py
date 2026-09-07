"""M07 — CHUẨN HOÁ · KHỞI TẠO · DÒNG GRADIENT.  ⏱ ~1,5h · Tuần 2

Đây là lớp giải thích nằm DƯỚI bài Transformer tuần 2 (`tuan02/bai_tap/02_transformer.py`).
Không có module này thì "Pre-LN ổn định hơn Post-LN" chỉ là câu HỌC THUỘC.
Làm xong, đó là con số BẠN TỰ ĐO.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k NormInit
"""
import torch
import torch.nn as nn


# ─────────────────────────────────────────── a. LayerNorm vs BatchNorm
class LayerNormScratch(nn.Module):
    """Chuẩn hoá theo chiều CUỐI (feature), từng mẫu ĐỘC LẬP.

        mu  = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)     # ⚠️ unbiased=False
        y   = (x - mu) / sqrt(var + eps) * w + b

    Tham số: w khởi tạo 1, b khởi tạo 0, đều là nn.Parameter shape (d,).
    """
    def __init__(self, d, eps=1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


@torch.no_grad()
def batch_sensitivity(norm, x, seed=0):
    """🔴 THÍ NGHIỆM QUAN TRỌNG NHẤT MODULE NÀY.

    Giữ nguyên mẫu 0, thay TOÀN BỘ các mẫu khác trong batch bằng nhiễu.
    Đo |Δ| lớn nhất tại ĐẦU RA của mẫu 0.

        norm.train()                       # BatchNorm chỉ lộ vấn đề ở chế độ train
        y1 = norm(x)[0].clone()
        x2 = x.clone()
        x2[1:] = torch.randn(x2[1:].shape, generator=g) * 3.0 + 2.0
        return (y1 - norm(x2)[0]).abs().max().item()

    Kỳ vọng:  LayerNorm -> ~0      ·      BatchNorm -> LỚN (bậc đơn vị)

    ⚠️ ĐÂY LÀ CÂU TRẢ LỜI THẬT cho "vì sao Transformer dùng LayerNorm":
       batch NLP có **padding** và **độ dài thay đổi**, nên thống kê theo batch
       bị ô nhiễm bởi ô đệm và đổi theo việc batch tình cờ gồm những câu nào.
       Nói "vì nó chuẩn hoá theo feature" là chưa trả lời.
    """
    raise NotImplementedError


# ─────────────────────────────────────────── b. Khởi tạo
def activation_variance(depth=20, d=256, init="he", seed=0):
    """Phương sai kích hoạt sau MỖI tầng Linear(d,d,bias=False) + ReLU.
    Trả list độ dài `depth`. Chạy trong torch.no_grad(), x đầu vào randn(512, d).

    init ∈ {"he", "xavier", "naive"}:
        he     -> nn.init.kaiming_normal_(w, nonlinearity="relu")
        xavier -> nn.init.xavier_normal_(w)
        naive  -> nn.init.normal_(w, std=1.0)

    ⚠️ Bạn sẽ thấy điều BẤT NGỜ: **Xavier TẮT DẦN với ReLU**. Xavier được thiết kế
       cho tanh/sigmoid (đối xứng quanh 0); ReLU vứt một nửa tín hiệu nên cần
       hệ số gấp đôi — đó chính là He. Còn `naive` thì NỔ tới inf.
       Cả hai lỗi đều KHÔNG crash — mạng vẫn chạy, chỉ không học được.
    """
    raise NotImplementedError


# ─────────────────────────────────────────── c. Pre-LN vs Post-LN
def grad_norm_first_layer(mode="pre", n_layers=12, d=64, seed=0):
    """Dựng stack `n_layers` block residual, backward MỘT lần, trả norm gradient
    của tham số tầng ĐẦU TIÊN.

        pre :  x = x + ff(ln(x))       <- đường identity SẠCH từ đỉnh xuống đáy
        post:  x = ln(x + ff(x))       <- mỗi bước lùi phải qua thêm một LayerNorm

    ff = Sequential(Linear(d,d), ReLU(), Linear(d,d)); ln = nn.LayerNorm(d).
    Mất mát: net(x).pow(2).mean().backward() với x = randn(32, d).
    Trả: net[0].ff[0].weight.grad.norm().item()

    🔴 KỲ VỌNG: pre lớn hơn post NHIỀU BẬC. Đó là lý do Post-LN cần warmup dài
       còn Pre-LN thì không — và là lý do BT 02 tuần 2 bắt dùng Pre-LN.
    """
    raise NotImplementedError
