"""BT 00 — TORCH CƠ BẢN.  ⏱ ~1,5h · Tuần 1 · N5–N6

🔴 LÀM BÀI NÀY TRƯỚC BT 06 `train_loop.py`. BT 01–05 thuần numpy, nhưng từ BT 06
trở đi — và toàn bộ Tuần 2 — mọi thứ đứng trên bảy hàm dưới đây.

Bảy hàm này KHÔNG phải bài tập cú pháp. Mỗi hàm là một chỗ mà **bản sai vẫn chạy
và vẫn ra số** — đúng loại lỗi mà sổ assert (kế hoạch §4C) tồn tại để bắt.

Chấm:  python3 -m pytest bai_tap/test_all.py -q -k TorchBasics
"""
import torch
import torch.nn as nn


def batched_dot(a, b):
    """a, b: (B, D) -> (B,) tích vô hướng TỪNG HÀNG.

    ⚠️ `(a*b).sum()` cho một SỐ VÔ HƯỚNG, không phải vector (B,) — và nó
       không báo lỗi. Phải chỉ rõ `dim=-1`.
    """
    raise NotImplementedError


def add_feature_bias(x, bias):
    """x: (B, L, D) · bias: (D,) -> cộng bias theo chiều ĐẶC TRƯNG (chiều cuối).

    🔴 BẪY BROADCASTING — chỗ này hỏng âm thầm nhất cả bài:
       torch căn shape TỪ PHẢI SANG. Nếu ai đó đưa nhầm `bias` shape (L,)
       mà tình cờ L == D, phép cộng **vẫn chạy** và ra kết quả **sai hoàn toàn**
       (cộng theo chiều thời gian thay vì chiều đặc trưng).
       ⇒ Hãy KIỂM TRA shape rồi `raise ValueError` nếu bias.dim() != 1
         hoặc bias.shape[0] != x.shape[-1].
    """
    raise NotImplementedError


def split_heads(x, n_heads):
    """(B, L, D) -> (B, H, L, D//H).  Đây chính là bước 2 của MultiHeadAttention
    ở Tuần 2 BT 01.

        x.view(B, L, H, dh).transpose(1, 2)

    ⚠️ THỨ TỰ QUAN TRỌNG: view TRƯỚC rồi transpose SAU.
       Làm ngược (transpose trước) cho ra shape ĐÚNG nhưng dữ liệu XẾP SAI.
       Ném ValueError nếu D không chia hết cho n_heads.
    """
    raise NotImplementedError


def merge_heads(x):
    """(B, H, L, dh) -> (B, L, H*dh). Phải là nghịch đảo CHÍNH XÁC của split_heads.

    🔴 Ở đây có một lỗi to tiếng và một lỗi âm thầm:
       - `.view()` ngay sau `.transpose()` -> **RuntimeError** (bộ nhớ không liền).
         Đó là lỗi TỐT, bạn thấy ngay.
       - `.reshape()` thì **KHÔNG lỗi** — nó tự copy. Nhưng nếu bạn quên
         `.transpose(1,2)` trước đó thì reshape vẫn ra đúng shape (B,L,H*dh)
         với dữ liệu **trộn sai**, và không ai báo gì cả.
       ⇒ Cách an toàn: `.transpose(1,2).contiguous().view(...)`.
    """
    raise NotImplementedError


def masked_mean(x, mask):
    """x: (B, L, D) · mask: (B, L) bool, True = token THẬT (False = đệm).
    Trung bình theo chiều L, **bỏ qua ô đệm**. Câu toàn pad -> trả vector 0.

    ⚠️ `x.mean(1)` tính cả ô đệm -> pha loãng vector câu theo độ dài đệm.
       Không crash, chỉ làm điểm tệ đi. Đây là bản thu nhỏ của bài học pad_mask
       ở Tuần 2 (assert §4C #4).
    Gợi ý: mask.unsqueeze(-1) để broadcast, và clamp mẫu số tránh chia 0.
    """
    raise NotImplementedError


def grad_wrt(f, x):
    """Gradient của f(x) (trả vô hướng) theo x. Trả tensor cùng shape với x.

        x = x.detach().clone().requires_grad_(True)
        f(x).backward()
        return x.grad

    ⚠️ `.detach().clone()` để không đụng vào đồ thị của người gọi.
    """
    raise NotImplementedError


class RunningMean(nn.Module):
    """Trung bình luỹ kế. Trạng thái phải là **buffer**, KHÔNG phải Parameter.

    🔴 Vì sao quan trọng:
       - buffer: nằm trong `state_dict`, đi theo `.to(device)`,
         **KHÔNG** nằm trong `.parameters()` -> optimizer không đụng tới.
       - Parameter: optimizer sẽ **"học"** một con số vốn chỉ là thống kê.
         Model vẫn train, vẫn ra số, chỉ là thống kê bị bóp méo dần.
       Đây đúng là cơ chế của `running_mean`/`running_var` trong BatchNorm, và
       cũng là lý do `PositionalEncoding` ở Tuần 2 phải dùng `register_buffer`.

    __init__: register_buffer("total", zeros(())) và ("count", zeros(()))
    update(x): cộng dồn, trả trung bình hiện tại. Nhớ @torch.no_grad().
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def update(self, x):
        raise NotImplementedError


def running_loss(losses):
    """Cộng dồn loss để GHI LOG. Trả một float Python.

    🔴 Phải `.detach()` (hoặc `.item()`). Nếu cộng thẳng tensor còn gắn đồ thị,
       đồ thị của MỌI batch bị giữ lại -> bộ nhớ phình dần tới khi hết VRAM.
       Vòng lặp vẫn chạy đúng vài chục batch đầu rồi mới sập — rất khó truy.
    """
    raise NotImplementedError
