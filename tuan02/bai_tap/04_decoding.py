"""BT 04 — GIẢI MÃ: GREEDY & BEAM SEARCH.  ⏱ ~1,5h · Ngày N6

Đây là chỗ ăn điểm BLEU rẻ nhất — cùng một model, chỉ đổi cách giải mã.
Chấm:  python3 -m pytest bai_tap/test_all.py -q -k Decoding
"""
import torch


@torch.inference_mode()
def greedy_decode(model, src, bos_id, eos_id, max_len=64):
    """Giải mã cả batch. Trả list[list[int]] — KHÔNG gồm bos, CẮT tại eos.

    Vòng lặp:
        ys = [[bos]] * B
        lặp max_len lần:
            logits = model.decode(ys, memory, src_mask)[:, -1]
            nxt = argmax
            câu nào đã xong thì ép nxt = eos (đừng để nó sinh tiếp rác)
            nối vào ys; dừng khi TẤT CẢ đã xong

    ⚠️ Nhớ model.eval() — dropout đang bật sẽ làm kết quả không tái lập.
    """
    raise NotImplementedError


@torch.inference_mode()
def beam_search(model, src, bos_id, eos_id, beam_size=4, max_len=64,
                length_penalty=1.0):
    """Beam search cho MỘT câu (src có shape (1, L)). Trả list[int].

    Mỗi beam giữ (chuỗi, TỔNG log-prob). Mỗi bước:
        - với từng beam, lấy top-k token tiếp theo
        - gom tất cả ứng viên, sắp xếp theo tổng logprob giảm dần
        - beam nào vừa sinh eos -> chuyển sang danh sách 'đã xong'
        - giữ lại beam_size beam chưa xong

    Cuối cùng chọn theo điểm CHUẨN HOÁ THEO ĐỘ DÀI:
        score = tổng_logprob / (độ_dài ** length_penalty)

    🔴 VÌ SAO CẦN CHUẨN HOÁ: mỗi token thêm vào cộng một logprob ÂM, nên tổng
       logprob luôn giảm khi câu dài ra. Không chuẩn hoá thì beam search
       LUÔN THIÊN VỊ CÂU NGẮN — và BLEU phạt câu ngắn rất nặng qua brevity
       penalty (bạn đã học ở Tuần 1). Hai lỗi này cộng dồn.

        length_penalty = 0.0  -> thuần tổng logprob, thiên vị câu ngắn
        length_penalty = 1.0  -> logprob trung bình mỗi token
        0.6 – 0.7             -> khoảng thường dùng trong NMT

    ⚠️ Beam quá lớn có thể làm BLEU GIẢM: model càng tự tin vào chuỗi có
       xác suất cao nhất thì càng hay cho ra câu ngắn, chung chung. Đây là
       hiện tượng đã biết trong NMT — quét beam ∈ {1,4,8} trên dev, đừng đoán.
    """
    raise NotImplementedError


@torch.inference_mode()
def sequence_logprob(model, src, seq, bos_id):
    """Tổng log-prob model gán cho `seq` (list[int], không có bos).
    Dùng để KIỂM CHỨNG beam thực sự tìm được chuỗi điểm cao hơn greedy.

    Cho model.decode chạy trên [bos] + seq[:-1], rồi cộng logp[i, seq[i]].
    """
    raise NotImplementedError
