"""Đáp án — BT 04: Greedy & Beam search với length penalty."""
import torch


@torch.inference_mode()
def greedy_decode(model, src, bos_id, eos_id, max_len=64):
    """Trả list[list[int]] — chuỗi sinh ra (KHÔNG gồm bos, cắt tại eos)."""
    model.eval()
    memory, src_mask = model.encode(src)
    B = src.size(0)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits = model.decode(ys, memory, src_mask)[:, -1]     # (B, V)
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        ys = torch.cat([ys, nxt[:, None]], 1)
        done |= nxt.eq(eos_id)
        if done.all():
            break
    out = []
    for row in ys[:, 1:].tolist():
        out.append(row[:row.index(eos_id)] if eos_id in row else row)
    return out


@torch.inference_mode()
def beam_search(model, src, bos_id, eos_id, beam_size=4, max_len=64,
                length_penalty=1.0):
    """Beam search cho MỘT câu mỗi lần (src: (1, L)).

    score chuẩn hoá = sum(logprob) / (len ** length_penalty)
        length_penalty = 0.0 -> không phạt, THIÊN VỊ CÂU NGẮN
                               (vì mỗi bước cộng thêm một logprob ÂM)
        length_penalty = 1.0 -> logprob trung bình mỗi token
        0.6–0.7             -> khoảng thường dùng trong NMT
    """
    assert src.size(0) == 1, "cài đặt này giải mã từng câu một"
    model.eval()
    memory, src_mask = model.encode(src)
    dev = src.device

    beams = [(torch.tensor([[bos_id]], device=dev), 0.0)]   # (chuỗi, tổng logprob)
    finished = []
    for _ in range(max_len):
        cand = []
        for ys, sc in beams:
            logits = model.decode(ys, memory, src_mask)[:, -1]
            logp = logits.log_softmax(-1)[0]
            top_lp, top_ix = logp.topk(beam_size)
            for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                nys = torch.cat([ys, torch.tensor([[ix]], device=dev)], 1)
                cand.append((nys, sc + lp))
        cand.sort(key=lambda t: t[1], reverse=True)
        beams = []
        for ys, sc in cand:
            if ys[0, -1].item() == eos_id:
                finished.append((ys, sc))
            else:
                beams.append((ys, sc))
            if len(beams) >= beam_size:
                break
        if not beams:
            break

    pool = finished if finished else beams
    def norm(item):
        ys, sc = item
        n = max(ys.size(1) - 1, 1)                  # trừ bos
        return sc / (n ** length_penalty)
    best = max(pool, key=norm)[0][0, 1:].tolist()
    return best[:best.index(eos_id)] if eos_id in best else best


@torch.inference_mode()
def sequence_logprob(model, src, seq, bos_id):
    """Tổng log-prob mà model gán cho `seq` — dùng để so greedy vs beam."""
    model.eval()
    memory, src_mask = model.encode(src)
    dev = src.device
    ys = torch.tensor([[bos_id] + list(seq)], device=dev)
    logits = model.decode(ys[:, :-1], memory, src_mask)
    logp = logits.log_softmax(-1)[0]
    return float(sum(logp[i, t] for i, t in enumerate(seq)))
