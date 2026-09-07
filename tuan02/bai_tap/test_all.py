"""Bộ chấm tự động Tuần 2.

    python3 -m pytest bai_tap/test_all.py -q           # chấm bài CỦA BẠN
    SOLUTION=1 python3 -m pytest bai_tap/test_all.py -q # chấm ĐÁP ÁN

Oracle: torch.nn.functional cho attention; ví dụ kinh điển của Sennrich cho BPE;
BLEU tự viết ở Tuần 1 cho bài dịch đầu-cuối.
"""
import os, math, importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
SRC = ROOT / ("dap_an" if os.environ.get("SOLUTION") else "bai_tap")


def load(name, folder=None):
    p = (folder or SRC) / f"{name}.py"
    if not p.exists():
        pytest.skip(f"chưa có {p}")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError:
        pytest.skip(f"{name}: chưa cài đặt")
    return mod


# ══════════════════════════════════════════════ BT 01 — ATTENTION
class TestAttention:
    def setup_method(self):
        self.a = load("01_attention")

    def test_matches_torch_reference(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
        got, w = self.a.scaled_dot_product_attention(q, k, v)
        exp = F.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(got, exp, atol=1e-5), "phải khớp F.scaled_dot_product_attention"
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5), \
            "trọng số attention mỗi hàng phải cộng lại bằng 1"

    def test_scaling_by_sqrt_dk(self):
        """Quên chia sqrt(d_k) -> softmax bão hoà khi d_k lớn. Đây là bug âm thầm."""
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 5, 64) for _ in range(3))
        _, w = self.a.scaled_dot_product_attention(q, k, v)
        ent = -(w * (w + 1e-12).log()).sum(-1).mean()
        assert ent > 0.5, "entropy quá thấp -> nhiều khả năng bạn quên chia sqrt(d_k)"

    def test_causal_mask(self):
        m = self.a.causal_mask(4)
        assert m.dtype == torch.bool and m.shape == (4, 4)
        assert m.equal(torch.ones(4, 4, dtype=torch.bool).tril()), "True = ĐƯỢC nhìn, tam giác dưới"

    def test_mask_blocks_future(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 5, 8) for _ in range(3))
        m = self.a.causal_mask(5)[None, None]
        _, w = self.a.scaled_dot_product_attention(q, k, v, m)
        assert torch.allclose(w.triu(1), torch.zeros_like(w.triu(1)), atol=1e-6), \
            "vị trí tương lai phải có trọng số 0"

    def test_mha_shape_and_heads(self):
        mha = self.a.MultiHeadAttention(32, 4)
        x = torch.randn(3, 7, 32)
        out, w = mha(x, x, x)
        assert out.shape == (3, 7, 32)
        assert w.shape == (3, 4, 7, 7), "attention weights phải có chiều n_heads"

    def test_mha_is_causal_with_mask(self):
        torch.manual_seed(0)
        mha = self.a.MultiHeadAttention(16, 2, dropout=0.0).eval()
        x = torch.randn(1, 6, 16)
        m = self.a.causal_mask(6)
        with torch.no_grad():
            a = mha(x, x, x, m)[0]
            x2 = x.clone(); x2[:, -1] += 5.0
            b = mha(x2, x2, x2, m)[0]
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5), \
            "đổi token CUỐI không được ảnh hưởng các vị trí TRƯỚC"


# ══════════════════════════════════════════════ BT 02 — TRANSFORMER
class TestTransformer:
    def setup_method(self):
        self.t = load("02_transformer")

    def _net(self, **kw):
        torch.manual_seed(0)
        d = dict(src_vocab=40, tgt_vocab=50, d_model=32, n_heads=4, d_ff=64,
                 n_enc=2, n_dec=2, dropout=0.0)
        d.update(kw)
        return self.t.Seq2SeqTransformer(**d)

    def test_positional_encoding_has_no_parameters(self):
        pe = self.t.PositionalEncoding(16, 50, dropout=0.0)
        assert len(list(pe.parameters())) == 0, "sinusoidal PE KHÔNG học -> phải là buffer"
        assert any(n == "pe" for n, _ in pe.named_buffers()), "phải register_buffer('pe', ...)"
        x = torch.zeros(1, 5, 16)
        assert not torch.allclose(pe(x), x), "PE phải thực sự cộng vào đầu vào"

    def test_forward_shape(self):
        net = self._net()
        src = torch.randint(1, 40, (3, 7)); tgt = torch.randint(1, 50, (3, 6))
        assert net(src, tgt).shape == (3, 6, 50)

    def test_model_is_causal(self):
        net = self._net().eval()
        src = torch.randint(1, 40, (2, 5)); tgt = torch.randint(1, 50, (2, 6))
        with torch.no_grad():
            a = net(src, tgt)
            t2 = tgt.clone(); t2[:, -1] = (t2[:, -1] + 7) % 50
            b = net(src, t2)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5), \
            "decoder rò rỉ tương lai -> mask sai. Bug này làm train tốt mà dịch hỏng."

    def test_tied_embeddings_share_storage(self):
        tied = self._net(tie_decoder_output=True)
        assert tied.out.weight is tied.tgt_emb.weight, "phải BUỘC cùng một tensor"
        untied = self._net(tie_decoder_output=False)
        assert sum(p.numel() for p in tied.parameters()) < \
               sum(p.numel() for p in untied.parameters()), "buộc trọng số phải GIẢM tham số"

    def test_padding_is_masked_out(self):
        """Đổi nội dung tại vị trí PAD không được đổi kết quả."""
        net = self._net().eval()
        src = torch.randint(1, 40, (2, 8)); src[:, 5:] = 0
        tgt = torch.randint(1, 50, (2, 4))
        with torch.no_grad():
            a = net(src, tgt)
            s2 = src.clone(); s2[:, 5:] = 0          # vẫn pad
            b = net(s2, tgt)
        assert torch.allclose(a, b, atol=1e-6)


# ══════════════════════════════════════════════ BT 03 — BPE
class TestBPE:
    def setup_method(self):
        self.b = load("03_bpe")
        # ví dụ kinh điển trong bài báo Sennrich et al. (2016), mục 3.2
        self.toy = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3

    def test_get_stats(self):
        vocab = {("l", "o", "w", "</w>"): 5, ("l", "o", "w", "e", "r", "</w>"): 2}
        st = self.b.get_stats(vocab)
        assert st[("l", "o")] == 7 and st[("o", "w")] == 7
        assert st[("w", "</w>")] == 5 and st[("w", "e")] == 2

    def test_merge_vocab(self):
        vocab = {("l", "o", "w", "</w>"): 5}
        assert self.b.merge_vocab(("l", "o"), vocab) == {("lo", "w", "</w>"): 5}

    def test_stats_find_the_three_way_tie(self):
        """Trong ví dụ Sennrich có HOÀ BA CHIỀU ở tần suất 9:
        ('e','s'), ('s','t'), ('t','</w>')  — mỗi cặp = newest×6 + widest×3.
        Cặp nào được gộp trước là do CÁCH PHÁ HOÀ của bạn, không có đáp án duy nhất."""
        vocab = {self.b.word_to_symbols(w): n for w, n in
                 [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]}
        st = self.b.get_stats(vocab)
        assert st[("e", "s")] == st[("s", "t")] == st[("t", "</w>")] == 9
        assert st[("w", "e")] == 8 and st[("l", "o")] == 7

    def test_first_merge_is_a_max_pair(self):
        merges = self.b.learn_bpe(self.toy, 10)
        assert merges[0] in {("e", "s"), ("s", "t"), ("t", "</w>")}, \
            f"merge đầu phải là một trong ba cặp tần suất 9; bạn gộp {merges[0]}"

    def test_discovers_shared_est_subword(self):
        """Đích đến quan trọng hơn đường đi: 'newest' và 'widest' phải CHIA SẺ
        một đơn vị con chứa 'est'. Đó chính là lý do BPE giúp ngôn ngữ chắp dính."""
        merges = self.b.learn_bpe(self.toy, 4)
        a = set(self.b.apply_bpe("newest", merges))
        b = set(self.b.apply_bpe("widest", merges))
        shared = a & b
        assert any("est" in t for t in shared), \
            f"phải tìm ra đơn vị chung chứa 'est'; chung được: {shared}"
        # ⚠️ Thử lại với 10 merge: ('w','est</w>') sẽ nuốt mất đơn vị chung,
        # 'newest' thành ['n','e','west</w>'] và KHÔNG còn chia sẻ gì với 'widest'.
        # Gộp quá tay PHÁ HUỶ chính thứ khiến BPE có giá trị. Đó là lý do
        # kích thước vocab là siêu tham số phải quét, không phải càng to càng tốt.

    def test_roundtrip_is_lossless(self):
        merges = self.b.learn_bpe(self.toy, 10)
        for w in ["low", "lower", "newest", "widest"]:
            assert self.b.decode(self.b.encode(w, merges)) == w

    def test_more_merges_fewer_tokens(self):
        texts = ["hling rong gu-ne-ma", "miu bnam tuk-ne", "ybhok hina om-ma"] * 20
        lens = []
        for n in (0, 20, 60):
            m = self.b.learn_bpe(texts, n)
            lens.append(sum(len(self.b.encode(t, m)) for t in texts))
        assert lens[0] > lens[1] > lens[2], "càng nhiều merge, chuỗi càng ngắn"

    def test_apply_bpe_respects_merge_order(self):
        """Luôn gộp cặp có RANK NHỎ NHẤT trước — không phải quét trái sang phải."""
        merges = [("a", "b"), ("b", "c")]
        assert self.b.apply_bpe("abc", merges) == ["ab", "c", "</w>"]


# ══════════════════════════════════════════════ BT 04 — GIẢI MÃ
@pytest.fixture(scope="module")
def tiny_mt():
    """Model bé đã train sơ trên corpus đồ chơi — đủ để so greedy vs beam."""
    T = load("02_transformer", ROOT / "dap_an")
    M = load("05_train_mt", ROOT / "dap_an")
    B = load("03_bpe", ROOT / "dap_an")
    torch.manual_seed(0)
    tr = M.read_tsv(ROOT / "data/train.tsv")[:600]
    merges = B.learn_bpe([s for s, _ in tr], 80)
    se = lambda s: B.encode(s, merges)
    te = lambda t: t.split()
    sv, _ = M.build_vocab([se(s) for s, _ in tr])
    tv, itos = M.build_vocab([te(t) for _, t in tr])
    ds = M.MTDataset(tr, se, te, sv, tv)
    dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=M.collate)
    net = T.Seq2SeqTransformer(len(sv), len(tv), d_model=64, n_heads=4, d_ff=128,
                               n_enc=2, n_dec=2, dropout=0.0, pad_id=M.PAD)
    M.train_model(net, dl, dl, epochs=4, warmup=100, device="cpu")
    src, _, _ = M.collate([ds[0]])
    return net, src, M


class TestDecoding:
    def setup_method(self):
        self.d = load("04_decoding")

    def test_greedy_stops_at_eos(self, tiny_mt):
        net, src, M = tiny_mt
        out = self.d.greedy_decode(net, src, M.BOS, M.EOS, max_len=20)
        assert isinstance(out, list) and isinstance(out[0], list)
        assert M.EOS not in out[0], "phải CẮT tại eos, không trả về eos"
        assert M.BOS not in out[0], "không trả về bos"

    def test_beam1_equals_greedy(self, tiny_mt):
        net, src, M = tiny_mt
        g = self.d.greedy_decode(net, src, M.BOS, M.EOS, max_len=20)[0]
        b = self.d.beam_search(net, src, M.BOS, M.EOS, beam_size=1, max_len=20,
                               length_penalty=0.0)
        assert g == b, "beam_size=1 PHẢI trùng greedy"

    def test_beam_finds_higher_scoring_sequence(self, tiny_mt):
        """Với length_penalty=0 (thuần tổng logprob), beam rộng hơn không được TỆ HƠN."""
        net, src, M = tiny_mt
        g = self.d.greedy_decode(net, src, M.BOS, M.EOS, max_len=20)[0]
        b = self.d.beam_search(net, src, M.BOS, M.EOS, beam_size=5, max_len=20,
                               length_penalty=0.0)
        sg = self.d.sequence_logprob(net, src, g, M.BOS)
        sb = self.d.sequence_logprob(net, src, b, M.BOS)
        assert sb >= sg - 1e-4, f"beam {sb:.3f} không được thấp hơn greedy {sg:.3f}"

    def test_length_penalty_changes_length(self, tiny_mt):
        """lp=0 thiên vị câu NGẮN (mỗi bước cộng thêm một logprob âm).
        Tăng lp -> chuẩn hoá theo độ dài -> cho phép câu dài hơn."""
        net, src, M = tiny_mt
        short = self.d.beam_search(net, src, M.BOS, M.EOS, beam_size=5, max_len=25,
                                   length_penalty=0.0)
        long_ = self.d.beam_search(net, src, M.BOS, M.EOS, beam_size=5, max_len=25,
                                   length_penalty=1.5)
        assert len(long_) >= len(short)


# ══════════════════════════════════════════════ BT 05 — HUẤN LUYỆN
class TestTrainMT:
    def setup_method(self):
        self.m = load("05_train_mt")

    def test_collate_pads_and_shifts(self):
        batch = [([5, 6, 7], [1, 8, 9, 2]), ([5], [1, 8, 2])]
        src, tin, tout = self.m.collate(batch)
        assert src.shape == (2, 3) and src[1].tolist() == [5, 0, 0], "phải đệm PAD=0"
        assert tin[0].tolist() == [1, 8, 9], "tgt_in = bỏ token CUỐI"
        assert tout[0].tolist() == [8, 9, 2], "tgt_out = bỏ token ĐẦU (dịch trái 1)"

    def test_noam_schedule_shape(self):
        fn = self.m.noam_lambda(256, warmup=100)
        vals = [fn(s) for s in range(1, 400)]
        peak = int(np.argmax(vals)) + 1
        assert 90 <= peak <= 110, f"đỉnh phải ở quanh warmup=100, đang ở {peak}"
        assert vals[0] < vals[99] and vals[99] > vals[-1], "tăng rồi giảm"

    @pytest.mark.slow
    def test_end_to_end_bleu(self):
        """Ánh xạ Kơtu->Việt là XÁC ĐỊNH, nên Transformer đúng PHẢI đạt BLEU cao.
        BLEU thấp => code sai, không phải 'bài khó'."""
        T = load("02_transformer"); B = load("03_bpe"); D = load("04_decoding")
        M = self.m
        BL = load("02_bleu", ROOT.parent / "tuan01" / "dap_an")
        torch.manual_seed(0)
        tr = M.read_tsv(ROOT / "data/train.tsv")
        dv = M.read_tsv(ROOT / "data/dev.tsv")[:150]
        merges = B.learn_bpe([s for s, _ in tr], 150)
        se = lambda s: B.encode(s, merges); te = lambda t: t.split()
        sv, _ = M.build_vocab([se(s) for s, _ in tr])
        tv, itos = M.build_vocab([te(t) for _, t in tr])
        dtr = M.MTDataset(tr, se, te, sv, tv); ddv = M.MTDataset(dv, se, te, sv, tv)
        dl_tr = DataLoader(dtr, batch_size=64, shuffle=True, collate_fn=M.collate)
        dl_dv = DataLoader(ddv, batch_size=64, shuffle=False, collate_fn=M.collate)
        net = T.Seq2SeqTransformer(len(sv), len(tv), d_model=128, n_heads=4, d_ff=256,
                                   n_enc=2, n_dec=2, dropout=0.1, pad_id=M.PAD)
        net, _ = M.train_model(net, dl_tr, dl_dv, epochs=12, warmup=300, device="cpu")
        hyps = []
        for i in range(0, len(ddv), 64):
            src, _, _ = M.collate([ddv[j] for j in range(i, min(i + 64, len(ddv)))])
            for row in D.greedy_decode(net, src, M.BOS, M.EOS, max_len=24):
                hyps.append(" ".join(itos[t] for t in row))
        bleu = BL.corpus_bleu(hyps, [t for _, t in dv])
        print(f"\n  BLEU = {bleu:.2f}")
        assert bleu > 40, f"BLEU {bleu:.2f} quá thấp — pipeline có lỗi ở đâu đó"


# ══════════════════════════════ M07 — CHUẨN HOÁ · KHỞI TẠO · GRADIENT (Tầng 1)
class TestNormInitGradFlow:
    def setup_method(self):
        self.m = load("06_norm_init_gradflow")

    def test_layer_norm_matches_torch(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16)
        got = self.m.LayerNormScratch(16)(x)
        exp = nn.LayerNorm(16)(x)
        assert torch.allclose(got, exp, atol=1e-5), "phải khớp nn.LayerNorm"

    def test_layernorm_is_batch_invariant_but_batchnorm_is_not(self):
        """Đổi các mẫu KHÁC trong batch: LayerNorm không đổi, BatchNorm đổi hẳn.
        Đây là câu trả lời thật cho 'vì sao Transformer dùng LayerNorm'."""
        torch.manual_seed(0)
        x = torch.randn(8, 32)
        d_ln = self.m.batch_sensitivity(self.m.LayerNormScratch(32), x)
        d_bn = self.m.batch_sensitivity(nn.BatchNorm1d(32), x)
        assert d_ln < 1e-5, f"LayerNorm phải BẤT BIẾN theo batch, đang lệch {d_ln:.2e}"
        assert d_bn > 0.1, f"BatchNorm phải bị batch kéo đi, chỉ lệch {d_bn:.2e}"

    def test_he_keeps_variance_alive_xavier_decays_naive_explodes(self):
        """He giữ phương sai; Xavier TẮT DẦN vì ReLU vứt nửa tín hiệu; naive NỔ.
        Cả ba đều không crash — đó là lý do phải ĐO."""
        he = self.m.activation_variance(init="he")[-1]
        xa = self.m.activation_variance(init="xavier")[-1]
        na = self.m.activation_variance(init="naive")[-1]
        assert 0.01 < he < 100, f"He phải giữ phương sai sống, đang {he:.2e}"
        assert xa < he / 100, f"Xavier phải tắt dần với ReLU, đang {xa:.2e} vs He {he:.2e}"
        assert na > 1e6 or na == float("inf"), f"naive std=1.0 phải nổ, đang {na:.2e}"

    def test_preln_gradient_reaches_first_layer_far_better_than_postln(self):
        """Lý do BT 02 tuần 2 bắt dùng Pre-LN, đo bằng số thay vì học thuộc."""
        pre = self.m.grad_norm_first_layer("pre")
        post = self.m.grad_norm_first_layer("post")
        assert pre > 100 * post, \
            f"Pre-LN phải đưa gradient xuống tầng đáy tốt hơn nhiều: pre={pre:.2e} post={post:.2e}"
