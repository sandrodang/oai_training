import os, traceback
os.environ.setdefault("HF_HOME", "/home/namdp36/oai/work/hf")
from transformers import AutoTokenizer, AutoModel, AutoConfig
CANDS = [
    "uitnlp/visobert",
    "5CD-AI/visobert-14gb-corpus",
    "vinai/phobert-base-v2",
    "FacebookAI/xlm-roberta-large",
    "FacebookAI/xlm-roberta-base",
    "vinai/phobert-large",
]
for m in CANDS:
    try:
        cfg = AutoConfig.from_pretrained(m)
        tok = AutoTokenizer.from_pretrained(m)
        mod = AutoModel.from_pretrained(m)
        n = sum(p.numel() for p in mod.parameters())
        print(f"OK   {m}  params={n/1e6:.1f}M  hidden={cfg.hidden_size}  vocab={cfg.vocab_size}  maxpos={getattr(cfg,'max_position_embeddings','?')}")
        del mod
    except Exception as e:
        print(f"FAIL {m}  {type(e).__name__}: {str(e)[:200]}")
