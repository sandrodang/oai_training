"""Wire on-the-fly augmentation into the trainer.
Augmented rows keep their hate label and have the noise loss MASKED, so the
noise head never learns our reconstruction of the organisers' operators."""
p = 'src/train_mtl.py'; s = open(p).read()
assert '--aug' not in s, "already patched"

s = s.replace('p.add_argument("--cons", type=float, default=0.0)',
              'p.add_argument("--aug", type=float, default=0.0)   # prob. of noising a training row\np.add_argument("--cons", type=float, default=0.0)')

s = s.replace('''class DS(Dataset):
    def __init__(self, texts, tok, maxlen, yh=None, yn=None, pair=None):
        self.t, self.tok, self.m, self.yh, self.yn = list(texts), tok, maxlen, yh, yn
        self.pair = pair            # index of the same source comment, other noise form''',
'''class DS(Dataset):
    def __init__(self, texts, tok, maxlen, yh=None, yn=None, pair=None, aug=None, augp=0.0):
        self.t, self.tok, self.m, self.yh, self.yn = list(texts), tok, maxlen, yh, yn
        self.pair = pair            # index of the same source comment, other noise form
        self.aug, self.augp = aug, augp''')

s = s.replace('''    def _enc(self, i):
        x = self.t[i]
        return (self.tok(x[0], x[1], truncation=True, max_length=self.m)
                if isinstance(x, tuple) else self.tok(x, truncation=True, max_length=self.m))''',
'''    def _enc(self, i, noised=False):
        x = self.t[i]
        if noised and not isinstance(x, tuple):
            x = self.aug(x)
        return (self.tok(x[0], x[1], truncation=True, max_length=self.m)
                if isinstance(x, tuple) else self.tok(x, truncation=True, max_length=self.m))''')

s = s.replace('''    def __getitem__(self, i):
        e = self._enc(i)
        d = {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
        if self.yh is not None:
            d["yh"] = int(self.yh[i]); d["yn"] = int(self.yn[i])''',
'''    def __getitem__(self, i):
        did = self.aug is not None and self.augp > 0 and random.random() < self.augp
        e = self._enc(i, did)
        d = {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
        if self.yh is not None:
            d["yh"] = int(self.yh[i]); d["yn"] = int(self.yn[i])
            d["nmask"] = 0.0 if did else 1.0      # hate label survives, noise label does not''')

s = s.replace('''    if "yh" in batch[0]:
        out["yh"] = torch.tensor([b["yh"] for b in batch])
        out["yn"] = torch.tensor([b["yn"] for b in batch])
    if "p_input_ids" in batch[0]:''',
'''    if "yh" in batch[0]:
        out["yh"] = torch.tensor([b["yh"] for b in batch])
        out["yn"] = torch.tensor([b["yn"] for b in batch])
        out["nmask"] = torch.tensor([b.get("nmask", 1.0) for b in batch])
    if "p_input_ids" in batch[0]:''')

s = s.replace('''def crit(logits, target, weight, a):''',
'''def crit(logits, target, weight, a, mask=None):
    if mask is not None:
        if mask.sum() < 1:
            return logits.sum() * 0.0
        keep = mask > 0.5
        logits, target = logits[keep], target[keep]''')

s = s.replace('''                    L = (1 - a.w_noise) * crit(lh, th, wh, a) + a.w_noise * crit(ln, tn, wn, a)''',
'''                    nm = b["nmask"].to(dev) if "nmask" in b else None
                    L = (1 - a.w_noise) * crit(lh, th, wh, a) \\
                        + a.w_noise * crit(ln, tn, wn, a, nm)''')

s = s.replace('''    dl_tr = DataLoader(DS(list(trn["view"]), tok, a.maxlen, trn.y_hate.values, trn.y_noise.values, pair),''',
'''    augr = None
    if a.aug > 0:
        from augment import Augmenter, build as build_aug
        augr = Augmenter(build_aug(), random.Random(a.seed + k))
        print(f"  fold{k} augmentation p={a.aug}", flush=True)
    dl_tr = DataLoader(DS(list(trn["view"]), tok, a.maxlen, trn.y_hate.values,
                          trn.y_noise.values, pair, augr, a.aug),''')

s = s.replace('import numpy as np, pandas as pd, torch', 'import random\nimport numpy as np, pandas as pd, torch')
open(p, 'w').write(s); print("augmentation patched in")
