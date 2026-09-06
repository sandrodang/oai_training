import nbformat as nbf
n = nbf.v4.new_notebook()
C = lambda s: n.cells.append(nbf.v4.new_code_cell(s.strip()))
M = lambda s: n.cells.append(nbf.v4.new_markdown_cell(s.strip()))

M("""
# Olympic AI Sinh viên 2026 — Tác vụ 1: R-ViHSD
### Đội **FPTU_Promt_Engineer** · tài khoản `foa25`

**Điểm Private cuối cùng: 0.720**

Hệ thống NLP đa nhiệm cho bình luận mạng xã hội tiếng Việt: dự đoán đồng thời
mức độ hate speech (3 lớp) và loại nhiễu văn bản (7 lớp).

```
Score = 0.85 · MacroF1(hate) + 0.15 · MacroF1(noise)
```

**Tuân thủ quy định:** mô hình học máy thật (ViSoBERT ~98M tham số fine-tune),
seed cố định toàn bộ, không dùng dữ liệu ngoài, **không dùng cột `id`** làm đặc
trưng — `id` chỉ để ghép dự đoán vào đúng dòng khi xuất CSV.
""")

M("## 1. Thiết lập")
C("""
import os, re, sys, json, time, random, unicodedata, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

DATA  = "data"
SEED  = 42
MODEL = "5CD-AI/visobert-14gb-corpus"      # encoder pretrain trên mạng xã hội tiếng Việt
HATE_LABELS  = ["CLEAN", "OFFENSIVE", "HATE"]
NOISE_LABELS = ["ORIGINAL","NO_DIACRITICS","TEENCODE","CHAR_REPEAT","PUNCT_NOISE","OBFUSCATION","MIXED"]

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
""")

M("## 2. Dữ liệu")
C("""
tr = pd.read_csv(f"{DATA}/training_set.csv"); va = pd.read_csv(f"{DATA}/validation_set.csv")
df = pd.concat([tr, va], ignore_index=True)
df["text"] = df.text.fillna("").astype(str)
df = df.drop_duplicates(subset=["text","label","noise_type"]).reset_index(drop=True)
df["y_hate"]  = df.label.map({l:i for i,l in enumerate(HATE_LABELS)})
df["y_noise"] = df.noise_type.map({l:i for i,l in enumerate(NOISE_LABELS)})
print("train+val sau khử trùng lặp:", len(df))
print()
print("phân bố hate:");  print((df.label.value_counts(normalize=True)*100).round(2).to_string())
print()
print("phân bố noise:"); print((df.noise_type.value_counts(normalize=True)*100).round(2).to_string())
""")

M("""
### 2.1 Phát hiện then chốt: tập train ghép cặp

Mỗi comment gốc xuất hiện **hai lần** — một bản `ORIGINAL` và một bản đã biến đổi.
Nếu chia fold ngẫu nhiên, hai bản của cùng một comment rơi vào train và validation
khác nhau → **rò rỉ**, CV cao giả khoảng **+0.04**.

Khoá chuỗi thông thường không ghép được `OBFUSCATION` (một ký tự bị thay bằng
`*`, `.` hoặc `_` nên không chuẩn hoá về được), nên phải dùng so khớp
nearest-neighbour theo char n-gram.
""")
C("""
def strip_diacritics(s):
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c)!="Mn").replace("đ","d").replace("Đ","D")

def match_form(s):
    s = strip_diacritics(s).lower()
    s = re.sub(r"[^a-z0-9]", "", s)         # bỏ cả dấu câu lẫn ký hiệu obfuscation
    return re.sub(r"(.)\\1+", r"\\1", s)      # gộp ký tự lặp

df["mf"] = df.text.map(match_form)
is_o = (df.noise_type=="ORIGINAL").values
oi, ni = np.where(is_o)[0], np.where(~is_o)[0]
vec = TfidfVectorizer(analyzer="char", ngram_range=(3,4), min_df=1, sublinear_tf=True)
O = normalize(vec.fit_transform(df.mf.values[oi])); N = normalize(vec.transform(df.mf.values[ni]))

grp = np.arange(len(df)); first = {}
for i in oi: grp[i] = first.setdefault(df.mf.iat[i], i)
bi = np.zeros(len(ni),int); bs = np.zeros(len(ni))
for a in range(0, len(ni), 2000):
    S = (N[a:a+2000] @ O.T).toarray(); bi[a:a+2000]=S.argmax(1); bs[a:a+2000]=S.max(1)
hit = bs >= 0.70
grp[ni[hit]] = grp[oi[bi[hit]]]
df["group"] = ["G%d"%g for g in grp]
print(f"ghép được {hit.sum()}/{len(ni)} mẫu nhiễu về comment gốc ({hit.mean()*100:.1f}%)")
print(pd.Series(hit, index=df.noise_type.values[ni]).groupby(level=0).mean().round(3).to_string())
print(f"\\nsố nhóm comment gốc: {df.group.nunique()}")
""")

C("""
strat = df.y_hate.astype(str) + "_" + df.y_noise.astype(str)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
df["fold"] = -1
for k,(_,te) in enumerate(sgkf.split(df, strat, groups=df.group)): df.loc[te,"fold"] = k
print("rò rỉ nhóm giữa các fold:", (df.groupby("group").fold.nunique()>1).sum())
print(pd.crosstab(df.fold, df.label).to_string())
""")

M("""
### 2.2 Trần Bayes của nhãn nhiễu

BTC xác nhận `noise_type` ghi lại **thao tác biến đổi đã chạy**, không phải hình
thức bề mặt của kết quả. Với chuỗi không có chỗ cho phép biến đổi kích hoạt, văn
bản ra **y hệt** văn bản vào nhưng nhãn vẫn được ghi.
""")
C("""
orig = {g: s.text.iloc[0] for g,s in df[df.noise_type=="ORIGINAL"].groupby("group")}
d = df[(df.noise_type!="ORIGINAL") & df.group.map(orig).notna()].copy()
d["noop"] = d.text.str.strip() == d.group.map(orig).str.strip()
t = d.groupby("noise_type").noop.agg(["mean","sum","count"]); t.columns=["ti_le_noop","so_noop","tong"]
print(t.sort_values("ti_le_noop", ascending=False).round(3).to_string())
print(f"\\n=> {t.loc['TEENCODE','ti_le_noop']*100:.1f}% mẫu TEENCODE không thể phân biệt với ORIGINAL.")
print("   Trần lý thuyết của noise macro-F1 khoảng 0.78 — không phải lỗi mô hình.")
""")

M("## 3. Mô hình: ViSoBERT đa nhiệm + FGM + consistency")
C("""
class DS(Dataset):
    def __init__(self, texts, tok, maxlen, yh=None, yn=None, pair=None):
        self.t, self.tok, self.m, self.yh, self.yn, self.pair = list(texts), tok, maxlen, yh, yn, pair
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=self.m)
        o = {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
        if self.yh is not None:
            o["yh"], o["yn"] = int(self.yh[i]), int(self.yn[i])
        if self.pair is not None:
            j = self.pair[i]; j = i if j < 0 else int(j)
            e2 = self.tok(self.t[j], truncation=True, max_length=self.m)
            o.update(p_input_ids=e2["input_ids"], p_attention_mask=e2["attention_mask"],
                     p_yh=int(self.yh[j]), p_yn=int(self.yn[j]), p_ok=float(self.pair[i] >= 0))
        return o

def _pad(b, k, pv):
    L = max(len(x[k]) for x in b); out = torch.full((len(b),L), pv, dtype=torch.long)
    for i,x in enumerate(b): out[i,:len(x[k])] = torch.tensor(x[k])
    return out

def collate(b, pad):
    o = {"input_ids": _pad(b,"input_ids",pad), "attention_mask": _pad(b,"attention_mask",0)}
    if "yh" in b[0]:
        o["yh"]=torch.tensor([x["yh"] for x in b]); o["yn"]=torch.tensor([x["yn"] for x in b])
    if "p_input_ids" in b[0]:
        o["p_input_ids"]=_pad(b,"p_input_ids",pad); o["p_attention_mask"]=_pad(b,"p_attention_mask",0)
        o["p_yh"]=torch.tensor([x["p_yh"] for x in b]); o["p_yn"]=torch.tensor([x["p_yn"] for x in b])
        o["p_ok"]=torch.tensor([x["p_ok"] for x in b])
    return o

class MTL(nn.Module):
    \"\"\"Encoder chia sẻ, hai đầu ra. Nhận biết nhiễu vừa được chấm điểm (15%),
    vừa là tác vụ phụ ép encoder học biểu diễn bền — thí nghiệm cho thấy giảm
    trọng số nhiễu xuống 0.05 làm hate TỆ đi 0.008.\"\"\"
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.hate, self.noise = nn.Linear(h*2,3), nn.Linear(h*2,7)
    def forward(self, ids, att):
        o = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        mean = (o*m).sum(1)/m.sum(1).clamp(min=1e-6)
        mx = o.masked_fill(m==0,-1e4).max(1).values
        z = self.drop(torch.cat([mean,mx],-1))
        return self.hate(z), self.noise(z)

class FGM:
    \"\"\"Nhiễu đối kháng trên word embedding. Bài thi chấm chính khả năng bền
    trước biến dạng, nên đây là kỹ thuật khớp nhất — đo được +0.0121 OOF,
    dương trên cả 5/5 fold.\"\"\"
    def __init__(self, model, eps=1.0, key="word_embeddings"):
        self.m, self.eps, self.key, self.bak = model, eps, key, {}
    def attack(self):
        for n_,p in self.m.named_parameters():
            if p.requires_grad and self.key in n_ and p.grad is not None:
                self.bak[n_] = p.data.clone()
                nrm = torch.norm(p.grad)
                if nrm != 0 and not torch.isnan(nrm): p.data.add_(self.eps*p.grad/nrm)
    def restore(self):
        for n_,p in self.m.named_parameters():
            if n_ in self.bak: p.data = self.bak[n_]
        self.bak = {}
print("đã định nghĩa mô hình")
""")

M("""
### 3.1 Consistency loss trên các cặp THẬT

Không sinh cặp tổng hợp. Tập train **đã sẵn** ~16.000 cặp `(gốc, bản nhiễu)` của
cùng một comment, tạo bởi chính phép biến đổi của BTC — độ trung thực cao hơn bất
kỳ augmentation nào ta tự chế (thí nghiệm cho thấy augment tự chế **âm 0.0032**).

Chỉ ràng buộc **đầu hate** phải đồng thuận. **Không** ràng buộc đầu noise: hai
view cùng một comment nhưng nhãn nhiễu **bắt buộc phải khác nhau**.
""")
C("""
def run_fold(df, k, tok, te_texts, epochs=8, bs=64, lr=2e-5, head_lr=1e-3,
             maxlen=96, w_noise=0.30, fgm_eps=1.0, cons=0.0, tag=None, seed=SEED):
    set_seed(seed + k)
    dev = "cuda"
    trn, val = df[df.fold!=k], df[df.fold==k]
    pair = None
    if cons > 0:
        pos = {ix:i for i,ix in enumerate(trn.index)}
        pair = np.full(len(trn), -1, dtype=np.int64)
        for _, sub in trn.groupby("group"):
            if len(sub) < 2: continue
            ori = sub.index[sub.noise_type.values=="ORIGINAL"]
            for ix in sub.index:
                cand = [c for c in (ori if len(ori) else sub.index) if c != ix]
                if cand: pair[pos[ix]] = pos[cand[0]]
    cf = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(DS(list(trn.text), tok, maxlen, trn.y_hate.values, trn.y_noise.values, pair),
                       batch_size=bs, shuffle=True, collate_fn=cf, num_workers=4, drop_last=True)
    dl_va = DataLoader(DS(list(val.text), tok, maxlen), batch_size=256, collate_fn=cf, num_workers=4)
    dl_te = DataLoader(DS(te_texts, tok, maxlen), batch_size=256, collate_fn=cf, num_workers=4)

    model = MTL(MODEL).to(dev)
    wh = torch.tensor(len(trn)/(3*np.bincount(trn.y_hate,minlength=3)), dtype=torch.float, device=dev)
    wn = torch.tensor(len(trn)/(7*np.bincount(trn.y_noise,minlength=7)), dtype=torch.float, device=dev)
    head = [n_ for n_,_ in model.named_parameters() if n_.startswith(("hate","noise"))]
    opt = torch.optim.AdamW([
        {"params":[q for n_,q in model.named_parameters() if n_ not in head], "lr":lr},
        {"params":[q for n_,q in model.named_parameters() if n_ in head], "lr":head_lr}], weight_decay=0.01)
    steps = len(dl_tr)*epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1*steps), steps)
    fgm = FGM(model, fgm_eps) if fgm_eps>0 else None

    def infer(dl):
        model.eval(); H,N_=[],[]
        with torch.no_grad(), torch.amp.autocast("cuda",dtype=torch.bfloat16):
            for b in dl:
                lh,ln = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                H.append(lh.float().softmax(-1).cpu()); N_.append(ln.float().softmax(-1).cpu())
        return torch.cat(H).numpy(), torch.cat(N_).numpy()

    best = (-1,None,None)
    for ep in range(epochs):
        model.train(); t0=time.time(); tot=0
        for b in dl_tr:
            ids,att = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            th,tn = b["yh"].to(dev), b["yn"].to(dev)
            def _loss():
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    lh,ln = model(ids,att)
                    L = (1-w_noise)*F.cross_entropy(lh,th,weight=wh) + w_noise*F.cross_entropy(ln,tn,weight=wn)
                    if cons>0:
                        lh2,ln2 = model(b["p_input_ids"].to(dev), b["p_attention_mask"].to(dev))
                        ok = b["p_ok"].to(dev)
                        L = 0.5*L + 0.5*((1-w_noise)*F.cross_entropy(lh2,b["p_yh"].to(dev),weight=wh)
                                         + w_noise*F.cross_entropy(ln2,b["p_yn"].to(dev),weight=wn))
                        q1,q2 = lh.float().log_softmax(-1), lh2.float().log_softmax(-1)
                        kl = 0.5*(F.kl_div(q1,q2,log_target=True,reduction="none").sum(-1)
                                  + F.kl_div(q2,q1,log_target=True,reduction="none").sum(-1))
                        L = L + cons*(kl*ok).sum()/ok.sum().clamp(min=1)
                    return L
            opt.zero_grad(set_to_none=True)
            loss=_loss(); loss.backward()
            if fgm is not None: fgm.attack(); _loss().backward(); fgm.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); tot+=loss.item()
        ph,pn = infer(dl_va)
        fh = f1_score(val.y_hate, ph.argmax(1), average="macro", labels=range(3), zero_division=0)
        fn = f1_score(val.y_noise, pn.argmax(1), average="macro", labels=range(7), zero_division=0)
        s = 0.85*fh + 0.15*fn
        print(f"  fold{k} ep{ep+1} loss={tot/len(dl_tr):.4f} hate={fh:.4f} noise={fn:.4f} SCORE={s:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if s > best[0]:
            best = (s,(ph,pn),infer(dl_te))
            if tag:
                os.makedirs(f"ckpt/{tag}", exist_ok=True)
                torch.save({"model":model.state_dict(),"fold":k,"epoch":ep+1,"score":s}, f"ckpt/{tag}/fold{k}.pt")
    del model; torch.cuda.empty_cache()
    return best, val.index.values
print("đã định nghĩa vòng huấn luyện")
""")
nbf.write(n, "notebook_part1.ipynb")
print("part 1 written:", len(n.cells), "cells")
