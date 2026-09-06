import nbformat as nbf
n = nbf.read("notebook_part1.ipynb", as_version=4)
C = lambda s: n.cells.append(nbf.v4.new_code_cell(s.strip()))
M = lambda s: n.cells.append(nbf.v4.new_markdown_cell(s.strip()))

M("""
## 4. Huấn luyện

Ba mô hình được huấn luyện trong vòng thi, mỗi mô hình 5 fold, tổng 15 checkpoint:

| tag | cấu hình | OOF |
|---|---|---|
| `visobert_clean` | ViSoBERT đa nhiệm, `w_noise=0.30` | 0.7018 |
| `visobert_fgm`   | + FGM ε=1.0 | 0.7139 |
| `visobert_cons`  | + FGM ε=1.0 + consistency 0.5 | 0.7186 |

Đặt `TRAIN = True` để huấn luyện lại từ đầu (~90 phút trên 1×H100). Mặc định
`False`: nạp checkpoint đã lưu trong vòng thi và nhật ký huấn luyện thật.
""")
C("""
TRAIN = False
CONFIGS = {
    "visobert_clean": dict(fgm_eps=0.0, cons=0.0, epochs=10),
    "visobert_fgm":   dict(fgm_eps=1.0, cons=0.0, epochs=8),
    "visobert_cons":  dict(fgm_eps=1.0, cons=0.5, epochs=8),
}
te_pub = pd.read_csv(f"{DATA}/public_test.csv"); te_pub["text"] = te_pub.text.fillna("").astype(str)

if TRAIN:
    tok = AutoTokenizer.from_pretrained(MODEL)
    for tag, cfg in CONFIGS.items():
        oh = np.zeros((len(df),3)); on = np.zeros((len(df),7)); seen = np.zeros(len(df), bool); TH,TN = [],[]
        for k in range(5):
            (s,(vh,vn),(th,tn)), idx = run_fold(df, k, tok, list(te_pub.text), tag=tag, **cfg)
            oh[idx],on[idx],seen[idx] = vh,vn,True; TH.append(th); TN.append(tn)
            print(f"[fold {k}] best SCORE={s:.4f}")
        np.savez(f"oof/{tag}_oof.npz", h=oh, n=on, seen=seen, te_h=np.mean(TH,0), te_n=np.mean(TN,0))
else:
    for tag in CONFIGS:
        print(f"=== {tag} — nhật ký huấn luyện thật trong vòng thi ===")
        log = f"logs/{tag}.log"
        if os.path.exists(log):
            for ln in open(log):
                if re.search(r"fold \\d\\]|OOF over", ln): print("   ", ln.rstrip())
        print()
""")

M("## 5. Đánh giá OOF (51.663 dòng — độ phân giải cao gấp ~4 lần bảng public 3.340 dòng)")
C("""
def official(yh,ph,yn,pn):
    fh = f1_score(yh,ph,average="macro",labels=range(3),zero_division=0)
    fn = f1_score(yn,pn,average="macro",labels=range(7),zero_division=0)
    return 0.85*fh+0.15*fn, fh, fn

OOF = {t: np.load(f"oof/{t}_oof.npz") for t in CONFIGS}
rows=[]
for t,d in OOF.items():
    m=d["seen"]; s,fh,fn = official(df.y_hate.values[m], d["h"][m].argmax(1), df.y_noise.values[m], d["n"][m].argmax(1))
    rows.append(dict(model=t, hate=round(fh,4), noise=round(fn,4), SCORE=round(s,4)))
print(pd.DataFrame(rows).to_string(index=False))
""")

C("""
d = OOF["visobert_cons"]; m = d["seen"]
print("=== hate, từng lớp (mô hình tốt nhất) ===")
print(classification_report(df.y_hate.values[m], d["h"][m].argmax(1), target_names=HATE_LABELS, digits=4, zero_division=0))
cm = confusion_matrix(df.y_hate.values[m], d["h"][m].argmax(1), labels=range(3))
print(pd.DataFrame(cm, index=HATE_LABELS, columns=HATE_LABELS).to_string())
print()
for i,nm in enumerate(HATE_LABELS):
    r=cm[i]; print(f"  thực tế {nm:9s} n={r.sum():5d} -> " + ", ".join(f"{HATE_LABELS[j]} {r[j]/r.sum()*100:5.1f}%" for j in range(3)))
print("\\n=> Lỗi chủ đạo là ĐỘC HẠI -> CLEAN, không phải ranh giới OFFENSIVE<->HATE.")
""")

M("""
## 6. Sai số của thước đo — bootstrap

Trước khi tin bất kỳ so sánh nào trên bảng public, phải biết bảng public sai số bao nhiêu.
""")
C("""
yh, yn = df.y_hate.values[m], df.y_noise.values[m]
ph, pn = d["h"][m].argmax(1), d["n"][m].argmax(1)
rng = np.random.RandomState(0); sc=[]
for _ in range(400):
    i = rng.choice(len(yh), 3340, replace=True)
    sc.append(official(yh[i],ph[i],yn[i],pn[i])[0])
sc = np.array(sc)
print(f"mô phỏng một lần chấm trên 3.340 mẫu (400 lần bootstrap):")
print(f"  trung bình = {sc.mean():.4f}")
print(f"  SD         = {sc.std():.4f}   <-- sai số chuẩn của MỘT điểm public/private")
print(f"  khoảng 95% = [{np.percentile(sc,2.5):.4f}, {np.percentile(sc,97.5):.4f}]")
print(f"\\nOOF trên {len(yh):,} dòng: {official(yh,ph,yn,pn)[0]:.4f}  (sai số ≈ ±{sc.std()/np.sqrt(len(yh)/3340):.4f})")
print("\\n=> Chênh lệch dưới ~0.023 trên bảng public KHÔNG phân biệt được.")
print("   Mọi quyết định chọn mô hình đều dựa trên OOF, không dựa trên public.")
""")

M("## 7. Ensemble — trọng số tối ưu trên OOF, có kiểm soát overfit")
C("""
import itertools
def search_w(Ps, y, ncls, steps=11):
    best, bw = -1, None
    for combo in itertools.product(range(steps+1), repeat=len(Ps)-1):
        if sum(combo) > steps: continue
        w = np.array(list(combo)+[steps-sum(combo)], float)/steps
        s = f1_score(y, sum(wi*p for wi,p in zip(w,Ps)).argmax(1), average="macro", labels=range(ncls), zero_division=0)
        if s > best: best, bw = s, w
    return bw, best

TAGS = ["visobert_fgm","visobert_cons","visobert_clean"]
mm = OOF[TAGS[0]]["seen"].copy()
for t in TAGS[1:]: mm &= OOF[t]["seen"]
Hs=[OOF[t]["h"][mm] for t in TAGS]; Ns=[OOF[t]["n"][mm] for t in TAGS]
yh2, yn2 = df.y_hate.values[mm], df.y_noise.values[mm]

idx = np.random.RandomState(0).permutation(mm.sum()); A,B = idx[::2], idx[1::2]
hold=[]
for fit,ev in ((A,B),(B,A)):
    w,_ = search_w([P[fit] for P in Hs], yh2[fit], 3)
    hold.append(f1_score(yh2[ev], sum(wi*p[ev] for wi,p in zip(w,Hs)).argmax(1), average="macro", labels=range(3), zero_division=0))
wh_, sh = search_w(Hs, yh2, 3); wn_, sn = search_w(Ns, yn2, 7)
print(f"hate : single tốt nhất={max(f1_score(yh2,P.argmax(1),average='macro',labels=range(3),zero_division=0) for P in Hs):.4f}"
      f"  ensemble held-out={np.mean(hold):.4f}   trọng số={np.round(wh_,3)}")
print(f"noise: single tốt nhất={max(f1_score(yn2,P.argmax(1),average='macro',labels=range(7),zero_division=0) for P in Ns):.4f}"
      f"  ensemble full-fit={sn:.4f}   trọng số={np.round(wn_,3)}")
""")

M("""
## 8. Hiệu chỉnh prior của đầu nhiễu

Tập test có phân bố nhiễu **khác** tập train. Ước lượng bằng bốn thống kê bề mặt
tất định, đều được hiệu chuẩn trên train nơi có nhãn thật.
""")
C("""
VN = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ".upper()
         + "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
def sig(t):
    t=str(t)
    return (sum(c in VN for c in t), len(re.findall(r"(\\w)\\1{2,}",t)),
            len(re.findall(r"[A-Za-zÀ-ỹ][*._][A-Za-zÀ-ỹ]",t)), t.count("  "), len(re.findall(r"[A-Za-zÀ-ỹ]",t)))
cols=["diac","rep3","obf","dsp","alpha"]
te_pri = pd.read_csv(f"{DATA}/private_test.csv"); te_pri["text"]=te_pri.text.fillna("").astype(str)
S = pd.DataFrame([sig(t) for t in te_pri.text], columns=cols)
obs = dict(nodiac=((S.diac==0)&(S.alpha>3)).mean(), obf=(S.obf>0).mean(), rep3=(S.rep3>0).mean(), dsp=(S.dsp>0).mean())
coef = dict(nodiac=(1.375,0.060), obf=(1.153,0.018), rep3=(0.976,0.031), dsp=(0.851,0.013))
qs = {k: (obs[k]-c[1])/c[0] for k,c in coef.items()}
q = float(np.mean(list(qs.values())))
print("tỉ lệ tín hiệu trên private test:", {k: round(v,4) for k,v in obs.items()})
print("q suy ra từ từng phương trình :", {k: round(v,4) for k,v in qs.items()})
print(f"\\n=> mỗi loại nhiễu ≈ {q*100:.1f}%,  ORIGINAL ≈ {(1-6*q)*100:.1f}%   (train: ORIGINAL 50%)")
TEST_PRIOR = np.array([1-6*q] + [q]*6); TEST_PRIOR = np.clip(TEST_PRIOR,1e-3,None); TEST_PRIOR /= TEST_PRIOR.sum()
""")

C("""
Pn = sum(wi*p for wi,p in zip(wn_, Ns))
train_p = np.bincount(yn2, minlength=7)/len(yn2)
sw = (TEST_PRIOR/train_p)[yn2]
wf1 = lambda yp: f1_score(yn2, yp, average="macro", labels=range(7), zero_division=0, sample_weight=sw)
L = np.log(np.clip(Pn,1e-9,1)); bias_n = np.zeros(7); best = wf1(L.argmax(1))
for _ in range(6):
    imp=False
    for c in range(7):
        bs,bb = best, bias_n[c]
        for g in np.linspace(-2,2,41):
            bias_n[c]=g; s=wf1((L+bias_n).argmax(1))
            if s>bs: bs,bb=s,g
        bias_n[c]=bb
        if bs>best+1e-9: best,imp=bs,True
    if not imp: break
print(f"noise macro-F1 theo prior TEST: argmax={wf1(L.argmax(1)):.4f} -> sau hiệu chỉnh={best:.4f}")
print("bias:", np.round(bias_n,2))
print("\\nLƯU Ý: hiệu chỉnh phải theo prior TEST. Tune theo prior TRAIN đẩy ORIGINAL")
print("lên ~34% (thật ~21%) và đã làm mất 0.005 điểm ở một lượt nộp public.")
""")

M("## 9. Suy luận trên Private Test và xuất bài nộp")
C("""
def predict(tag, texts, maxlen=96, bs=256):
    tok = AutoTokenizer.from_pretrained(MODEL)
    dl = DataLoader(DS(list(texts), tok, maxlen), batch_size=bs,
                    collate_fn=lambda b: collate(b, tok.pad_token_id), num_workers=4)
    H,N_ = [],[]
    for f in sorted([f"ckpt/{tag}/{x}" for x in os.listdir(f"ckpt/{tag}") if x.endswith(".pt")]):
        st = torch.load(f, map_location="cpu", weights_only=False)
        mdl = MTL(MODEL).cuda(); mdl.load_state_dict(st["model"]); mdl.eval()
        h,nn_ = [],[]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for b in dl:
                lh,ln = mdl(b["input_ids"].cuda(), b["attention_mask"].cuda())
                h.append(lh.float().softmax(-1).cpu()); nn_.append(ln.float().softmax(-1).cpu())
        H.append(torch.cat(h).numpy()); N_.append(torch.cat(nn_).numpy())
        del mdl; torch.cuda.empty_cache()
    return np.mean(H,0), np.mean(N_,0)

P = {t: predict(t, te_pri.text) for t in TAGS}
te_h = sum(wi*P[t][0] for wi,t in zip(wh_, TAGS))
te_n = sum(wi*P[t][1] for wi,t in zip(wn_, TAGS))
te_n = np.log(np.clip(te_n,1e-9,1)) + bias_n

sub = pd.DataFrame({
    "id": te_pri["id"],                                   # CHỈ để ghép dòng, không phải đặc trưng
    "pred_label":      [HATE_LABELS[i]  for i in te_h.argmax(1)],
    "pred_noise_type": [NOISE_LABELS[i] for i in te_n.argmax(1)],
})
sub.to_csv("task1_private_output.csv", index=False, encoding="utf-8")
print("đã ghi task1_private_output.csv  rows =", len(sub))
print()
print(sub.pred_label.value_counts().to_string()); print()
print(sub.pred_noise_type.value_counts().to_string())
print(f"\\nkiểm tra: ORIGINAL dự đoán {(sub.pred_noise_type=='ORIGINAL').mean()*100:.1f}%  (ước lượng {(1-6*q)*100:.1f}%)")
print("ID:", "khớp private_test" if set(sub.id)==set(te_pri.id) else "LỆCH", "| trùng lặp:", sub.id.duplicated().sum())
""")

M("""
## 10. Kết quả

| bước | OOF | Private |
|---|---|---|
| TF-IDF baseline | — | — |
| ViSoBERT 5-fold (fold không rò rỉ) | 0.7018 | — |
| + FGM ε=1.0 | 0.7139 | — |
| + consistency trên cặp thật | 0.7186 | — |
| **ensemble 2 mô hình** | 0.7204 | **0.718** |
| **ensemble 3 mô hình** | **0.7220** | **0.720** |

**OOF dự báo đúng điểm private hai lần liên tiếp, sai số 0.002.**
Bảng public cùng lúc đó lệch tới 0.02 và chỉ sai hướng
(v6 0.703 → v7 0.702 → v8 0.697 trong khi OOF tăng đều 0.7018 → 0.7139 → 0.7204).

### Những gì đã thử và bị dữ liệu bác bỏ

| hướng | kết quả trên nền FGM (fold 0) |
|---|---|
| ε = 0.5 / ε = 2.0 | −0.0093 / −0.0116 → ε=1.0 tối ưu |
| Augmentation tổng hợp | −0.0032 (trùng chức năng với FGM) |
| GCE (loss chịu nhiễu nhãn) | −0.0033 (giúp noise, hại hate) |
| Đầu phân loại phân tầng | −0.0055 |
| LLRD 0.9 | −0.0022 |
| EMA 0.999 | −0.0006 |
| Dual-view raw + chuẩn hoá | −0.0041, tốn 2.1× compute |
| XLM-R large (560M) | −0.05, chậm 6× |
| BamiBERT (101M, tiếng Việt tổng quát) | kém ViSoBERT ngay từ epoch 1 |

**11 hướng thử, 2 thắng: FGM và consistency loss.**
Bài học chi phối: **khớp domain thắng kích thước** — ViSoBERT 98M pretrain trên
mạng xã hội tiếng Việt đánh bại XLM-R large 560M và BamiBERT, ba lần liên tiếp.
""")
nbf.write(n, "FPTU_Promt_Engineer_NLP_task1.ipynb")
print("notebook:", len(n.cells), "cells")
