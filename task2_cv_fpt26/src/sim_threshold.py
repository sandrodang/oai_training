"""Kiem chung 3 luan diem ve threshold/metric bang mo phong.
Score model: normal ~ N(0,1), anomaly ~ N(d, s). 6 category d khac nhau.
"""
import numpy as np
rng = np.random.default_rng(0)

D = [2.6, 2.2, 1.3, 2.0, 1.5, 2.8]   # separability moi cat (cat03,05 yeu)
S = 1.3                               # anomaly co variance lon hon

def ba(sc, y, tau):
    p = (sc >= tau).astype(int)
    tpr = p[y==1].mean(); tnr = 1-p[y==0].mean()
    return 0.5*(tpr+tnr)

# ---------- 1) tau* toi uu co phu thuoc prevalence khong? ----------
print("=== (1) tau* va p* khi prevalence thay doi (population, n=400k) ===")
d = 2.0
for pi in [0.15, 0.25, 0.40, 0.55]:
    n = 400_000; y = (rng.random(n) < pi).astype(int)
    sc = np.where(y==1, rng.normal(d, S, n), rng.normal(0,1,n))
    grid = np.linspace(-1, 5, 601)
    bas = np.array([ba(sc,y,t) for t in grid])
    tau = grid[bas.argmax()]
    p_star = (sc >= tau).mean()                 # quantile tuong ung
    q_star = (sc[y==0] >= tau).mean()           # ti le normal vuot nguong = 1-TNR
    print(f"  pi={pi:.2f}  tau*={tau:5.2f}  BA*={bas.max():.4f}  p*(quantile tren TEST)={p_star:.3f}  q*(quantile tren NORMAL)={q_star:.3f}")

# ---------- 2) So sanh chien luoc calibration: p-quantile (test) vs q-quantile (normal) ----------
def draw(pi, n_per_cat, rng):
    """tra ve list (scores, y) cho 6 cat"""
    out=[]
    for d in D:
        y=(rng.random(n_per_cat)<pi).astype(int)
        sc=np.where(y==1, rng.normal(d,S,n_per_cat), rng.normal(0,1,n_per_cat))
        out.append((sc,y))
    return out

def macro(sets, taus):
    return np.mean([ba(sc,y,t) for (sc,y),t in zip(sets,taus)])

def fit_vertex(xs, ys):
    """quadratic fit, vertex CLIP ve trong khoang quan sat"""
    c=np.polyfit(xs,ys,2)
    if c[0]>=0: return xs[int(np.argmax(ys))]
    v=-c[1]/(2*c[0])
    return float(np.clip(v, min(xs), max(xs)))

PI_PUB, PI_PRI = 0.40, 0.25     # prevalence LECH giua public va private
GRID_P=[0.20,0.25,0.30,0.35,0.40,0.45,0.50]
GRID_Q=[0.02,0.05,0.08,0.12,0.16,0.20,0.25]
NREP=400
res={k:[] for k in ['A_argmax_p','B_fit_p','C_argmax_q','D_fit_q','ORACLE']}
for rep in range(NREP):
    r=np.random.default_rng(1000+rep)
    pub=draw(PI_PUB,80,r); pri=draw(PI_PRI,160,r)
    norm_hold=[r.normal(0,1,150) for _ in D]     # 150 train-normal held-out / cat

    # --- chien luoc p: nguong = quantile tren chinh tap test ---
    sc_p=[macro(pub,[np.quantile(s,1-p) for s,_ in pub]) for p in GRID_P]
    pA=GRID_P[int(np.argmax(sc_p))]; pB=fit_vertex(GRID_P,sc_p)
    res['A_argmax_p'].append(macro(pri,[np.quantile(s,1-pA) for s,_ in pri]))
    res['B_fit_p'].append(macro(pri,[np.quantile(s,1-pB) for s,_ in pri]))

    # --- chien luoc q: nguong = quantile tren train-normal held-out ---
    sc_q=[macro(pub,[np.quantile(nh,1-q) for nh in norm_hold]) for q in GRID_Q]
    qA=GRID_Q[int(np.argmax(sc_q))]; qB=fit_vertex(GRID_Q,sc_q)
    res['C_argmax_q'].append(macro(pri,[np.quantile(nh,1-qA) for nh in norm_hold]))
    res['D_fit_q'].append(macro(pri,[np.quantile(nh,1-qB) for nh in norm_hold]))

    g=np.linspace(-1,5,301)
    res['ORACLE'].append(np.mean([max(ba(s,y,t) for t in g) for s,y in pri]))

print(f"\n=== (2) Private macro-BA x100 | prevalence public={PI_PUB} -> private={PI_PRI} | {NREP} lan ===")
for k,v in res.items():
    v=np.array(v)*100
    print(f"  {k:12s} mean={v.mean():6.2f}  sd={v.std():4.2f}  (mat so voi oracle: {np.array(res['ORACLE']).mean()*100-v.mean():5.2f})")
