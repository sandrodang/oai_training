import numpy as np
D=[2.6,2.2,1.3,2.0,1.5,2.8]; S=1.3
def ba(sc,y,t):
    p=(sc>=t).astype(int); return 0.5*(p[y==1].mean()+(1-p[y==0].mean()))
def draw(pi,n,r,shift=0.0):
    return [(np.where((y:=(r.random(n)<pi).astype(int))==1, r.normal(d,S,n), r.normal(shift,1,n)),y) for d in D]
def macro(s,t): return np.mean([ba(a,y,x) for (a,y),x in zip(s,t)])
GP=[0.20,0.25,0.30,0.35,0.40,0.45,0.50]

def run(pi_pub,pi_pri,drift,nrep=500):
    o={'p_only':[],'q_derived':[],'hybrid_derived':[],'oracle':[]}
    for rep in range(nrep):
        r=np.random.default_rng(9000+rep)
        pub=draw(pi_pub,80,r,0.0); pri=draw(pi_pri,160,r,drift)
        nh=[r.normal(0,1,150) for _ in D]
        # SWEEP CHI 1 THAM SO p tren public LB
        sp=[macro(pub,[np.quantile(s,1-p) for s,_ in pub]) for p in GP]
        p=GP[int(np.argmax(sp))]
        tau_pub=[np.quantile(s,1-p) for s,_ in pub]
        # SUY RA q MIEN PHI: ti le train-normal vuot tau_pub  (khong ton submission)
        q=[max(1e-3,(h>=t).mean()) for h,t in zip(nh,tau_pub)]
        tp=[np.quantile(s,1-p) for s,_ in pri]
        tq=[np.quantile(h,1-qq) for h,qq in zip(nh,q)]
        o['p_only'].append(macro(pri,tp)); o['q_derived'].append(macro(pri,tq))
        o['hybrid_derived'].append(macro(pri,[(a+b)/2 for a,b in zip(tp,tq)]))
        g=np.linspace(-2,6,321); o['oracle'].append(np.mean([max(ba(s,y,t) for t in g) for s,y in pri]))
    return {k:np.mean(v)*100 for k,v in o.items()}

print(f"{'kich ban':44s} {'p only':>8s} {'q suy ra':>9s} {'HYBRID':>8s} {'oracle':>8s}")
for n,a in [("prevalence giong (0.40->0.40)",(0.40,0.40,0.0)),
            ("prevalence lech  (0.40->0.25)",(0.40,0.25,0.0)),
            ("prevalence lech  (0.40->0.55)",(0.40,0.55,0.0)),
            ("drift normal +0.35s, prev giong",(0.40,0.40,0.35)),
            ("drift normal +0.35s, prev lech",(0.40,0.25,0.35))]:
    r=run(*a); print(f"{n:44s} {r['p_only']:8.2f} {r['q_derived']:9.2f} {r['hybrid_derived']:8.2f} {r['oracle']:8.2f}")
