import numpy as np
D=[2.6,2.2,1.3,2.0,1.5,2.8]; S=1.3
def ba(sc,y,t):
    p=(sc>=t).astype(int); return 0.5*((p[y==1].mean() if (y==1).any() else 0)+(1-p[y==0].mean()))
def draw(pi,n,r,shift=0.0):
    return [( np.where((y:=(r.random(n)<pi).astype(int))==1, r.normal(d,S,n), r.normal(shift,1,n)), y) for d in D]
def macro(sets,taus): return np.mean([ba(s,y,t) for (s,y),t in zip(sets,taus)])
GP=[0.20,0.25,0.30,0.35,0.40,0.45,0.50]; GQ=[0.02,0.05,0.08,0.12,0.16,0.20,0.25]

def run(pi_pub,pi_pri,drift,nrep=400):
    out={'p_test':[], 'q_normal':[], 'hybrid':[], 'oracle':[]}
    for rep in range(nrep):
        r=np.random.default_rng(7000+rep)
        pub=draw(pi_pub,80,r,0.0); pri=draw(pi_pri,160,r,drift)
        nh=[r.normal(0,1,150) for _ in D]                      # train-normal held-out (KHONG drift)
        sp=[macro(pub,[np.quantile(s,1-p) for s,_ in pub]) for p in GP]
        sq=[macro(pub,[np.quantile(h,1-q) for h in nh]) for q in GQ]
        p=GP[int(np.argmax(sp))]; q=GQ[int(np.argmax(sq))]
        tp=[np.quantile(s,1-p) for s,_ in pri]; tq=[np.quantile(h,1-q) for h in nh]
        out['p_test'].append(macro(pri,tp)); out['q_normal'].append(macro(pri,tq))
        out['hybrid'].append(macro(pri,[(a+b)/2 for a,b in zip(tp,tq)]))
        g=np.linspace(-2,6,321); out['oracle'].append(np.mean([max(ba(s,y,t) for t in g) for s,y in pri]))
    return {k:np.mean(v)*100 for k,v in out.items()}

print(f"{'kich ban':46s} {'p-quantile(test)':>17s} {'q-quantile(normal)':>19s} {'hybrid':>8s} {'oracle':>8s}")
for name,args in [
  ("prevalence GIONG nhau (0.40 -> 0.40)",        (0.40,0.40,0.0)),
  ("prevalence LECH   (0.40 -> 0.25)",            (0.40,0.25,0.0)),
  ("prevalence LECH   (0.40 -> 0.55)",            (0.40,0.55,0.0)),
  ("prevalence giong + DRIFT normal +0.35 sigma", (0.40,0.40,0.35)),
  ("prevalence lech  + DRIFT normal +0.35 sigma", (0.40,0.25,0.35)),
]:
    r=run(*args)
    print(f"{name:46s} {r['p_test']:17.2f} {r['q_normal']:19.2f} {r['hybrid']:8.2f} {r['oracle']:8.2f}")
