import sys, json, time
sys.path.insert(0,'.'); sys.path.insert(0,'research')
import numpy as np
from sweep_core import setup, seeds_for, classify, dF
from solver import solve_all
Nk=int(sys.argv[1]); n=float(sys.argv[2]); Vmax=float(sys.argv[3])
Vs=np.arange(0.0,Vmax+1e-9,0.05)
prev=None; out=[]
t0=time.time()
for V in Vs:
    g=setup(n,V,Nk)
    sols,ntr=solve_all(g['eps'],g['kern'],Nk,4.5,n,seeds_for(g,prev))
    br=dF(g,classify(g,ntr))
    if br:
        pick=min(br,key=lambda b: np.max(np.abs(b['x']-prev))) if prev is not None else br[0]
        prev=pick['x']
    out.append(dict(V=float(V),lam=g['lam'],muN=g['nrm']['mu'],eN=g['nrm']['e'],
        branches=[[b['D0'],b['DS'],b['mu'],b['e'],b['dF']] for b in br]))
    print(f"n={n} V={V:.2f} lam={g['lam']:.4g} nb={len(br)} "+" | ".join(f"D0={b['D0']:+.4f} DS={b['DS']:+.4f} dF={b['dF']:+.3e}" for b in br),flush=True)
json.dump(dict(Nk=Nk,n=n,rows=out),open(f'research/data/boundary_n{n}_Nk{Nk}.json','w'),indent=1)
print('time',time.time()-t0)
