"""Census of ALL self-consistent branches over a 2D (n,V) grid."""
import sys, json, time
sys.path.insert(0,'.'); sys.path.insert(0,'research')
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy

Nk=int(sys.argv[1]); 
ns=np.round(np.arange(0.85,1.51,0.05),3)
Vs=[0.0,0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.4,1.6]
def seeds(g):
    mu=g['nrm']['mu']
    S=[[1e-7,1e-7,mu],[0.001,0.004,mu],[0.01,0.04,mu],[0.03,-0.06,mu],
       [0.05,0.12,mu],[0.05,-0.12,mu],[0.15,0.4,mu],[0.15,-0.4,mu],
       [0.3,0.9,mu],[0.3,-0.9,mu],[0.3,1.9,mu],[0.3,-1.9,mu],
       [0.6,2.6,mu],[0.6,-2.6,mu],[1.0,3.2,mu],[1.0,-3.2,mu]]
    return [np.array(s,float) for s in S]
out=[]
t0=time.time()
for n in ns:
    for V in Vs:
        g=setup(n,V,Nk)
        sols,ntr=solve_all(g['eps'],g['kern'],Nk,4.5,n,seeds(g))
        br=[]
        for s in ntr:
            e,eg,dphi=sc_energy(s,g['eps'],Nk,n)
            br.append([float(s[0]),float(s[1]),float(s[2]),float(e),float(e-g['nrm']['e'])])
        br.sort(key=lambda b:b[3])
        out.append(dict(n=float(n),V=float(V),lam=float(g['lam']),muN=g['nrm']['mu'],
                        eN=g['nrm']['e'],nb=len(br),branches=br))
        tag='  <== MULTI' if len(br)>1 else ''
        print(f"n={n:.2f} V={V:.2f} lam={g['lam']:6.3f} nb={len(br)} "+" | ".join(f"D0={b[0]:+.4f} DS={b[1]:+.4f} dF={b[4]:+.2e}" for b in br)+tag,flush=True)
json.dump(dict(Nk=Nk,rows=out),open(f'research/data/branch_census_Nk{Nk}.json','w'),indent=1)
print('time',time.time()-t0)
