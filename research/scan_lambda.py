import sys, json, time
sys.path.insert(0,'.'); sys.path.insert(0,'research')
import numpy as np
from sweep_core import setup
Nk=int(sys.argv[1]) if len(sys.argv)>1 else 201
ns=[0.5,0.75,1.0,1.25,1.4,1.6,1.875,1.95]
Vs=np.concatenate([np.arange(0.0,0.6,0.05),np.arange(0.6,1.55,0.05)])
out=[]
t0=time.time()
for n in ns:
    for V in Vs:
        g=setup(n,V,Nk)
        out.append(dict(n=float(n),V=float(V),lam=g['lam'],muN=g['nrm']['mu'],eN=g['nrm']['e']))
    print('n',n,'done',round(time.time()-t0,1),flush=True)
json.dump(dict(Nk=Nk,U=75.0,Dt=4.5,rows=out),open(f'research/data/lambda_scan_Nk{Nk}.json','w'),indent=1)
