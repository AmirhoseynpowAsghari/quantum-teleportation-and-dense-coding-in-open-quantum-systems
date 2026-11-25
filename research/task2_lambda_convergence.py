"""TEST: is the lambda_max=1 pairing boundary a thermodynamic phase boundary
or a finite-momentum-grid artifact?

In 2D at T=0 the BCS linearised gap equation has a logarithmic Cooper
instability.  On a finite k-mesh the 1/|xi| sum is cut off at the level
spacing delta ~ 1/Nk^2, so

        lambda_max(Nk) = A(n,V) * ln(Nk) + B(n,V) + o(1).

If A(n,V) != 0 the condition lambda_max=1 defines an Nk-dependent pseudo-
boundary and must NOT be reported as a quantum critical point.
"""
import sys, json, time
sys.path.insert(0,'.'); sys.path.insert(0,'research')
import numpy as np
from solver import make_grid, kernel_array, normal_state

def lam(Nk, n, V, U=75.0, Dt=4.5):
    sk, ssin, eps, KX, KY, kx = make_grid(Nk, 1.0, V)
    kern = kernel_array(sk, ssin, U, Dt, V)
    nrm = normal_state(eps, n)
    N2 = float(Nk*Nk)
    xi = eps - nrm["mu"]
    invE = 1.0/np.clip(np.abs(xi), 1e-12, None)
    M = np.array([
        [-np.sum(kern*invE)/(4*N2), np.sum(kern*eps*invE)/(4*N2*4)],
        [-2*Dt*np.sum(invE)/N2,     2*Dt*np.sum(eps*invE)/(N2*4)]])
    return float(np.max(np.linalg.eigvals(M).real))

NKS=[101,201,301,401,601]
out={}
t0=time.time()
for V in [0.50,0.55,0.60,0.70,1.00]:
    for n in [0.90,0.95,1.00,1.02,1.05,1.10]:
        vals=[lam(Nk,n,V) for Nk in NKS]
        out[f"{V:.2f}|{n:.2f}"]=dict(V=V,n=n,vals=vals)
        print(f"V={V:.2f} n={n:.2f}  "+"  ".join(f"{N}:{v:8.4f}" for N,v in zip(NKS,vals)),flush=True)
# log fits
print("\n--- log-slope A = d lambda / d ln Nk ---")
fits={}
for k,d in out.items():
    x=np.log(np.array(NKS,float)); y=np.log(np.abs(d['vals']))*np.sign(d['vals'])
    A,B=np.polyfit(x,y,1)
    fits[k]=dict(A=float(A),B=float(B))
    print(f"V={d['V']:.2f} n={d['n']:.2f}  A={A:+.4f}  B={B:+.4f}")
json.dump(dict(Nks=NKS,out=out,fits=fits),open('research/data/lambda_convergence.json','w'),indent=1)
print('time',time.time()-t0)
