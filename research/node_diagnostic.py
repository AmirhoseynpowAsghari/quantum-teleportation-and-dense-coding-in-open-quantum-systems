"""Is there a BOGOLIUBOV gap-closing (nodal) transition where the Delta_k
node eps* = 4t D0/DS crosses the Fermi energy mu?"""
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'.'); sys.path.insert(0,'research')
import numpy as np
from sweep_core import setup
from solver import solve_all, sc_energy
def seeds(g):
    mu=g['nrm']['mu']
    return [np.array(s,float) for s in [[1e-7,1e-7,mu],[0.01,0.03,mu],[0.01,-0.03,mu],[0.05,0.15,mu],
      [0.05,-0.2,mu],[0.15,0.6,mu],[0.15,-0.7,mu],[0.3,1.3,mu],[0.3,-1.4,mu]]]
print(f"{'n':>5} {'V':>6} {'D0':>9} {'DS':>9} {'mu':>8} {'eps*':>8} {'eps*-mu':>9} {'minFS':>9}")
for n in [1.0,1.2,1.4,1.6,1.875]:
    for V in [0.4,0.525,0.7,0.9,1.1,1.235,1.4]:
        g=setup(n,V,201)
        sols,ntr=solve_all(g['eps'],g['kern'],201,4.5,n,seeds(g),tol=1e-11)
        if not ntr:
            print(f"{n:5.2f} {V:6.3f}   (normal)"); continue
        b=min(((sc_energy(s,g['eps'],201,n)[0],s) for s in ntr))[1]
        D0,DS,mu=b
        eps_star=4.0*D0/DS if abs(DS)>1e-12 else np.nan
        Dk=D0-DS*g['eps']/4.0
        xi=np.abs(g['eps']-mu); m=xi<0.01
        gapFS=np.min(np.abs(Dk[m])) if m.any() else np.nan
        print(f"{n:5.2f} {V:6.3f} {D0:+9.5f} {DS:+9.5f} {mu:8.4f} {eps_star:+8.4f} {eps_star-mu:+9.4f} {gapFS:9.2e}")
