"""Fast BdG solver for Hutchinson et al. model, validated against repo residual."""
import numpy as np
from scipy.optimize import fsolve

def make_grid(Nk, t=1.0, Vso=0.0):
    kx = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, kx)
    ssin = np.sqrt(np.sin(KX)**2 + np.sin(KY)**2)
    sk = 0.5*(np.cos(KX)+np.cos(KY))
    eps = np.empty((Nk,Nk,2))
    eps[...,0] = -2*t*(np.cos(KX)+np.cos(KY)) - 2*Vso*ssin
    eps[...,1] = -2*t*(np.cos(KX)+np.cos(KY)) + 2*Vso*ssin
    return sk, ssin, eps, KX, KY, kx

def kernel_array(sk, ssin, U, Dt, Vso, t=1.0):
    s_prime = np.array([1.0,-1.0])
    return (U + 8.0*Dt*sk[...,None] + 4.0*(Dt/t)*Vso*s_prime[None,None,:]*ssin[...,None])

def normal_state(eps, n):
    N2 = eps.shape[0]*eps.shape[1]
    flat = eps.ravel()
    k = int(round(n*N2))
    k = min(max(k,1), flat.size-1)
    # partition to find k-th order stats without full sort
    # lo = s[k-1], hi = s[k]
    part = np.partition(flat, [k-1, k])
    lo, hi = part[k-1], part[k]
    mu = 0.5*(lo+hi) if hi>lo else lo
    xi = eps-mu
    E = np.abs(xi)
    eg = 0.5*np.sum(xi-E)
    e = eg/N2 + mu*n
    return {"mu": float(mu), "e": float(e), "eg": float(eg)}

def sc_energy(x, eps, Nk, n):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk*Nk)
    Dks = D0 - (DS/4.0)*eps  # t=1
    E = np.clip(np.sqrt(Dks**2 + (eps-mu)**2), 1e-12, None)
    xi = eps-mu
    eg = 0.5*np.sum(xi - E + Dks**2/(2.0*E))
    e = eg/N2 + mu*n
    dphi = 0.5*np.sum(-1.0 + xi/E)
    return float(e), float(eg), float(dphi + n*N2)

def residual_jac(x, eps, kern, Nk, Dt, n):
    D0, DS, mu = float(x[0]), float(x[1]), float(x[2])
    N2 = float(Nk*Nk)
    Dks = D0 - (DS/4.0)*eps
    xi = eps-mu
    E = np.clip(np.sqrt(Dks**2+xi**2), 1e-12, None)
    invE = 1.0/E
    D_over_E = Dks*invE
    S1 = np.sum(kern*D_over_E)
    S2 = np.sum(D_over_E)
    r1 = D0 + S1/(4.0*N2)
    r2 = DS + 2.0*Dt*S2/N2
    r3 = np.sum(xi*invE)/(2.0*N2) - (1.0-n)
    # Jacobian
    xi2_E3 = (xi*xi)/(E**3)
    # d r1
    dr1_dD0 = 1.0 + np.sum(kern*xi2_E3)/(4.0*N2)
    dr1_dDS = np.sum(kern*(-eps/4.0)*xi2_E3)/(4.0*N2)
    dr1_dmu = np.sum(kern*Dks*xi/(E**3))/(4.0*N2)
    dr2_dD0 = 2.0*Dt*np.sum(xi2_E3)/N2
    dr2_dDS = 1.0 + 2.0*Dt*np.sum((-eps/4.0)*xi2_E3)/N2
    dr2_dmu = 2.0*Dt*np.sum(Dks*xi/(E**3))/N2
    dr3_dD0 = -np.sum(xi*Dks/(E**3))/(2.0*N2)
    dr3_dDS = np.sum(xi*Dks*eps/(4.0*(E**3)))/(2.0*N2)
    dr3_dmu = -np.sum((Dks*Dks)/(E**3))/(2.0*N2)
    J = np.array([[dr1_dD0, dr1_dDS, dr1_dmu],
                  [dr2_dD0, dr2_dDS, dr2_dmu],
                  [dr3_dD0, dr3_dDS, dr3_dmu]])
    return np.array([r1,r2,r3]), J

def residual_only(x, eps, kern, Nk, Dt, n):
    r,_ = residual_jac(x, eps, kern, Nk, Dt, n)
    return r

def _canon(x):
    x=np.asarray(x,float)
    if x[0] < -1e-9 or (abs(x[0])<=1e-9 and x[1]<0):
        x=np.array([-x[0],-x[1],x[2]])
    return x

def _jac(x, eps, kern, Nk, Dt, n):
    return residual_jac(x, eps, kern, Nk, Dt, n)[1]

def solve_one(eps, kern, Nk, Dt, n, x0, xtol=1.49e-12, use_jac=False,
              maxfev=600):
    try:
        sol, info, ier, _ = fsolve(residual_only, np.asarray(x0,float),
            args=(eps,kern,Nk,Dt,n), full_output=True, xtol=xtol, maxfev=maxfev,
            fprime=(_jac if use_jac else None))
        cand=np.asarray(sol,float)
        r=float(np.max(np.abs(residual_only(cand,eps,kern,Nk,Dt,n))))
        return cand, r, ier
    except Exception as e:
        return None, np.inf, 0

def solve_all(eps, kern, Nk, Dt, n, seeds, tol=1e-10, early=None):
    """early: if set, run the first `early` seeds, and only continue through
    the remaining seeds when no converged root was found."""
    sols=[]
    for i,x0 in enumerate(seeds):
        if early is not None and i >= early and sols:
            break
        cand,r,ier = solve_one(eps,kern,Nk,Dt,n,x0,
                               maxfev=(200 if early is not None else 600))
        if cand is not None and r<tol:
            c=_canon(cand)
            # dedupe
            if not any(np.max(np.abs(c-np.array(o)))<1e-7 for o in sols):
                sols.append(c)
    # separate trivial
    nontrivial=[s for s in sols if max(abs(s[0]),abs(s[1]))>1e-6]
    return sols, nontrivial

def lambda_max(eps, kern, Nk, Dt, mu):
    """Linearized gap eigenvalue at normal state (mu=mu_N)."""
    N2=float(Nk*Nk)
    xi=eps-mu
    invE=1.0/np.clip(np.abs(xi),1e-12,None)
    M00=-np.sum(kern*invE)/(4.0*N2)
    M01=np.sum(kern*eps*invE)/(4.0*N2*4.0)
    M10=-2.0*Dt*np.sum(invE)/N2
    M11=2.0*Dt*np.sum(eps*invE)/(N2*4.0)
    M=np.array([[M00,M01],[M10,M11]])
    eig=np.linalg.eigvals(M)
    return float(np.max(eig.real)), M, eig

def bdg_dict(x, eps, KX, KY, ssin):
    D0,DS,mu=float(x[0]),float(x[1]),float(x[2])
    Dks=D0-(DS/4.0)*eps
    E=np.clip(np.sqrt(Dks**2+(eps-mu)**2),1e-12,None)
    uks=np.sqrt(np.clip(0.5*(1.0+(eps-mu)/E),0,1))
    vks=np.sqrt(np.clip(0.5*(1.0-(eps-mu)/E),0,1))
    return {"Delta_ks":Dks,"E_ks":E,"u_ks":uks,"v_ks":vks,"KX":KX,"KY":KY,"epsilon_ks":eps,"sqrt_sin":ssin,"Delta0":D0,"DeltaS":DS,"mu":mu}

