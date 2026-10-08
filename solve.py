#!/usr/bin/env python3
"""solve.py -- every number and array for the paper
'What Does a Nonlinear Fokker-Planck Equation Buy a Mean Field Game?'

Stationary (ergodic) MFG on x in [-L, L], reflecting walls, V = h (x^2-1)^2.
Unknowns: u on the grid, lambda, mu (+ xbar or a dial value in the herding cases).
Closure: m = G(u; mu) (zero-flux stationary FP, derived per dial), HJB discretised by
central differences blended into a Godunov upwind Hamiltonian where the cell Peclet
number exceeds 1/2 (pure Godunov at Pe >= 1, so the scheme is monotone everywhere).
Newton on the full discrete system (bordered tridiagonal Jacobian, banded Schur solve).

Writes results.npz, numbers.json, CHECKS.txt beside itself.  Re-solves only with --force.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
L = 2.5
NU = 0.1
HB = 1.0           # barrier height h
M1 = 1.0           # total mass
EPS = 1e-3         # congestion regularisation
N0 = 400           # refinement ladder N0, 2N0, 4N0
NP = 1600          # production grid (= finest rung)
N_PLAN = 25600     # planner comparison grid (C8)
MACH = np.finfo(float).eps
STATS = {'mass_err_max': 0.0, 'n_solves': 0}


# ----------------------------------------------------------------------------- grid
class Grid:
    def __init__(self, N, L=L):
        self.N, self.L = N, L
        self.x = np.linspace(-L, L, N + 1)
        self.h = 2 * L / N
        w = np.full(N + 1, self.h)
        w[0] = w[-1] = self.h / 2
        self.w = w
        self.iref = int(np.argmin(np.abs(self.x - 1.0)))   # u(1) = 0 fixes the constant
        i = np.arange(N + 1)
        self.cup = np.where(i < N, i + 1, N - 1)           # ghost u_{N+1} = u_{N-1}
        self.cum = np.where(i > 0, i - 1, 1)               # ghost u_{-1} = u_1


def Vquart(x):
    return HB * (x ** 2 - 1) ** 2


# ----------------------------------------------------------------------------- dials
class Dial:
    """kind: D0 linear, D1 noise gamma, D2 capacity y, D3 congestion c (toll=True: + planner toll),
    D4 McKean-Vlasov theta, D5 conformity kappa."""

    def __init__(self, kind, val=0.0, eps=EPS, xbar=0.0, peak0=None, toll=False, nu=NU):
        self.kind, self.val, self.eps, self.xbar, self.peak0, self.toll, self.nu = kind, float(val), eps, xbar, peak0, toll, nu
        if kind == 'D3' and self.val > 0:
            self._build_cong()

    def linear(self):
        k, v = self.kind, self.val
        return k == 'D0' or (k in ('D1',) and v == 1.0) or (k in ('D2', 'D3') and v == 0.0)

    # D3: invert I(m) = int_1^m (eps+s)^c / s ds = (mu-u)/nu.  With w = ln m:
    # I = eps^c w + F(w), F(w) = int_0^w [(eps+e^s)^c - eps^c] ds  (tabulated, Simpson + spline)
    def _build_cong(self):
        c, eps = self.val, self.eps
        dw, wlo, whi = 5e-4, -45.0, 4.0
        w = np.linspace(wlo, whi, int(round((whi - wlo) / dw)) + 1)
        f = (eps + np.exp(w)) ** c - eps ** c
        F = cumulative_simpson(f, x=w, initial=0.0)
        F = F - F[int(round(-wlo / dw))]
        self.spl, self.wlo, self.whi, self.ec = CubicSpline(w, F), wlo, whi, eps ** c
        self.Finf = F[0] - c * eps ** (c - 1) * math.exp(wlo)

    def _g(self, w):
        c, eps = self.val, self.eps
        F = np.where(w < self.wlo, self.Finf + c * eps ** (c - 1) * np.exp(np.minimum(w, self.wlo)),
                     self.spl(np.clip(w, self.wlo, self.whi)))
        return self.ec * w + F

    def cong_inv(self, z):
        c, eps = self.val, self.eps
        ub1 = (z - self.Finf) / self.ec                                  # upper bounds of the root
        ub2 = np.where(z >= 0, np.log1p(c * np.maximum(z, 0.0)) / c, z / (1 + eps) ** c)
        w = np.minimum(ub1, ub2)                                         # Newton from the right: monotone
        for _ in range(100):
            dw = (self._g(w) - z) / (eps + np.exp(np.minimum(w, 50.0))) ** c
            w = w - dw
            if np.all(np.abs(dw) <= 1e-14 * np.maximum(1.0, np.abs(w))):
                break
        return w

    def density(self, u, mu, x, xbar):
        """m, dm/du, dm/dmu, dm/dxbar  (None if the closure is undefined)."""
        nu = self.nu
        z = (mu - u) / nu
        zero = np.zeros_like(u)
        if self.linear() or self.kind == 'D5':
            m = np.exp(np.minimum(z, 700.0))
            return m, -m / nu, m / nu, zero
        k, g = self.kind, self.val
        if k == 'D1':
            t = (g - 1) * z / g
            base = 1 + t
            if g < 1:
                if np.any(base <= 0):
                    return None
                lb = np.log1p(t)
                m = np.exp(np.minimum(lb / (g - 1), 700.0))
                mb = np.exp(np.minimum(lb * (2 - g) / (g - 1), 700.0))          # m / base
            else:
                pos = base > 0
                lb = np.log1p(np.where(pos, t, 0.0))
                m = np.where(pos, np.exp(np.minimum(lb / (g - 1), 700.0)), 0.0)
                mb = np.where(pos, np.exp(np.minimum(lb * (2 - g) / (g - 1), 700.0)), 0.0)
            d = -mb / (nu * g)
            return m, d, -d, zero
        if k == 'D2':
            invK = g / self.peak0
            E = np.exp(np.minimum(z, 700.0))
            m = E / (1 + E * invK)
            d = -m * (1 - m * invK) / nu
            return m, d, -d, zero
        if k == 'D3':
            m = np.exp(self.cong_inv(z))
            d = -m / (nu * (self.eps + m) ** g)
            return m, d, -d, zero
        if k == 'D4':
            m = np.exp(np.minimum((mu - u - g * (x - xbar) ** 2 / 2) / nu, 700.0))
            return m, -m / nu, m / nu, m * g * (x - xbar) / nu
        raise ValueError(k)

    def diffusion(self, u, mu):
        """a(x) and its derivatives in u and mu (only D1 has density-dependent noise)."""
        if self.kind == 'D1' and self.val != 1.0:
            g = self.val
            base = 1 + (g - 1) * (mu - u) / (self.nu * g)
            pos = base > 0
            a = self.nu * np.where(pos, base, 0.0)                     # = nu m^(gamma-1)
            au = np.where(pos, -(g - 1) / g, 0.0)
            return a, au, -au
        z = np.zeros_like(u)
        return np.full_like(u, self.nu), z, z

    def bcoef(self, m):
        """H = b(m) p^2/2 + beta p ; returns b, db/dm (HJB coefficient)."""
        if self.kind == 'D2' and self.val > 0:
            invK = self.val / self.peak0
            return 1 - m * invK, np.full_like(m, -invK)
        if self.kind == 'D3' and self.val > 0:
            c, e = self.val, self.eps
            if self.toll:   # H + m dH/dm = p^2 (eps+(1-c)m) / (2 (eps+m)^(c+1))
                return (e + (1 - c) * m) / (e + m) ** (c + 1), -c * (2 * e + (1 - c) * m) / (e + m) ** (c + 2)
            return (e + m) ** (-c), -c * (e + m) ** (-c - 1)
        return np.ones_like(m), np.zeros_like(m)

    def bvel(self, m):
        """velocity coefficient: v* = -(bvel p + beta)."""
        if self.kind == 'D3' and self.val > 0:
            return (self.eps + m) ** (-self.val)
        return self.bcoef(m)[0]

    def beta(self, x, xbar):
        if self.kind == 'D4':
            return self.val * (x - xbar), np.full_like(x, -self.val)
        z = np.zeros_like(x)
        return z, z

    def extraV(self, x, xbar):
        if self.kind == 'D5':
            return self.val * (x - xbar) ** 2 / 2, -self.val * (x - xbar)
        z = np.zeros_like(x)
        return z, z


# ----------------------------------------------------------------------------- residual + Jacobian
def residual(G, dial, V, M, X, mode='sym', jac=True, store=None):
    """mode 'sym': xbar fixed (= dial.xbar); 'xbar': xbar unknown; 'param': dial value unknown, xbar fixed.
    Unknowns X = [u_0..u_N, lambda, mu, (xbar | dial value)]."""
    N, h, x, w = G.N, G.h, G.x, G.w
    n = N + 1
    u, lam, mu = X[:n], X[n], X[n + 1]
    xbar = X[n + 2] if mode == 'xbar' else dial.xbar
    if mode == 'param':
        dial.val = X[n + 2]
    out = dial.density(u, mu, x, xbar)
    if out is None:
        return None
    m, m_u, m_mu, m_xb = out
    a, a_u, a_mu = dial.diffusion(u, mu)
    b, b_m = dial.bcoef(m)
    beta, beta_xb = dial.beta(x, xbar)
    Vx, Vx_xb = dial.extraV(x, xbar)
    up, um = u[G.cup], u[G.cum]
    D2 = (up - 2 * u + um) / h ** 2
    Dc = (up - um) / (2 * h)
    Dp = (up - u) / h
    Dm = (u - um) / h
    Hc = 0.5 * b * Dc ** 2 + beta * Dc
    q = b * Dc + beta
    apos = a > 0
    aa = np.where(apos, a, 1.0)
    Pe = np.where(apos, np.abs(q) * h / (2 * aa), np.inf)
    th = np.clip(2.0 * (1.0 - Pe), 0.0, 1.0)          # 1 central (Pe<=1/2) ... 0 Godunov (Pe>=1)
    ramp = (Pe > 0.5) & (Pe < 1.0)
    p0 = -beta / b                                      # argmin of the convex H
    A = np.maximum(Dm - p0, 0.0) ** 2
    B = np.minimum(Dp - p0, 0.0) ** 2
    useA = (A >= B) & (A > 0)
    useB = (~useA) & (B > 0)
    Dg = np.where(useA, Dm, np.where(useB, Dp, p0))
    Hg = 0.5 * b * Dg ** 2 + beta * Dg
    H = th * Hc + (1 - th) * Hg
    k = 2 if mode == 'sym' else 3
    R = np.empty(n + k)
    R[:n] = lam - a * D2 + H - V - Vx
    R[n] = np.dot(w, m) - M
    R[n + 1] = u[G.iref]
    if mode == 'xbar':
        R[n + 2] = np.dot(w, x * m) - xbar * M
    elif mode == 'param':
        R[n + 2] = np.dot(w, x * m) - xbar * M
    if store is not None:
        store.update(m=m, a=a, b=b, beta=beta, th=th, xbar=xbar, u=u.copy())
    if not jac:
        return R
    gD = b * Dg + beta
    dH_dDc = th * q
    dH_dDp = (1 - th) * np.where(useB, gD, 0.0)
    dH_dDm = (1 - th) * np.where(useA, gD, 0.0)
    dH_db = th * 0.5 * Dc ** 2 + (1 - th) * 0.5 * Dg ** 2
    dH_dbeta = th * Dc + (1 - th) * Dg
    dth = np.where(ramp, -2.0, 0.0)
    k_q = (Hc - Hg) * dth * np.sign(q) * h / (2 * aa)
    k_a = (Hc - Hg) * dth * np.where(ramp, -Pe / aa, 0.0)
    dH_dDc = dH_dDc + k_q * b
    dH_db = dH_db + k_q * Dc
    dH_dbeta = dH_dbeta + k_q
    diag = 2 * a / h ** 2 - dH_dDp / h + dH_dDm / h - D2 * a_u + k_a * a_u + dH_db * b_m * m_u
    cup = -a / h ** 2 + dH_dDc / (2 * h) + dH_dDp / h
    cum = -a / h ** 2 - dH_dDc / (2 * h) - dH_dDm / h
    lo = cum[1:].copy()
    upd = cup[:-1].copy()
    upd[0] = cup[0] + cum[0]
    lo[-1] = cup[-1] + cum[-1]
    Bm = np.zeros((n, k))
    Cm = np.zeros((k, n))
    Dm_ = np.zeros((k, k))
    Bm[:, 0] = 1.0
    Bm[:, 1] = -D2 * a_mu + k_a * a_mu + dH_db * b_m * m_mu
    Cm[0] = w * m_u
    Dm_[0, 1] = np.dot(w, m_mu)
    Cm[1, G.iref] = 1.0
    if mode == 'xbar':
        Bm[:, 2] = dH_db * b_m * m_xb + dH_dbeta * beta_xb - Vx_xb
        Dm_[0, 2] = np.dot(w, m_xb)
        Cm[2] = w * x * m_u
        Dm_[2, 1] = np.dot(w, x * m_mu)
        Dm_[2, 2] = np.dot(w, x * m_xb) - M
    elif mode == 'param':
        Cm[2] = w * x * m_u
        Dm_[2, 1] = np.dot(w, x * m_mu)
        p = X[n + 2]
        dp = 1e-6 * max(abs(p), 1e-3)
        Xa = X.copy(); Xa[n + 2] = p + dp
        Xb = X.copy(); Xb[n + 2] = p - dp
        col = (residual(G, dial, V, M, Xa, mode, jac=False) - residual(G, dial, V, M, Xb, mode, jac=False)) / (2 * dp)
        dial.val = p
        Bm[:, 2] = col[:n]
        Dm_[:, 2] = col[n:]
    return R, (lo, diag, upd, Bm, Cm, Dm_)


def bordered_solve(Jp, rhs, iref):
    """Solve [[T, B],[C, D]] z = rhs, T tridiagonal (possibly singular on constants).
    Augment with s = z_iref:  (T + e e^T) z_u - e s + B z_y = r1 ;  C z_u + D z_y = r2 ;  e^T z_u - s = 0."""
    lo, di, up, B, C, D = Jp
    n, k = di.size, D.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = up
    ab[1] = di
    ab[1, iref] += 1.0
    ab[2, :-1] = lo
    e = np.zeros(n); e[iref] = 1.0
    Bt = np.column_stack([B, -e])
    Ct = np.vstack([C, e])
    Dt = np.zeros((k + 1, k + 1)); Dt[:k, :k] = D; Dt[k, k] = -1.0
    r1, r2 = rhs[:n], np.concatenate([rhs[n:], [0.0]])
    Z = sla.solve_banded((1, 1), ab, np.column_stack([r1, Bt]), check_finite=False)
    X1, Y = Z[:, 0], Z[:, 1:]
    zt = np.linalg.solve(Dt - Ct @ Y, r2 - Ct @ X1)
    zu = X1 - Y @ zt
    return np.concatenate([zu, zt[:k]])


def jac_matvec(Jp, z):
    lo, di, up, B, C, D = Jp
    n = di.size
    zu, zy = z[:n], z[n:]
    t = di * zu
    t[:-1] += up * zu[1:]
    t[1:] += lo * zu[:-1]
    return np.concatenate([t + B @ zy, C @ zu + D @ zy])


def jac_sparse(Jp):
    lo, di, up, B, C, D = Jp
    T = sp.diags([lo, di, up], [-1, 0, 1], format='csc')
    return sp.bmat([[T, sp.csc_matrix(B)], [sp.csc_matrix(C), sp.csc_matrix(D)]], format='csc')


def lin_solve(Jp, rhs, iref):
    with np.errstate(all='ignore'):
        try:
            z = bordered_solve(Jp, rhs, iref)
            ok = np.all(np.isfinite(z)) and np.linalg.norm(jac_matvec(Jp, z) - rhs) <= 1e-8 * max(1.0, np.linalg.norm(rhs))
        except (np.linalg.LinAlgError, ValueError):
            ok = False
    if not ok:
        z = spla.spsolve(jac_sparse(Jp), rhs)
    return z


def newton(G, dial, V, M, X0, mode='sym', maxit=80, tol=None, extra=0):
    X = X0.copy()
    out = residual(G, dial, V, M, X, mode)
    if out is None:
        return None, {'ok': False, 'why': 'invalid start'}
    R, Jp = out
    n = G.N + 1
    for it in range(maxit):
        amax = max(np.max(np.abs(dial.diffusion(X[:n], X[n + 1])[0])), NU)
        floor = 20 * MACH * amax * max(1.0, np.max(np.abs(X[:n]))) / G.h ** 2
        tl = max(1e-10, floor) if tol is None else max(tol, floor)
        ninf = np.max(np.abs(R))
        if ninf < tl and abs(R[n]) < 1e-12:
            for _ in range(extra):          # optional polish to the rounding floor
                Xe = X + lin_solve(Jp, -R, G.iref)
                Re = residual(G, dial, V, M, Xe, mode)
                if Re is None or np.linalg.norm(Re[0]) > 10 * np.linalg.norm(R):
                    break
                X, (R, Jp) = Xe, Re
            return X, _done(R, n, it, np.max(np.abs(R)))
        dX = lin_solve(Jp, -R, G.iref)
        if not np.all(np.isfinite(dX)):
            return None, {'ok': False, 'why': 'singular'}
        nr = np.linalg.norm(R)
        al, acc = 1.0, False
        while al > 1e-9:
            Xn = X + al * dX
            with np.errstate(all='ignore'):
                Rn = residual(G, dial, V, M, Xn, mode, jac=False)
            if Rn is not None and np.all(np.isfinite(Rn)) and np.linalg.norm(Rn) < (1 - 1e-4 * al) * nr:
                acc = True
                break
            al *= 0.5
        if not acc:
            if ninf < 100 * tl and abs(R[n]) < 1e-10:
                return X, _done(R, n, it, ninf, stalled=True)
            return None, {'ok': False, 'why': 'linesearch', 'res': float(ninf)}
        X = Xn
        R, Jp = residual(G, dial, V, M, X, mode)
    ninf = np.max(np.abs(R))
    if ninf < 1e-7:
        return X, _done(R, n, maxit, ninf, stalled=True)
    return None, {'ok': False, 'why': 'maxit', 'res': float(ninf)}


def _done(R, n, it, ninf, stalled=False):
    STATS['n_solves'] += 1
    STATS['mass_err_max'] = max(STATS['mass_err_max'], abs(R[n]))
    return {'ok': True, 'it': it, 'res': float(ninf), 'mass_err': float(abs(R[n])), 'stalled': stalled}


def follow(G, make, vals, V, M, X0, mode='sym', min_step=1e-7):
    """Natural-parameter continuation with adaptive sub-steps.  Returns list of (val, X) and the
    last value reached (== vals[-1] unless the branch was lost)."""
    res, X, cur = [], X0.copy(), vals[0]
    for tgt in vals:
        step = tgt - cur
        while True:
            nxt = tgt if abs(tgt - cur) <= abs(step) + 1e-15 else cur + step
            Xn, info = newton(G, make(nxt), V, M, X, mode)
            if Xn is None:
                step /= 2
                if abs(step) < min_step:
                    return res, cur
                continue
            X, cur = Xn, nxt
            if nxt == tgt:
                break
            step = min(abs(step) * 2, abs(tgt - cur)) * np.sign(tgt - cur)
        res.append((tgt, X.copy()))
    return res, cur


def state(G, dial, V, M, X, mode='sym'):
    st = {}
    residual(G, dial, V, M, X, mode, jac=False, store=st)
    return st


def d0_solution(G, V=None, M=M1):
    V = Vquart(G.x) if V is None else V
    u = 0.5 * G.x ** 2
    X = np.concatenate([u - u[G.iref], [0.3, 0.0]])
    X, info = newton(G, Dial('D0'), V, M, X)
    return X


# ----------------------------------------------------------------------------- small utilities
def nearest(dct, v):
    return dct[min(dct, key=lambda k: abs(k - v))]


def crossings(x, f):
    s = np.sign(f)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return np.array([x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i]) for i in idx])


def crossings_cubic(x, f):
    """zero crossings of a smooth sampled f: local cubic through 4 nodes, root in the bracketing cell."""
    out = []
    for i in np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
        j = min(max(i - 1, 0), len(x) - 4)
        c = np.polyfit(x[j:j + 4] - x[i], f[j:j + 4], 3)
        out.append(x[i] + brentq(lambda t: np.polyval(c, t), 0.0, x[i + 1] - x[i]))
    return np.array(out)


def interp_peak(m):
    k = int(np.argmax(m))
    a, b, c = m[k - 1], m[k], m[k + 1]
    den = a - 2 * b + c
    return b - (c - a) ** 2 / (8 * den) if den < 0 else b


def width_above(x, f):
    """measure of {f > 0}; crossings by local cubic interpolation."""
    cr = crossings_cubic(x, f)
    pts = np.concatenate([[x[0]] if f[0] > 0 else [], cr, [x[-1]] if f[-1] > 0 else []])
    return float(np.sum(pts[1::2] - pts[0::2])), cr


def pad(a, k, fill=np.nan):
    out = np.full(k, fill)
    out[:min(k, len(a))] = a[:k]
    return out


def obs_order(q):
    d1, d2 = abs(q[0] - q[1]), abs(q[1] - q[2])
    return float(np.log2(d1 / d2)) if d2 > 0 and d1 > 0 else float('nan')


def sig3_stable(q):
    q = np.asarray(q, float)
    e = math.floor(math.log10(abs(q[-1])))
    return bool(np.max(np.abs(q - q[-1])) <= 0.5 * 10 ** (e - 2))


def face_velocity(G, dial, st, X, mode='sym'):
    n = G.N + 1
    u, m = X[:n], st['m']
    mf = 0.5 * (m[1:] + m[:-1])
    xf = 0.5 * (G.x[1:] + G.x[:-1])
    p = np.diff(u) / G.h
    beta = dial.beta(xf, st['xbar'])[0]
    return -(dial.bvel(mf) * p + beta), mf


# ----------------------------------------------------------------------------- Hopf-Cole reference (C1)
def hopf_cole(V_fun, L_, nu, K=400, Q=2400):
    """ground state of -2 nu^2 phi'' + V phi = lambda phi, Neumann, cosine-Galerkin (spectral)."""
    xq, wq = np.polynomial.legendre.leggauss(Q)
    xq, wq = xq * L_, wq * L_
    k = np.arange(K)
    nrm = np.where(k == 0, np.sqrt(1 / (2 * L_)), np.sqrt(1 / L_))
    P = nrm[:, None] * np.cos(np.outer(k, xq + L_) * np.pi / (2 * L_))
    A = (P * (wq * V_fun(xq))) @ P.T + np.diag(2 * nu ** 2 * (k * np.pi / (2 * L_)) ** 2)
    ev, vec = np.linalg.eigh(A)
    c = vec[:, 0]

    def phi(x):
        return c @ (nrm[:, None] * np.cos(np.outer(k, x + L_) * np.pi / (2 * L_)))
    return ev[0], phi, ev[1]


# ----------------------------------------------------------------------------- forward-equation solver (C3)
def fp_solve(G, vf, nu, gamma, M, symmetric):
    """Stationary FV forward equation  d/dx[ Phi(m) ]' - (v m)' = 0, Phi = nu m^gamma, zero flux on every face:
    -(Phi(m_{i+1})-Phi(m_i))/h + v_f (m_i+m_{i+1})/2 = 0   (central; upwind where the face Peclet > 1).
    Solved face by face from the density peak (where v changes sign + -> -), mass by scaling (linear)
    or shooting (nonlinear).  symmetric=True: right half with mass M/2, mirrored (fixes the island masses
    of a disconnected support by the symmetry of the problem)."""
    N, h = G.N, G.h
    lin = (gamma == 1.0)
    i0 = N // 2 if symmetric else 0
    cand = [i for i in range(max(i0, 1), N) if vf[i - 1] > 0 >= vf[i]]
    if vf[N - 1] > 0:
        cand.append(N)
    p = max(cand)
    wts = G.w.copy()
    if symmetric:
        wts[i0] = h / 2

    def Phi(y):
        return nu * y if lin else nu * y ** gamma

    def dPhi(y):
        return nu if lin else nu * gamma * y ** (gamma - 1)

    def solve_face(known, v, right):
        # right: unknown is m_{f+1}: Phi(y) - hv/2 y = Phi(k) + hv/2 k ; left: Phi(y) + hv/2 y = Phi(k) - hv/2 k
        s = -h * v / 2 if right else h * v / 2
        if lin:
            if abs(v) * h / (2 * nu) < 1.0:
                return known * (nu - s) / (nu + s)
            vp, vm = max(v, 0.0), min(v, 0.0)
            return known * ((nu + h * vp) / (nu - h * vm) if right else (nu - h * vm) / (nu + h * vp))
        rhs = Phi(known) - s * known
        if rhs <= 0.0 or known == 0.0:
            return 0.0
        hi = min((rhs / nu) ** (1 / gamma), rhs / s) if s > 0 else (rhs / nu) ** (1 / gamma)
        lo_, y = 0.0, hi
        for _ in range(200):
            g = Phi(y) + s * y - rhs
            if abs(g) <= 1e-15 * rhs:
                break
            if g > 0:
                hi = y
            else:
                lo_ = y
            yn = y - g / (dPhi(y) + s) if y > 0 else 0.5 * (lo_ + hi)
            y = yn if lo_ < yn < hi else 0.5 * (lo_ + hi)
            if hi - lo_ <= 1e-16 * hi:
                break
        return y

    def march(mp):
        m = np.zeros(N + 1)
        m[p] = mp
        for f in range(p, N):
            m[f + 1] = solve_face(m[f], vf[f], True)
        for f in range(p - 1, i0 - 1, -1):
            m[f] = solve_face(m[f + 1], vf[f], False)
        return m

    target = M / 2 if symmetric else M
    if lin:
        m = march(1.0)
        m *= target / np.dot(wts[i0:], m[i0:])
    else:
        f = lambda lmp: math.log(np.dot(wts[i0:], march(math.exp(lmp))[i0:]) / target)
        lmp = brentq(f, -30.0, 10.0, xtol=1e-14, rtol=1e-15)
        m = march(math.exp(lmp))
    if symmetric:
        m[:i0] = m[N:i0:-1]
    return m


# ----------------------------------------------------------------------------- planner (C8)
def J_of_v(v, G, V, c, eps, M=M1, nu=NU, grad=False):
    """social cost per agent for face velocities v: m_i ∝ exp(phi_i), phi_{i+1}-phi_i = h v_f / nu."""
    h = G.h
    phi = np.concatenate([[0.0], np.cumsum(v) * h / nu])
    e = np.exp(phi - phi.max())
    m = M * e / np.dot(G.w, e)
    mf = 0.5 * (m[1:] + m[:-1])
    k = (eps + mf) ** c * mf
    J = (np.sum(h * k * v ** 2 / 2) + np.dot(G.w, V * m)) / M
    if not grad:
        return J
    kp = (eps + mf) ** c + c * mf * (eps + mf) ** (c - 1)
    t = h * kp * v ** 2 / 4
    gm = G.w * V
    gm[:-1] += t
    gm[1:] += t
    gm /= M
    gphi = m * gm - (G.w * m / M) * np.dot(gm, m)
    rc = np.cumsum(gphi[::-1])[::-1]
    return J, h * k * v / M + (h / nu) * rc[1:]


def kkt_newton(phi, eta, G, V, c, eps, M=M1, nu=NU, iters=60, tol=1e-13):
    """Newton on the same discrete J, written in phi = ln m with the mass constraint by a multiplier
    (tridiagonal Hessian + one border).  Direct minimisation: no HJB anywhere."""
    h, n, s = G.h, G.N + 1, G.h / (2 * M)
    na, nb = -nu / h, nu / h
    for it in range(iters):
        m = np.exp(phi)
        mb = 0.5 * (m[1:] + m[:-1])
        v = nu * np.diff(phi) / h
        k = (eps + mb) ** c * mb
        k1 = (eps + mb) ** c + c * mb * (eps + mb) ** (c - 1)
        k2 = 2 * c * (eps + mb) ** (c - 1) + c * (c - 1) * mb * (eps + mb) ** (c - 2)
        aa, ab = m[:-1] / 2, m[1:] / 2
        g = G.w * V * m / M
        g[:-1] += s * (k1 * aa * v ** 2 + 2 * k * v * na)
        g[1:] += s * (k1 * ab * v ** 2 + 2 * k * v * nb)
        cg = G.w * m
        rg, rc = g + eta * cg, np.dot(G.w, m) - M
        if np.abs(rg).max() < tol and abs(rc) < 1e-14:
            break
        Haa = s * (k2 * aa * aa * v ** 2 + k1 * aa * v ** 2 + 4 * k1 * aa * v * na + 2 * k * na * na)
        Hbb = s * (k2 * ab * ab * v ** 2 + k1 * ab * v ** 2 + 4 * k1 * ab * v * nb + 2 * k * nb * nb)
        Hab = s * (k2 * aa * ab * v ** 2 + 2 * k1 * aa * v * nb + 2 * k1 * ab * v * na + 2 * k * na * nb)
        d = G.w * V * m / M + eta * G.w * m
        d[:-1] += Haa
        d[1:] += Hbb
        d = d + 1e-15 * np.abs(d).max()
        ab_ = np.zeros((3, n)); ab_[0, 1:] = Hab; ab_[1] = d; ab_[2, :-1] = Hab
        Z = sla.solve_banded((1, 1), ab_, np.column_stack([-rg, cg]), check_finite=False)
        de = (-rc - np.dot(cg, Z[:, 0])) / (-np.dot(cg, Z[:, 1]))
        dphi = Z[:, 0] - Z[:, 1] * de
        phi, eta = phi + dphi, eta + de
    return phi, eta, it


# ============================================================================= tasks
def task_baseline(out, num):
    num['setting'] = {'L_half_width': L, 'nu': NU, 'barrier_h': HB, 'mass_M': M1, 'congestion_eps': EPS,
                      'u_normalisation': 'u(x=1) = 0', 'grid_production_N': NP, 'grid_ladder_N': [N0, 2 * N0, 4 * N0],
                      'grid_planner_N': N_PLAN, 'parameter_changed': 'none'}
    base = {}
    for N in (N0, 2 * N0, 4 * N0):
        G = Grid(N)
        X = d0_solution(G)
        st = state(G, Dial('D0'), Vquart(G.x), M1, X)
        base[N] = (G, X, interp_peak(st['m']), st['m'])
    out['_base'] = base
    G, X, pk, m = base[NP]
    num['linear_game'] = {'lambda': float(X[NP + 1]), 'peak_density_interp': float(pk),
                          'peak_location_x': float(G.x[np.argmax(m)]), 'lambda_below_barrier_h': bool(X[NP + 1] < HB)}


def task_frontier(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    gam = np.linspace(1.0, 2.5, 61)
    res, last = follow(G, lambda g: Dial('D1', g), gam, V, M1, X0)
    assert last == gam[-1], f'frontier lost at {last}'
    mm, uu, lam, edges, nedge = [], [], [], [], []
    for g, X in res:
        st = state(G, Dial('D1', g), V, M1, X)
        mm.append(st['m']); uu.append(X[:G.N + 1]); lam.append(X[G.N + 1])
        if g > 1:
            base = 1 + (g - 1) * (X[G.N + 2] - X[:G.N + 1]) / (NU * g)
            e = crossings(G.x, base)
        else:
            e = np.array([])
        edges.append(pad(e, 4)); nedge.append(len(e))
    b0 = np.array([1 + (g - 1) * (X[G.N + 2] - X[G.N // 2]) / (NU * g) for g, X in res])
    bL = np.array([1 + (g - 1) * (X[G.N + 2] - X[G.N]) / (NU * g) for g, X in res])
    g_gap = float(crossings(gam, b0)[0])
    g_wall = float(crossings(gam, bL)[0])
    out.update(frontier_gamma=gam, frontier_m=np.array(mm), frontier_u=np.array(uu), frontier_lambda=np.array(lam),
               frontier_mu=np.array([X[G.N + 2] for g, X in res]),
               frontier_edges=np.array(edges), frontier_n_edges=np.array(nedge))
    out['_frontier_X'] = dict(res)
    E = np.array(edges)
    i15, i20, i25 = [int(np.argmin(abs(gam - v))) for v in (1.5, 2.0, 2.5)]
    first2 = gam[np.argmax(np.array(nedge) >= 2)]
    first4 = gam[np.argmax(np.array(nedge) >= 4)]
    num['frontier_D1_gamma_ge_1'] = {
        'gamma_range': [1.0, 2.5], 'n_rows': len(gam),
        'first_gamma_with_edges_in_domain': float(first2), 'first_gamma_with_empty_gap_4_edges': float(first4),
        'gamma_support_leaves_walls_interp': g_wall, 'gamma_gap_opens_at_x0_interp': g_gap,
        'edges_gamma_1p5': E[i15].tolist(), 'edges_gamma_2': E[i20].tolist(), 'edges_gamma_2p5': E[i25].tolist(),
        'lambda_gamma_1p5': float(lam[i15]), 'lambda_gamma_2': float(lam[i20]), 'lambda_gamma_2p5': float(lam[i25]),
        'peak_density_gamma_2': float(np.max(mm[i20])), 'island_width_gamma_2': float(E[i20][3] - E[i20][2]),
        'gap_half_width_gamma_2': float(E[i20][2]),
        'lambda_max_over_map': float(np.max(lam)), 'lambda_stays_below_h': bool(np.max(lam) < HB)}


def task_tails(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    # lower end: natural continuation, fine steps, on all three grids
    low = {}
    for N in (N0, 2 * N0, 4 * N0):
        Gk, Xk = out['_base'][N][0], out['_base'][N][1]
        vals = np.concatenate([np.round(np.arange(1, 0.3, -0.02), 6), np.round(np.arange(0.3, 0.2, -0.0025), 6)])
        res, last = follow(Gk, lambda g: Dial('D1', g), vals, Vquart(Gk.x), M1, Xk, min_step=1e-6)
        low[N] = (float(last), float(res[-1][1][Gk.N + 1]))
    # lambda-parameterised continuation past the fold (N = 2N0): gamma becomes the unknown
    Gk, Xk = out['_base'][2 * N0][0], out['_base'][2 * N0][1]
    res, last = follow(Gk, lambda g: Dial('D1', g), np.round(np.arange(1, 0.2199, -0.01), 4), Vquart(Gk.x), M1, Xk)
    s_curve = lambda_param_curve(Gk, res[-1][1], res[-1][0], Vquart(Gk.x))
    gmin = 0.225
    gam = np.linspace(1.0, gmin, 61)
    res, last = follow(G, lambda g: Dial('D1', g), gam, V, M1, X0)
    assert abs(last - gmin) < 1e-12
    mm, uu, lam, pexp, r2 = [], [], [], [], []
    xw = (1.5, 2.0)
    sel = (G.x >= xw[0] - 1e-12) & (G.x <= xw[1] + 1e-12)
    for g, X in res:
        st = state(G, Dial('D1', g), V, M1, X)
        m = st['m']
        mm.append(m); uu.append(X[:G.N + 1]); lam.append(X[G.N + 1])
        A = np.vstack([np.log(G.x[sel]), np.ones(sel.sum())]).T
        coef, rss, *_ = np.linalg.lstsq(A, np.log(m[sel]), rcond=None)
        y = np.log(m[sel])
        r2.append(1 - float(rss[0]) / float(np.sum((y - y.mean()) ** 2)))
        pexp.append(-coef[0])
    mm = np.array(mm)
    out.update(tails_gamma=gam, tails_m=mm, tails_log10m=np.log10(np.maximum(mm, 1e-300)), tails_u=np.array(uu),
               tails_lambda=np.array(lam), tails_exponent=np.array(pexp), tails_fit_r2=np.array(r2),
               tails_fit_window=np.array(xw), tails_scurve=np.array(s_curve),
               tails_mu=np.array([X[G.N + 2] for g, X in res]))
    out['_tails_X'] = dict(res)
    pick = lambda v: int(np.argmin(abs(gam - v)))
    num['tails_D1_gamma_le_1'] = {
        'gamma_range_map': [gmin, 1.0], 'n_rows': len(gam),
        'fit': 'least squares ln m = const - p ln x on the right flank',
        'fit_window_x': list(xw),
        'exponent_p_at_gamma': {f'{v:g}': float(pexp[pick(v)]) for v in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, gmin)},
        'fit_r2_min': float(np.min(r2)),
        'asymptotic_prediction_3_over_1_minus_gamma_at_0p6': 3 / (1 - 0.6),
        'log10_m_at_wall_x2p5': {f'{v:g}': float(np.log10(mm[pick(v)][-1])) for v in (1.0, 0.8, 0.6, 0.4, gmin)},
        'density_at_barrier_x0': {f'{v:g}': float(mm[pick(v)][G.N // 2]) for v in (1.0, 0.8, 0.6, 0.4, gmin)},
        'lambda_at_gamma': {f'{v:g}': float(lam[pick(v)]) for v in (1.0, 0.6, 0.4, gmin)},
        'lower_end': {
            'last_converged_gamma_natural_continuation': {str(N): low[N][0] for N in low},
            'lambda_there': {str(N): low[N][1] for N in low},
            'interpretation': 'fold of the solution branch (same gamma on every grid); not a grid effect',
            'lambda_continuation_N800_gamma_lambda_m0_argmaxabsx': [[float(v) for v in row] for row in s_curve],
            'gamma_min_on_lambda_curve': float(min(r[0] for r in s_curve)),
            'gamma_max_on_lambda_curve_after_first_fold': float(max(r[0] for r in s_curve if r[1] > 0.95)),
            'past_fold': 'for lambda > h the crowd sits on the barrier top (m(0) grows, argmax at x = 0) while gamma falls; the lambda-continuation stops where m^(gamma-1) -> 0 at x = 0 (density blow-up)',
            'map_gamma_min_reason': 'gamma_min = 0.225 sits just above the multi-valued window [0.2194, 0.2213]',
            'u_x3_tail_estimate_4_over_7': 4 / 7,
            'converges_below_4_over_7': True}}


def lambda_param_curve(G, X, g0, V, dlam=0.01, nmax=40):
    """continue the D1 branch with lambda prescribed and gamma unknown (passes the fold)."""
    n = G.N + 1
    d = Dial('D1', g0)

    def F(Y, lamf):
        Xf = Y.copy(); Xf[n] = lamf; d.val = Y[n]
        return residual(G, d, V, M1, Xf, 'sym', jac=False)

    Y = X.copy(); Y[n] = g0
    lam = X[n]
    def extras(Yc, lamc):
        Xf = Yc.copy(); Xf[n] = lamc; d.val = Yc[n]
        mm_ = state(G, d, V, M1, Xf)['m']
        return float(mm_[G.N // 2]), float(abs(G.x[np.argmax(mm_)]))

    curve = [(g0, lam) + extras(Y, lam)]
    for _ in range(nmax):
        lam_t = lam + dlam
        Yt = Y.copy()
        ok = False
        for it in range(60):
            R = F(Yt, lam_t)
            if R is None:
                break
            if np.abs(R).max() < 1e-10:
                ok = True
                break
            Xf = Yt.copy(); Xf[n] = lam_t; d.val = Yt[n]
            _, Jp = residual(G, d, V, M1, Xf, 'sym')
            Js = jac_sparse(Jp).tolil()
            dg = 1e-7
            Yp, Ym = Yt.copy(), Yt.copy(); Yp[n] += dg; Ym[n] -= dg
            Js[:, n] = ((F(Yp, lam_t) - F(Ym, lam_t)) / (2 * dg)).reshape(-1, 1)
            dY = spla.spsolve(Js.tocsc(), -R)
            al = 1.0
            while al > 1e-6:
                Rn = F(Yt + al * dY, lam_t)
                if Rn is not None and np.linalg.norm(Rn) < np.linalg.norm(R):
                    break
                al /= 2
            Yt = Yt + al * dY
        if not ok:
            break
        Y, lam = Yt, lam_t
        curve.append((Y[n], lam) + extras(Y, lam))
    return curve


def task_capacity(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    ys = np.linspace(0, 3, 61)
    res, last = follow(G, lambda y: Dial('D2', y, peak0=pk), ys, V, M1, X0)
    assert last == ys[-1]
    mm, uu, lam, wid, pe, mk = [], [], [], [], [], []
    for y, X in res:
        st = state(G, Dial('D2', y, peak0=pk), V, M1, X)
        m = st['m']
        mm.append(m); uu.append(X[:G.N + 1]); lam.append(X[G.N + 1])
        if y > 0:
            K = pk / y
            wd, cr = width_above(G.x, m - 0.95 * K)
            mk.append(np.max(m) / K)
        else:
            wd, cr = 0.0, np.array([])
            mk.append(0.0)
        wid.append(wd); pe.append(pad(cr, 4))
    mk_arr = np.array(mk)
    y_on = float(crossings(ys[1:], mk_arr[1:] - 0.95)[0])
    out.update(capacity_mu=np.array([X[G.N + 2] for y, X in res]), capacity_y=ys, capacity_K=np.where(ys > 0, pk / np.maximum(ys, 1e-300), np.inf), capacity_m=np.array(mm),
               capacity_u=np.array(uu), capacity_lambda=np.array(lam), capacity_plateau_width=np.array(wid),
               capacity_plateau_edges=np.array(pe), capacity_max_m_over_K=np.array(mk), linear_peak=pk)
    out['_capacity_X'] = dict(res)
    pick = lambda v: int(np.argmin(abs(ys - v)))
    w = np.array(wid)
    num['capacity_D2'] = {
        'dial': 'y = linear peak / K', 'linear_peak_m': float(pk), 'y_range': [0, 3], 'n_rows': len(ys),
        'plateau_definition': 'total length where m > 0.95 K (both islands)',
        'first_y_with_plateau': float(ys[np.argmax(w > 0)]), 'y_plateau_appears_interp_maxm_eq_0p95K': y_on,
        'plateau_width_at_y': {f'{v:g}': float(w[pick(v)]) for v in (1, 1.5, 2, 2.5, 3)},
        'plateau_edges_y2': pe[pick(2)].tolist(),
        'max_m_over_K': float(np.max(mk)),
        'lambda_at_y': {f'{v:g}': float(lam[pick(v)]) for v in (0, 1, 2, 3)}}


def cong_edges(x, m, frac=1e-6):
    lm = np.log(np.maximum(m, 1e-300))
    return crossings_cubic(x, lm - np.log(frac * np.max(m)))


def task_congestion(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    cs = np.linspace(0, 1, 61)
    res, last = follow(G, lambda c: Dial('D3', c), cs, V, M1, X0)
    assert last == cs[-1]
    mm, uu, lam, ed, ne = [], [], [], [], []
    for c, X in res:
        st = state(G, Dial('D3', c), V, M1, X)
        m = st['m']
        e = cong_edges(G.x, m)
        mm.append(m); uu.append(X[:G.N + 1]); lam.append(X[G.N + 1]); ed.append(pad(e, 6)); ne.append(len(e))
    r0 = np.array([np.log(max(m[G.N // 2], 1e-300) / np.max(m)) - np.log(1e-6) for m in mm])
    c_gap = float(crossings(cs, r0)[0])
    out.update(congestion_mu=np.array([X[G.N + 2] for c, X in res]), congestion_c=cs, congestion_m=np.array(mm), congestion_u=np.array(uu), congestion_lambda=np.array(lam),
               congestion_edges=np.array(ed), congestion_n_edges=np.array(ne), congestion_eps=EPS)
    out['_congestion_X'] = dict(res)
    pick = lambda v: int(np.argmin(abs(cs - v)))
    # epsilon sweep at c = 0.5
    eps_sweep = {}
    for eps in (1e-2, 1e-3, 1e-4):
        r, lst = follow(G, lambda c: Dial('D3', c, eps=eps), np.linspace(0, 0.5, 11), V, M1, X0)
        X = r[-1][1]
        st = state(G, Dial('D3', 0.5, eps=eps), V, M1, X)
        e = cong_edges(G.x, st['m'])
        eps_sweep[f'{eps:g}'] = {'edges': e.tolist(), 'lambda': float(X[G.N + 1]), 'peak': float(np.max(st['m']))}
    # eps -> 0 limit edges at c = 0.5 for reference: m = [1 + c (mu-u)/nu]_+^(1/c) evaluated on the eps=1e-4 u
    num['congestion_D3'] = {
        'eps': EPS, 'c_range': [0, 1], 'n_rows': len(cs), 'edge_definition': 'm = 1e-6 max m (local cubic interpolation of ln m)',
        'edges_at_c': {f'{v:g}': ed[pick(v)][:ne[pick(v)]].tolist() for v in (0, 0.25, 0.5, 0.75, 1)},
        'n_edges_at_c': {f'{v:g}': int(ne[pick(v)]) for v in (0, 0.25, 0.5, 0.75, 1)},
        'first_c_with_4_edges': float(cs[np.argmax(np.array(ne) >= 4)]) if np.any(np.array(ne) >= 4) else None,
        'c_gap_opens_interp_m0_eq_1e-6max': c_gap,
        'lambda_at_c': {f'{v:g}': float(lam[pick(v)]) for v in (0, 0.5, 1)},
        'peak_at_c': {f'{v:g}': float(np.max(mm[pick(v)])) for v in (0, 0.5, 1)},
        'eps_sweep_c0p5': eps_sweep}


def herd_threshold_lin(G, kind, X0, lo, hi):
    """linearised self-consistency: d xbar_out / d xbar_in = 1 at xbar = 0 (implicit-function derivative)."""
    V = Vquart(G.x)
    n = G.N + 1
    cache = {'X': X0.copy()}

    def slope(val):
        d = Dial(kind, val, xbar=0.0)
        X, info = newton(G, d, V, M1, cache['X'])
        cache['X'] = X
        R, Jp = residual(G, d, V, M1, X, 'sym')
        dl = 1e-6
        d.xbar = dl; Rp = residual(G, d, V, M1, X, 'sym', jac=False)
        d.xbar = -dl; Rm = residual(G, d, V, M1, X, 'sym', jac=False)
        d.xbar = 0.0
        dX = lin_solve(Jp, -(Rp - Rm) / (2 * dl), G.iref)
        m, m_u, m_mu, m_xb = d.density(X[:n], X[n + 1], G.x, 0.0)
        return np.dot(G.w, G.x * (m_u * dX[:n] + m_mu * dX[n + 1] + m_xb)) / M1

    return brentq(lambda t: slope(t) - 1.0, lo, hi, xtol=1e-15, rtol=1e-13), slope


def herd_branch(G, kind, crit, X0):
    """herded branch parameterised by s = xbar (dial value unknown).  Returns s, dial(s), solutions."""
    V = Vquart(G.x)
    n = G.N + 1
    start = 2.0 * crit
    d = Dial(kind, start, xbar=0.9)
    Xs, _ = newton(G, d, V, M1, X0)                          # xbar_in held at 0.9
    Xh, info = newton(G, Dial(kind, start), V, M1, np.concatenate([Xs, [0.9]]), mode='xbar')
    s0 = Xh[n + 2]
    svals = np.concatenate([np.linspace(s0, 0.3, 14)[1:], [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01]])
    Y = Xh.copy(); Y[n + 2] = start
    S, P, SOL = [s0], [start], [Xh.copy()]
    for s in svals:
        d = Dial(kind, Y[n + 2], xbar=s)
        Yn, info = newton(G, d, V, M1, Y, mode='param')
        if Yn is None:
            break
        Y = Yn
        S.append(s); P.append(Y[n + 2])
        Z = Y[:n + 2].copy()
        SOL.append(np.concatenate([Z, [s]]))
    S, P = np.array(S), np.array(P)
    sel = S <= 0.1 + 1e-12
    A = np.vstack([np.ones(sel.sum()), S[sel] ** 2, S[sel] ** 4]).T
    coef = np.linalg.lstsq(A, P[sel], rcond=None)[0]
    return S, P, SOL, float(coef[0]), coef


def task_herding(out, num):
    thc, kac, thl = {}, {}, {}
    for N in (N0, 2 * N0, 4 * N0):
        G, X0 = out['_base'][N][0], out['_base'][N][1]
        thc[N], _ = herd_threshold_lin(G, 'D4', X0, 1e-3, 0.05)
        kac[N], _ = herd_threshold_lin(G, 'D5', X0, 1e-6, 2.5e-3)
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    n = G.N + 1
    tc, kc = thc[NP], kac[NP]
    S4, P4, SOL4, tc_cont, coef4 = herd_branch(G, 'D4', tc, X0)
    S5, P5, SOL5, kc_cont, coef5 = herd_branch(G, 'D5', kc, X0)
    # map
    ths = np.linspace(0, 2 * tc, 61)
    mm, uu, lam, xb = [], [], [], []
    Xsym = X0.copy()
    sol_by_theta = {}
    for t in ths:
        if t <= tc * (1 + 1e-12):
            d = Dial('D4', t, xbar=0.0)
            Xsym, info = newton(G, d, V, M1, Xsym)
            X, xbar_v = Xsym, 0.0
            st = state(G, d, V, M1, X)
        else:
            j = int(np.argmin(np.abs(P4 - t)))
            Xg = SOL4[j].copy()
            X, info = newton(G, Dial('D4', t), V, M1, Xg, mode='xbar')
            assert X is not None and X[n + 2] > 1e-3, f'herded solve failed at theta={t}'
            xbar_v = X[n + 2]
            st = state(G, Dial('D4', t), V, M1, X, 'xbar')
        mm.append(st['m']); uu.append(X[:n]); lam.append(X[n]); xb.append(xbar_v)
        sol_by_theta[float(t)] = X.copy()
    xb = np.array(xb)
    i15 = int(np.argmin(np.abs(ths - 1.5 * tc)))
    out.update(herding_theta=ths, herding_m=np.array(mm), herding_u=np.array(uu), herding_lambda=np.array(lam),
               herding_xbar=xb, herding_theta_c=tc, herding_branch_s=S4, herding_branch_theta=P4,
               conformity_branch_s=S5, conformity_branch_kappa=P5)
    out['_herd'] = {'tc': tc, 'kc': kc, 'sol': sol_by_theta, 'thc': thc, 'kac': kac}
    # D5 at 1.5 kappa_c for completeness
    j = int(np.argmin(np.abs(P5 - 1.5 * kc)))
    X5, _ = newton(G, Dial('D5', 1.5 * kc), V, M1, SOL5[j], mode='xbar')
    out['_herd']['X5_15'] = X5
    num['herding_D4_D5'] = {
        'theta_c_linearised': float(tc), 'theta_c_continuation': tc_cont,
        'theta_c_rel_diff': abs(tc_cont - tc) / tc,
        'kappa_c_linearised': float(kc), 'kappa_c_continuation': kc_cont, 'kappa_c_rel_diff': abs(kc_cont - kc) / kc,
        'sqrt_kappa_c': float(np.sqrt(kc)), 'theta_c_vs_sqrt_kappa_c_rel_diff': abs(tc - np.sqrt(kc)) / tc,
        'xbar_at_1p5_theta_c': float(xb[i15]), 'theta_at_that_row': float(ths[i15]),
        'xbar_D5_at_1p5_kappa_c': float(X5[n + 2]),
        'xbar_at_2_theta_c': float(xb[-1]),
        'branch_fit': 'dial(s) = c0 + c1 s^2 + c2 s^4 on s = xbar <= 0.1',
        'branch_fit_coefs_D4': coef4.tolist(),
        'branch_supercritical': bool(coef4[1] > 0),
        'tunnelling_gap_lambda1_minus_lambda0': None,
        'theta_c_by_grid': {str(N): float(thc[N]) for N in thc},
        'kappa_c_by_grid': {str(N): float(kac[N]) for N in kac}}


def task_decoupling(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    n = G.N + 1
    h = out['_herd']
    cases = {
        'D0': (Dial('D0'), X0, 'sym'),
        'D1_gamma_1p75': (Dial('D1', 1.75), out['_frontier_X'][min(out['_frontier_X'], key=lambda g: abs(g - 1.75))], 'sym'),
        'D1_gamma_0p6': (Dial('D1', 0.6), out['_tails_X'][min(out['_tails_X'], key=lambda g: abs(g - 0.6))], 'sym'),
        'D2_y_1p5': (Dial('D2', 1.5, peak0=pk), nearest(out['_capacity_X'], 1.5), 'sym'),
        'D3_c_0p5': (Dial('D3', 0.5), nearest(out['_congestion_X'], 0.5), 'sym'),
        'D4_theta_c_symmetric': (Dial('D4', h['tc'], xbar=0.0), h['sol'][min(h['sol'], key=lambda t: abs(t - h['tc']))], 'sym'),
        'D4_1p5theta_c_herded': (Dial('D4', 1.5 * h['tc']), h['sol'][min(h['sol'], key=lambda t: abs(t - 1.5 * h['tc']))], 'xbar'),
        'D5_1p5kappa_c_herded': (Dial('D5', 1.5 * h['kc']), h['X5_15'], 'xbar'),
    }
    res = {}
    for name, (d, Xg, mode) in cases.items():
        X1, _ = newton(G, d, V, 1.0, Xg, mode=mode, extra=3)
        # mass continuation 1 -> 2 (the closure is nonlinear in the mass for D1-D3)
        Xc = X1.copy()
        for M in np.linspace(1.0, 2.0, 11)[1:]:
            Xc, info = newton(G, d, V, M, Xc, mode=mode, extra=3)
            assert Xc is not None, name
        res[name] = {'delta_lambda': float(Xc[n] - X1[n]), 'sup_abs_delta_u': float(np.max(np.abs(Xc[:n] - X1[:n]))),
                     'lambda_M1': float(X1[n]), 'lambda_M2': float(Xc[n])}
        if mode == 'xbar':
            res[name]['xbar_M1'] = float(X1[n + 2]); res[name]['xbar_M2'] = float(Xc[n + 2])
    num['decoupling_M1_to_M2'] = res


def task_mechanisms(out, num):
    G, X0, pk, _ = out['_base'][NP]
    V = Vquart(G.x)
    n = G.N + 1
    Xa = nearest(out['_frontier_X'], 1.5)
    Xb = nearest(out['_congestion_X'], 0.5)
    da, db = Dial('D1', 1.5), Dial('D3', 0.5)
    sa, sb = state(G, da, V, M1, Xa), state(G, db, V, M1, Xb)
    va, mfa = face_velocity(G, da, sa, Xa)
    vb, mfb = face_velocity(G, db, sb, Xb)
    crowd = np.maximum(mfa, mfb) >= 1e-3 * max(mfa.max(), mfb.max())
    supp = mfa > 0
    num['two_mechanisms_D1g1p5_vs_D3c0p5'] = {
        'sup_abs_m_noise_minus_m_cong': float(np.max(np.abs(sa['m'] - sb['m']))),
        'sup_rel_to_max_m': float(np.max(np.abs(sa['m'] - sb['m'])) / max(sa['m'].max(), sb['m'].max())),
        'lambda_noise_D1': float(Xa[n]), 'lambda_congestion_D3': float(Xb[n]),
        'sup_abs_v_diff_where_crowd_ge_1e-3_max': float(np.max(np.abs(va - vb)[crowd])),
        'sup_abs_v_diff_on_D1_support': float(np.max(np.abs(va - vb)[supp])),
        'sup_abs_v_noise_on_crowd': float(np.max(np.abs(va[crowd]))),
        'peak_noise': float(sa['m'].max()), 'peak_cong': float(sb['m'].max()),
        'q_exponent_both': 2.0,
        'statistics_scale': 'D1: m = [1 + (mu-u)/(3 nu)]_+^2 ; D3 (eps->0): m = [1 + (mu-u)/(2 nu)]_+^2'}
    out.update(mech_x=G.x, mech_m_noise=sa['m'], mech_m_cong=sb['m'], mech_v_noise=va, mech_v_cong=vb)


def task_planner(out, num):
    cs = (0.0, 0.25, 0.5, 0.75, 1.0)
    G0 = Grid(N0)
    V0 = Vquart(G0.x)
    Gp, Xp_map = out['_base'][NP][0], out['_congestion_X']
    rng = np.random.default_rng(20261008)
    res = {}
    vstore = {}
    for c in cs:
        t0 = time.time()
        # (a) planner: L-BFGS over face velocities on the coarse grid from v = 0, then Newton on the same J,
        #     cascading to the comparison grid.  No HJB.
        lb = minimize(J_of_v, np.zeros(N0), args=(G0, V0, c, EPS, M1, NU, True), jac=True, method='L-BFGS-B',
                      options={'maxiter': 60000, 'maxfun': 120000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxcor': 30})
        phi = np.concatenate([[0.0], np.cumsum(lb.x) * G0.h / NU])
        phi -= np.log(np.dot(G0.w, np.exp(phi)))
        eta, Gc = -0.25, G0
        kkt_its = []
        for N in (N0, 4 * N0, 16 * N0, N_PLAN):
            G = Grid(N)
            phi = np.interp(G.x, Gc.x, phi)
            phi -= np.log(np.dot(G.w, np.exp(phi)))
            phi, eta, it = kkt_newton(phi, eta, G, Vquart(G.x), c, EPS)
            kkt_its.append(it)
            Gc = G
        V = Vquart(G.x)
        v_mfc = NU * np.diff(phi) / G.h
        J_mfc = J_of_v(v_mfc, G, V, c, EPS)
        Jg, gnorm = J_of_v(v_mfc, G, V, c, EPS, grad=True)
        # (b) toll game and (c) plain game, by the HJB, cascaded from the production grid
        games = {}
        for toll in (False, True):
            d = Dial('D3', c, toll=toll)
            Xg = Xp_map[min(Xp_map, key=lambda cc: abs(cc - c))]
            X, info = newton(Gp, d, Vquart(Gp.x), M1, Xg)
            if X is None:
                r, lst = follow(Gp, lambda cc: Dial('D3', cc, toll=True), np.linspace(0, c, 21), Vquart(Gp.x), M1, out['_base'][NP][1])
                X = r[-1][1]
            Gc_ = Gp
            for N in (4 * NP, N_PLAN):
                Gf = Grid(N)
                X = np.concatenate([np.interp(Gf.x, Gc_.x, X[:Gc_.N + 1]), X[Gc_.N + 1:]])
                X, info = newton(Gf, d, Vquart(Gf.x), M1, X)
                assert X is not None, f'planner game c={c} toll={toll}'
                Gc_ = Gf
            st = state(Gf, d, Vquart(Gf.x), M1, X)
            v, mf = face_velocity(Gf, d, st, X)
            games[toll] = (J_of_v(v, Gf, V, c, EPS), v, mf, X[Gf.N + 1])
        J_toll, v_toll, mf_toll, lam_toll = games[True]
        J_mfg, v_mfg, mf_mfg, lam_mfg = games[False]
        bulk = mf_toll > 1e-3 * mf_toll.max()
        dv_sup = float(np.max(np.abs(v_mfc - v_toll)[bulk]) / np.max(np.abs(v_toll[bulk])))
        dv_l2m = float(np.sqrt(np.sum(G.h * mf_toll * (v_mfc - v_toll) ** 2) / np.sum(G.h * mf_toll * v_toll ** 2)))
        # random perturbations of the planner's velocity field
        xf = 0.5 * (G.x[1:] + G.x[:-1])
        vscale = np.max(np.abs(v_mfc[bulk]))
        lowered, min_inc = 0, np.inf
        for trial in range(100):
            kk = np.arange(1, 9)
            amp = rng.normal(size=8) / kk
            ph = rng.uniform(0, 2 * np.pi, 8)
            shape = (amp[:, None] * np.sin(np.outer(kk, (xf + L) * np.pi / (2 * L)) + ph[:, None])).sum(0)
            A = 10 ** rng.uniform(-3, -1) * vscale / np.max(np.abs(shape))
            dJ = J_of_v(v_mfc + A * shape, G, V, c, EPS) - J_mfc
            lowered += int(dJ < 0)
            min_inc = min(min_inc, dJ / J_mfc)
        res[f'{c:g}'] = {'J_MFC_direct': float(J_mfc), 'J_toll_game_HJB': float(J_toll), 'J_MFG_plain_game': float(J_mfg),
                         'rel_diff_J_toll_vs_MFC': float((J_toll - J_mfc) / J_mfc),
                         'gap_MFG_vs_MFC_rel': float((J_mfg - J_mfc) / J_mfc),
                         'v_rel_sup_diff_bulk_m_ge_1e-3max': dv_sup, 'v_rel_L2m_diff': dv_l2m,
                         'perturbations_lowering_J': lowered, 'perturbation_min_rel_increase': float(min_inc),
                         'planner_multiplier_eta': float(eta), 'lambda_toll_game': float(lam_toll), 'lambda_plain_game': float(lam_mfg),
                         'lbfgs_iterations_N400': int(lb.nit), 'newton_iterations_cascade': kkt_its,
                         'planner_grad_inf_norm': float(np.max(np.abs(gnorm))), 'grid_N': N_PLAN,
                         'seconds': time.time() - t0}
        vstore[c] = (v_mfc[::16], v_toll[::16], v_mfg[::16], mf_toll[::16])
    num['planner_D3'] = {'toll_formula': 'tau(x) = m dl/dm = c m (eps+m)^(c-1) v(x)^2 / 2  [= c (eps+m)^c v^2/2 * m/(eps+m); exact c(eps+m)^c v^2/2 only at eps = 0]',
                         'toll_charged_as': 'a fee per unit time at x, set at the equilibrium flow velocity v(x); it does not enter the agent first-order condition',
                         'toll_game_HJB': "lambda - nu u'' + H(u',m) + m dH/dm(u',m) = V,  H = p^2/(2(eps+m)^c)",
                         'by_c': res}
    xf = 0.5 * (Grid(N_PLAN).x[1:] + Grid(N_PLAN).x[:-1])[::16]
    out['planner_xf'] = xf
    for c in cs:
        a, b, g, mf = vstore[c]
        out[f'planner_c{int(round(100 * c)):03d}_v_mfc'] = a
        out[f'planner_c{int(round(100 * c)):03d}_v_toll'] = b
        out[f'planner_c{int(round(100 * c)):03d}_v_mfg'] = g
        out[f'planner_c{int(round(100 * c)):03d}_m'] = mf


# ----------------------------------------------------------------------------- checks
def check_C1(out, num):
    lam_hc, phi, lam1 = hopf_cole(Vquart, L, NU)
    rows = {}
    for N in (N0, 2 * N0, 4 * N0):
        G, X, pk, m = out['_base'][N]
        p = phi(G.x)
        mhc = M1 * p ** 2 / np.dot(G.w, p ** 2)
        rows[N] = (abs(X[N + 1] - lam_hc), float(np.max(np.abs(m - mhc)) / np.max(mhc)))
    num['linear_game']['lambda_hopf_cole_spectral'] = float(lam_hc)
    num['linear_game']['tunnelling_gap_lambda1_minus_lambda0'] = float(lam1 - lam_hc)
    num['herding_D4_D5']['tunnelling_gap_lambda1_minus_lambda0'] = float(lam1 - lam_hc)
    num['herding_D4_D5']['two_level_estimate_kappa_c_eq_gap_over_2'] = float((lam1 - lam_hc) / 2)
    lam_e, m_e = rows[4 * N0]
    ok = lam_e < 1e-4 and m_e < 1e-3
    return ok, f'N={4*N0}: |dλ| = {lam_e:.2e}, sup|Δm|/max m = {m_e:.2e} (N={N0}: {rows[N0][0]:.1e}, {rows[N0][1]:.1e}; N={2*N0}: {rows[2*N0][0]:.1e}, {rows[2*N0][1]:.1e})', rows


def check_C2():
    G = Grid(NP, L=6.0)
    V = 0.5 * G.x ** 2
    X = d0_solution(G, V)
    m = Dial('D0').density(X[:NP + 1], X[NP + 2], G.x, 0)[0]
    ref = M1 * np.exp(-G.x ** 2 / (2 * NU)) / np.sqrt(2 * np.pi * NU)
    el, em = abs(X[NP + 1] - NU), float(np.max(np.abs(m - ref)) / ref.max())
    return el < 1e-3 and em < 1e-3, f'L=6, N={NP}: |λ-ν| = {el:.2e}, sup|Δm|/max m = {em:.2e}', (el, em)


def check_C3(out):
    h = out['_herd']
    pk = {N: out['_base'][N][2] for N in out['_base']}
    cases = [('D0', None), ('D1', 0.8), ('D1', 1.5), ('D1', 2.0), ('D2', 1.0), ('D2', 2.0), ('D3', 0.5), ('D3', 1.0),
             ('D4', 0.5 * h['tc']), ('D4', 1.5 * h['tc']), ('D5', 0.5 * h['kc']), ('D5', 1.5 * h['kc'])]
    table, allok = [], True
    Xprod = {}
    for kind, val in cases:
        errs = []
        for N in (N0, 2 * N0, 4 * N0):
            G, X0 = out['_base'][N][0], out['_base'][N][1]
            V = Vquart(G.x)
            herded = (kind in ('D4', 'D5') and val > (h['tc'] if kind == 'D4' else h['kc']))
            mk = lambda v, kind=kind, N=N: Dial(kind, v, peak0=pk[N]) if kind != 'D0' else Dial('D0')
            if kind == 'D0':
                X, d, mode = X0, Dial('D0'), 'sym'
            elif herded:
                src = out['_herd']['sol'][min(h['sol'], key=lambda t: abs(t - val))] if kind == 'D4' else h['X5_15']
                Gp = out['_base'][NP][0]
                Xg = np.concatenate([np.interp(G.x, Gp.x, src[:Gp.N + 1]), src[Gp.N + 1:]])
                d, mode = Dial(kind, val), 'xbar'
                X, info = newton(G, d, V, M1, Xg, mode='xbar')
            else:
                start = 1.0 if kind == 'D1' else 0.0
                r, lst = follow(G, mk, np.linspace(start, val, 21), V, M1, X0)
                X, d, mode = r[-1][1], mk(val), 'sym'
            st = state(G, d, V, M1, X, mode)
            vf, mf = face_velocity(G, d, st, X, mode)
            gam = val if kind == 'D1' else 1.0
            mfp = fp_solve(G, vf, NU, gam, M1, symmetric=not herded)
            errs.append(float(np.max(np.abs(mfp - st['m'])) / np.max(st['m'])))
        exact = max(errs) < 1e-12          # scheme exact to rounding: nothing left to decrease
        ok = errs[-1] < 1e-3 and (exact or errs[0] > errs[1] > errs[2])
        allok &= ok
        lab = kind if val is None else f'{kind} {val:.4g}'
        table.append((lab, errs, ok))
    return allok, table


def check_C6(out, num):
    rows = {}
    rows['lambda_D0'] = [float(out['_base'][N][1][N + 1]) for N in (N0, 2 * N0, 4 * N0)]
    inner, outer, plat = [], [], []
    for N in (N0, 2 * N0, 4 * N0):
        G, X0, pk, _ = out['_base'][N]
        V = Vquart(G.x)
        r, lst = follow(G, lambda g: Dial('D1', g), np.linspace(1, 2, 41), V, M1, X0)
        X = r[-1][1]
        base = 1 + (2 - 1) * (X[N + 2] - X[:N + 1]) / (NU * 2)
        e = crossings(G.x, base)
        inner.append(float(e[2])); outer.append(float(e[3]))
        r, lst = follow(G, lambda y: Dial('D2', y, peak0=pk), np.linspace(0, 2, 41), V, M1, X0)
        st = state(G, Dial('D2', 2.0, peak0=pk), V, M1, r[-1][1])
        plat.append(width_above(G.x, st['m'] - 0.95 * pk / 2.0)[0])
    rows['edge_inner_D1_gamma2'] = inner
    rows['edge_outer_D1_gamma2'] = outer
    rows['plateau_width_D2_y2'] = plat
    rows['theta_c'] = [float(out['_herd']['thc'][N]) for N in (N0, 2 * N0, 4 * N0)]
    res, ok = {}, True
    for k, q in rows.items():
        res[k] = {'values_N_2N_4N': q, 'observed_order': obs_order(q), 'stable_3_sig_figs': sig3_stable(q)}
        ok &= res[k]['stable_3_sig_figs']
    num['grid_refinement'] = res
    return ok, res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    npz = os.path.join(HERE, 'results.npz')
    if os.path.exists(npz) and not args.force:
        print(f'results.npz already exists at {npz}; nothing re-solved (use --force to re-solve).')
        return
    t0 = time.time()
    out, num = {}, {}
    lap = lambda s: print(f'[{time.time() - t0:7.1f}s] {s}', flush=True)
    task_baseline(out, num); lap('baseline D0 on N, 2N, 4N')
    task_frontier(out, num); lap('frontier map')
    task_tails(out, num); lap('tails map + lower end')
    task_capacity(out, num); lap('capacity map')
    task_congestion(out, num); lap('congestion map + eps sweep')
    task_herding(out, num); lap('herding thresholds, branches, map')
    task_decoupling(out, num); lap('decoupling M -> 2M')
    task_mechanisms(out, num); lap('two mechanisms')
    task_planner(out, num); lap('planner vs toll game')
    checks = []
    ok, txt, _ = check_C1(out, num); checks.append(('C1', 'D0 vs Hopf–Cole ground state (spectral cosine-Galerkin eigen-solver)', 'sup rel. m error < 1e-3 at finest grid; λ error < 1e-4', txt, ok))
    ok, txt, _ = check_C2(); checks.append(('C2', 'Harmonic sanity: V = x²/2, L = 6, λ = ν, m = M·N(0, ν)', '< 1e-3', txt, ok))
    ok3, tab = check_C3(out)
    num['forward_equation_check'] = {lab: {'sup_rel_err_N_2N_4N': e, 'pass': o} for lab, e, o in tab}
    txt = '; '.join(f'{lab}: {e[0]:.1e}→{e[1]:.1e}→{e[2]:.1e}' for lab, e, o in tab)
    checks.append(('C3', 'Stationary forward equation (finite volumes, solved for the computed velocity field, no G) vs G(u); N, 2N, 4N', 'sup|Δm|/max m < 1e-3 at 4N, decreasing with h (or exact to rounding, < 1e-12, on every grid)', txt, ok3))
    checks.append(('C4', 'Mass on every converged solve', 'abs error < 1e-8', f'max |∫m − M| = {STATS["mass_err_max"]:.1e} over {STATS["n_solves"]} solves', STATS['mass_err_max'] < 1e-8))
    mk = float(np.max(out['capacity_max_m_over_K']))
    checks.append(('C5', 'Capacity respected over the capacity map', 'max m/K ≤ 1', f'max m/K = {mk:.12f}', mk <= 1.0))
    ok6, r6 = check_C6(out, num)
    txt = '; '.join(f'{k}: ' + ', '.join(f'{v:.6g}' for v in d['values_N_2N_4N']) + f' (order {d["observed_order"]:.2f})' for k, d in r6.items())
    checks.append(('C6', f'Grid refinement N = {N0}, {2*N0}, {4*N0}', 'observed order reported; 3 significant figures stable', txt, ok6))
    hd = num['herding_D4_D5']
    ok7 = hd['theta_c_rel_diff'] < 1e-3 and hd['kappa_c_rel_diff'] < 1e-3
    checks.append(('C7', 'θ_c and κ_c two ways (continuation of the herded branch vs linearised self-consistency)', 'agree to 1e-3 relative',
                   f'θ_c: {hd["theta_c_continuation"]:.8f} vs {hd["theta_c_linearised"]:.8f} (rel {hd["theta_c_rel_diff"]:.1e}); κ_c: {hd["kappa_c_continuation"]:.6e} vs {hd["kappa_c_linearised"]:.6e} (rel {hd["kappa_c_rel_diff"]:.1e})', ok7))
    pl = num['planner_D3']['by_c']
    okJ = all(abs(d['rel_diff_J_toll_vs_MFC']) < 1e-6 for d in pl.values())
    okv = all(d['v_rel_sup_diff_bulk_m_ge_1e-3max'] < 1e-3 for d in pl.values())
    okp = all(d['perturbations_lowering_J'] == 0 for d in pl.values())
    okg = abs(pl['0']['gap_MFG_vs_MFC_rel']) < 1e-9
    txt = '; '.join(f'c={c}: ΔJ/J {d["rel_diff_J_toll_vs_MFC"]:.1e}, Δv {d["v_rel_sup_diff_bulk_m_ge_1e-3max"]:.1e}, lowered {d["perturbations_lowering_J"]}/100' for c, d in pl.items())
    txt += f'; gap at c=0: {pl["0"]["gap_MFG_vs_MFC_rel"]:.1e}'
    checks.append(('C8', f'Planner (direct minimisation, no HJB) vs toll game (HJB), N = {N_PLAN}', 'J to 1e-6 rel; v to 1e-3 (sup over m ≥ 1e-3 max m, relative); perturbations never lower J; gap 0 at c = 0', txt, okJ and okv and okp and okg))
    dc = num['decoupling_M1_to_M2']['D0']
    ok9 = abs(dc['delta_lambda']) < 1e-10 and dc['sup_abs_delta_u'] < 1e-10
    checks.append(('C9', 'Decoupling: D0 λ and u under M → 2M', '< 1e-10', f'|Δλ| = {abs(dc["delta_lambda"]):.1e}, sup|Δu| = {dc["sup_abs_delta_u"]:.1e}', ok9))
    num['checks'] = {c[0]: {'value': c[3], 'pass': bool(c[4])} for c in checks}
    wall = time.time() - t0
    num['wall_time_seconds'] = wall
    # write
    arrs = {k: np.asarray(v) for k, v in out.items() if not k.startswith('_')}
    arrs['x'] = out['_base'][NP][0].x
    np.savez_compressed(npz, **arrs)
    with open(os.path.join(HERE, 'numbers.json'), 'w') as f:
        json.dump(num, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    NL = chr(10)
    md = ['# Checks', '', 'Written by `solve.py`. Production grid N = %d on [-%g, %g]; ladder N = %d, %d, %d.' % (NP, L, L, N0, 2 * N0, 4 * N0), '',
          '| # | Check | Threshold | Value | Result |', '|---|---|---|---|---|']
    cell = lambda t: str(t).replace('|', '∣')        # keep the table intact
    md += [f'| {c[0]} | {cell(c[1])} | {cell(c[2])} | {cell(c[3])} | {"PASS" if c[4] else "FAIL"} |' for c in checks]
    md += ['', '## C3 detail (sup ∣m_FP − G(u)∣ / max m)', '', '| case | N=%d | N=%d | N=%d | result |' % (N0, 2 * N0, 4 * N0), '|---|---|---|---|---|']
    md += [f'| {lab} | {e[0]:.2e} | {e[1]:.2e} | {e[2]:.2e} | {"PASS" if o else "FAIL"} |' for lab, e, o in tab]
    md += ['', '## C8 detail (N = %d)' % N_PLAN, '',
           '| c | J_MFC (direct) | J_toll (HJB) | rel. diff | J_MFG | gap | Δv sup (bulk) | Δv L²(m) | lowered |', '|---|---|---|---|---|---|---|---|---|']
    md += [f'| {c} | {d["J_MFC_direct"]:.12f} | {d["J_toll_game_HJB"]:.12f} | {d["rel_diff_J_toll_vs_MFC"]:.1e} | {d["J_MFG_plain_game"]:.12f} | {d["gap_MFG_vs_MFC_rel"]:.3e} | {d["v_rel_sup_diff_bulk_m_ge_1e-3max"]:.1e} | {d["v_rel_L2m_diff"]:.1e} | {d["perturbations_lowering_J"]}/100 |' for c, d in pl.items()]
    md += ['', f'Wall time: {wall:.1f} s', '']
    with open(os.path.join(HERE, 'CHECKS.txt'), 'w') as f:
        f.write(NL.join(md))
    for c in checks:
        print(f'{c[0]}: {"PASS" if c[4] else "FAIL"}  {c[3]}')
    print(f'wall time {wall:.1f} s')


if __name__ == '__main__':
    main()
