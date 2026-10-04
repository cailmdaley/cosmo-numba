"""dqags non-linearity of the pure E/B transform with the repo's own test data, used
the way the pipeline uses it: local xi_pm bin-averaged into 20 reporting bins.

Data (issue23-repo/data = cosmo-numba tests/cosmo_numba/data): CCL xi_pm interpolated
in ln theta onto theta_int = midpoints of logspace(0.5, 250, 1001); LSST CosmoCov cov C.
Reporting bins: edges logspace(0, log10 250, 21); B row b = weights w_i ∝ theta_i^2 dln(theta_i)
(pair-count proxy) over fine nodes in bin b, normalised; theta_b = weighted mean theta.
get_pure_EB_modes(theta, B xip, B xim, theta_int, xip, xim, theta[0], theta[-1],
pad_xim=True, interp_order=5). Adaptive arm = quadrature="adaptive" (dqags);
fixed arm = M' x with M' = M J, M = get_pure_EB_operator(..., local_from_int=False).
Draws x = xi + alpha L z, z from default_rng([SEED, c]) per chunk c (common random numbers).

    PYTHONPATH=../issue23-repo/src python3.12 run.py check
    PYTHONPATH=../issue23-repo/src python3.12 run.py chunk ALPHA C0 C1 NPROC
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes, get_pure_EB_operator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{HERE}/../issue23-repo/data"
SEED = 23
NCH = 500

theta_ccl, xip_ccl, xim_ccl = np.load(f"{DATA}/ccl_xi_pm_0.5_250_10_000.npy")
cov = np.load(f"{DATA}/cosmocov_cov_lsst_xipm_0.5_250_1000.npy")
edges_f = np.logspace(np.log10(0.5), np.log10(250.0), 1001)
ti = 0.5 * (edges_f[:-1] + edges_f[1:])
NI = len(ti)
xipm = np.concatenate([np.interp(np.log(ti), np.log(theta_ccl), xip_ccl),
                       np.interp(np.log(ti), np.log(theta_ccl), xim_ccl)])
L = np.linalg.cholesky(cov)

edges = np.logspace(0, np.log10(250.0), 21)
w_all = ti**2 * np.diff(np.log(edges_f))
B = np.zeros((20, NI))
for b in range(20):
    s = (ti >= edges[b]) & (ti < edges[b + 1])
    B[b, s] = w_all[s] / w_all[s].sum()
theta = B @ ti
TMIN, TMAX = float(theta[0]), float(theta[-1])
Z = np.zeros_like(B)
J = np.block([[B, Z], [Z, B], [np.eye(NI), np.zeros((NI, NI))], [np.zeros((NI, NI)), np.eye(NI)]])


def operator():
    ops = get_pure_EB_operator(theta, ti, TMIN, TMAX, pad_xim=True, interp_order=5,
                               local_from_int=False)
    return np.stack([op @ J for op in ops])  # (6, 20, 2000)


def T_ad(x):
    p, m = x[:NI], x[NI:]
    return np.array(get_pure_EB_modes(theta, B @ p, B @ m, ti, p, m, TMIN, TMAX, parallel=False,
                                      pad_xim=True, interp_order=5, quadrature="adaptive"))


def T_fx(x):
    p, m = x[:NI], x[NI:]
    return np.array(get_pure_EB_modes(theta, B @ p, B @ m, ti, p, m, TMIN, TMAX,
                                      pad_xim=True, interp_order=5, quadrature="fixed"))


def noise(c):
    return np.random.default_rng([SEED, c]).standard_normal((NCH, 2 * NI)) @ L.T


def fname(a, c):
    return f"{HERE}/cache/adapt_a{a:g}_c{c:03d}.npz"


_A, _N = None, None


def _w(i):
    return T_ad(xipm + _A * _N[i])


def main():
    global _A, _N
    mode = sys.argv[1]
    if mode == "check":
        print("fine nodes per bin:", (B > 0).sum(1))
        print("theta:", np.round(theta, 3), "tmin/tmax", TMIN, TMAX)
        M = operator()
        sig = np.sqrt(np.einsum("jkl,lm,jkm->jk", M, cov, M))
        n = noise(0)
        for a in (1.0, 12.6):
            t0 = time.time()
            chk = max(np.max(np.abs(T_fx(xipm + a * n[i]) - M @ (xipm + a * n[i])) / (a * sig)) for i in range(3))
            print(f"alpha={a}: max|fixed call - M'x|/sigma = {chk:.2e}", flush=True)
            T_ad(xipm + a * n[0]); t0 = time.time(); T_ad(xipm + a * n[1])
            print(f"alpha={a}: adaptive one draw {time.time()-t0:.2f}s", flush=True)
    elif mode == "chunk":
        a = float(sys.argv[2]); c0, c1, nproc = map(int, sys.argv[3:6])
        _A = a
        for c in range(c0, c1):
            if os.path.exists(fname(a, c)):
                continue
            _N = noise(c)
            t0 = time.time()
            with mp.get_context("fork").Pool(nproc) as pool:
                T = np.array(pool.map(_w, range(NCH), chunksize=4))
            np.savez(fname(a, c), T=T, alpha=a, chunk=c, seed=SEED, wall=time.time() - t0)
            print(f"alpha={a} chunk {c}: {time.time()-t0:.0f}s nonfinite {(~np.isfinite(T)).sum()}", flush=True)


if __name__ == "__main__":
    main()
