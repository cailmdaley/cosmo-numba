"""Adaptive (dqags) vs fixed quadrature for the pure E/B transform, in the exact
setup of tests/cosmo_numba/test_schneider2022.py::test_pure_eb_linearity.

Data: CCL xi_pm (ccl_xi_pm_0.5_250_10_000.npy) interpolated onto the 1000-bin
CosmoCov grid, LSST CosmoCov covariance C (cosmocov_cov_lsst_xipm_0.5_250_1000.npy).
theta = 20 nodes of theta_int; tmin, tmax = theta[0], theta[-1]; pad_xim=True;
interp_order=5; local terms read from the integrated vector. Draws:
x = xi + alpha * L z, C = L L^T, z ~ N(0, I) seeded per chunk.

    PYTHONPATH=src python3.12 run.py time ALPHA
    PYTHONPATH=src python3.12 run.py chunk ALPHA C0 C1 NPROC   # chunks [C0, C1) of NCH draws
    PYTHONPATH=src python3.12 run.py check                    # fixed operator vs public fixed path

Each chunk c uses z from default_rng([SEED, c]) whatever alpha is, so every
alpha (and the fixed arm) sees the same z: common random numbers.
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes, get_pure_EB_operator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f"{HERE}/data"
SEED = 23
NCH = 500  # draws per chunk

theta_ccl, xip_ccl, xim_ccl = np.load(f"{DATA}/ccl_xi_pm_0.5_250_10_000.npy")
cov = np.load(f"{DATA}/cosmocov_cov_lsst_xipm_0.5_250_1000.npy")
theta_ = np.logspace(np.log10(0.5), np.log10(250.0), 1_000 + 1)
theta_int = np.mean([theta_[:-1], theta_[1:]], axis=0)
xipm = np.concatenate([
    np.interp(np.log(theta_int), np.log(theta_ccl), xip_ccl),
    np.interp(np.log(theta_int), np.log(theta_ccl), xim_ccl),
])
idx = np.round(np.linspace(0, len(theta_int) - 1, 22)).astype(int)
theta = theta_int[idx[1:-1]]
tmin, tmax = theta[0], theta[-1]
NI = len(theta_int)
L = np.linalg.cholesky(cov)


def T_adaptive(x):
    """test_schneider2022._pure_eb, quadrature='adaptive' (the dqags path of main)."""
    p, m = x[:NI], x[NI:]
    i = np.searchsorted(theta_int, theta)
    return np.array(get_pure_EB_modes(theta, p[i], m[i], theta_int, p, m, tmin, tmax,
                                      parallel=False, pad_xim=True, interp_order=5,
                                      quadrature="adaptive"))


def T_fixed_call(x):
    p, m = x[:NI], x[NI:]
    i = np.searchsorted(theta_int, theta)
    return np.array(get_pure_EB_modes(theta, p[i], m[i], theta_int, p, m, tmin, tmax,
                                      pad_xim=True, interp_order=5, quadrature="fixed"))


def operator():
    return np.stack(get_pure_EB_operator(theta, theta_int, tmin, tmax, pad_xim=True,
                                         interp_order=5, local_from_int=True))  # (6, 20, 2000)


def noise(c):
    z = np.random.default_rng([SEED, c]).standard_normal((NCH, 2 * NI))
    return z @ L.T  # (NCH, 2000), unit alpha


_ALPHA, _NOISE = None, None


def _w(i):
    return T_adaptive(xipm + _ALPHA * _NOISE[i])


def fname(alpha, c):
    return f"{HERE}/cache/adapt_a{alpha:g}_c{c:03d}.npz"


def main():
    global _ALPHA, _NOISE
    mode = sys.argv[1]
    if mode == "time":
        a = float(sys.argv[2])
        T_adaptive(xipm)
        n = noise(0)
        for i in range(5):
            t0 = time.time()
            T_adaptive(xipm + a * n[i])
            print(f"alpha={a} adaptive one draw: {time.time() - t0:.3f} s", flush=True)
    elif mode == "check":
        M = operator()
        n = noise(0)
        sig = np.sqrt(np.einsum("jkl,lm,jkm->jk", M, cov, M))
        for a in (1.0, 12.6):
            chk = max(np.max(np.abs(T_fixed_call(xipm + a * n[i]) - M @ (xipm + a * n[i])) / (a * sig))
                      for i in range(3))
            print(f"alpha={a}: max |fixed call - M x| / sigma = {chk:.2e}", flush=True)
        print("T_adaptive(xi) - M xi, in sigma(alpha=1):",
              np.max(np.abs(T_adaptive(xipm) - M @ xipm) / sig, axis=1), flush=True)
    elif mode == "chunk":
        a = float(sys.argv[2])
        c0, c1, nproc = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        os.makedirs(f"{HERE}/cache", exist_ok=True)
        _ALPHA = a
        for c in range(c0, c1):
            if os.path.exists(fname(a, c)):
                continue
            _NOISE = noise(c)
            t0 = time.time()
            with mp.get_context("fork").Pool(nproc) as pool:
                T = np.array(pool.map(_w, range(NCH), chunksize=10))
            np.savez(fname(a, c), T=T, alpha=a, chunk=c, seed=SEED, wall=time.time() - t0)
            print(f"alpha={a} chunk {c}: {time.time() - t0:.0f} s, nonfinite {(~np.isfinite(T)).sum()}",
                  flush=True)


if __name__ == "__main__":
    main()
