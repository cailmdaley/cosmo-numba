"""Is get_pure_EB_modes linear in its input? Each noise draw n = alpha L z (L = chol of the
test-data covariance, alpha = 12.6) goes in as an antithetic pair xi +/- n; the same z at
unit amplitude gives the small-signal response, sigma_ref = alpha * std[(T(xi+Lz) - T(xi-Lz))/2].
A linear T has <T(xi +/- n)> = T(xi) and std T(xi+n) = sigma_ref; deviations are non-linearity.

    python reproduce.py --data <cosmo-numba>/tests/cosmo_numba/data --n 500 --nproc 8 \
        [--transform module:function | path/to/file.py:function]

A custom transform takes the same arguments as default_transform below and returns the
6 outputs of get_pure_EB_modes (xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb).
"""
import argparse
import importlib
import importlib.util
import multiprocessing as mp
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 12.6  # noise amplitude relative to the LSST test covariance (UNIONS-like)
ALPHA_REF = 1.0  # small-signal reference amplitude (the test covariance itself)
SEED = 23


def default_transform(theta, xip, xim, theta_int, xip_int, xim_int, tmin, tmax):
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    return get_pure_EB_modes(theta, xip, xim, theta_int, xip_int, xim_int, tmin, tmax,
                             pad_xim=True, interp_order=5)


def load_transform(spec):
    if spec is None:
        return default_transform
    mod, fn = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location("user_transform", mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, fn)


def setup(data):
    """Fine integration grid theta_int, 20 reporting bins with theta^2 dln(theta) weights."""
    th_ccl, xip_ccl, xim_ccl = np.load(f"{data}/ccl_xi_pm_0.5_250_10_000.npy")
    cov = np.load(f"{data}/cosmocov_cov_lsst_xipm_0.5_250_1000.npy")
    edges_f = np.logspace(np.log10(0.5), np.log10(250.0), 1001)
    ti = 0.5 * (edges_f[:-1] + edges_f[1:])
    xi = np.concatenate([np.interp(np.log(ti), np.log(th_ccl), xip_ccl),
                         np.interp(np.log(ti), np.log(th_ccl), xim_ccl)])
    edges = np.logspace(0, np.log10(250.0), 21)
    w = ti**2 * np.diff(np.log(edges_f))
    B = np.zeros((20, len(ti)))
    for b in range(20):
        s = (ti >= edges[b]) & (ti < edges[b + 1])
        B[b, s] = w[s] / w[s].sum()
    return ti, xi, np.linalg.cholesky(cov), B, B @ ti


def T(x):
    """Binned pure E/B transform of a fine-grid input x = [xip_int, xim_int]."""
    p, m = x[:NI], x[NI:]
    return np.array(TRANSFORM(THETA, B @ p, B @ m, TI, p, m, THETA[0], THETA[-1]))


def draw(i):
    """T(xi +/- alpha L z) and the unit-amplitude half-difference for the same z."""
    n = L @ np.random.default_rng([SEED, i]).standard_normal(2 * NI)
    ref = (T(XI + ALPHA_REF * n) - T(XI - ALPHA_REF * n)) / (2 * ALPHA_REF)
    return T(XI + ALPHA * n), T(XI - ALPHA * n), ref


def main():
    global TRANSFORM, TI, XI, L, B, THETA, NI
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cosmo-numba tests/cosmo_numba/data")
    ap.add_argument("--n", type=int, default=500, help="number of noise draws")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--transform", default=None, help="module:function (default get_pure_EB_modes)")
    ap.add_argument("--out", default="reproduce.png")
    ap.add_argument("--save", default=None, help="optional .npz of all transform outputs")
    a = ap.parse_args()

    TRANSFORM = load_transform(a.transform)
    TI, XI, L, B, THETA = setup(a.data)
    NI = len(TI)

    t0 = time.time()
    T0 = T(XI)  # noiseless reference; also compiles numba before forking
    print(f"noiseless call (incl. JIT): {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.nproc) as pool:
        Tp, Tm, R = map(np.array, zip(*pool.map(draw, range(a.n), chunksize=2)))
    wall = time.time() - t0
    print(f"{a.n} draws ({4 * a.n} calls) on {a.nproc} procs: {wall:.0f}s "
          f"(~{wall * a.nproc / (4 * a.n):.2f} core-s per call)")
    N = a.n
    if a.save:
        np.savez(a.save, theta=THETA, T0=T0, Tp=Tp, Tm=Tm, R=R)

    sig_ref = ALPHA * R.std(0, ddof=1)
    s = (Tp + Tm) / 2
    bias = (s.mean(0) - T0) / sig_ref
    bias_err = s.std(0, ddof=1) / np.sqrt(N) / sig_ref
    ratio = sig_ref / Tp.std(0, ddof=1)
    ratio_err = ratio / np.sqrt(2 * (N - 1))

    names = {2: "xip_B", 3: "xim_B"}
    print(f"\n{'theta':>7} | " + " | ".join(f"{names[j]} bias/sig   ratio" for j in (2, 3)))
    for k, t in enumerate(THETA):
        print(f"{t:7.2f} | " + " | ".join(f"{bias[j, k]:+.3f}+-{bias_err[j, k]:.3f}  {ratio[j, k]:.3f}"
                                         for j in (2, 3)))

    plt.rcParams.update({"font.size": 11})
    col = {2: "#D55E00", 3: "#0072B2"}
    lab = {2: r"$\xi_+^B$", 3: r"$\xi_-^B$"}
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 5.8), sharex=True, height_ratios=(1.6, 1))
    for j, dx in ((2, 0.97), (3, 1.03)):
        kw = dict(marker="o", ms=5, lw=0.9, color=col[j], capsize=0, label=lab[j])
        ax0.errorbar(THETA * dx, bias[j], bias_err[j], **kw)
        ax1.errorbar(THETA * dx, ratio[j], ratio_err[j], **kw)
    ax0.axhline(0, color="0.4", lw=0.8, ls="--")
    ax1.axhline(1, color="0.4", lw=0.8, ls="--")
    ax0.set_ylabel(r"$(\langle T(\xi\pm n)\rangle - T(\xi))\,/\,\sigma_{\rm ref}$")
    ax1.set_ylabel(r"$\sigma_{\rm ref}/\sigma_{\rm emp}$")
    ax1.set_xlabel(r"$\theta$ [arcmin]")
    ax1.set_xscale("log")
    ax1.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax0.legend(frameon=False, loc="lower right")
    ax0.set_title(f"{a.transform or 'get_pure_EB_modes'}: noise x{ALPHA:g}, {N} draws",
                  fontsize=9.5, color="0.3")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
