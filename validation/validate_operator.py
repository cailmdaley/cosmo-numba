"""Validation of the fixed-quadrature pure E/B operator ("Route B").

What is being validated
-----------------------
``get_pure_EB_modes`` defines its transform in two steps: (i) build a
degree-``k`` local-Taylor interpolant through the *integrand* samples, (ii)
integrate it.  Step (i) is exactly linear in ``(xi_p, xi_m)``; step (ii) is
delegated to QUADPACK's adaptive ``dqags``, which is not.
``get_pure_EB_operator`` keeps step (i) untouched and replaces step (ii) by a
fixed Gauss-Legendre rule applied cell by cell, which is exact for a piecewise
polynomial.

The right notion of correctness is therefore *"does the operator return the
exact integral of the reference interpolant?"* -- not *"does it return what
dqags returns?"*.  This script checks the former directly, against a
brute-force integration that uses only ``nb_interp1d`` (the same interpolator
object the reference integrand callback builds) and
``numpy.polynomial.legendre.leggauss``.  It then reports the comparison with
``get_pure_EB_modes`` as a diagnostic of the adaptive quadrature.

Run inside the project container:

    python3 validation/validate_operator.py --n-grid 1000

Author: Cail Daley
"""

import argparse
import time

import numpy as np

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes
from cosmo_numba.B_modes.schneider2022_nb import H_m, H_p, K_m, K_p
from cosmo_numba.B_modes.schneider2022_operator import (
    _QuadRule,
    _quad_weights,
    get_pure_EB_operator,
)
from cosmo_numba.math.interpolate.interpolate_1D import nb_interp1d
from cosmo_numba.math.utils import extend_log_grid, pad_1D

ALL_OUTPUTS = ("xip_E", "xip_B", "xim_E", "xim_B", "xip_amb", "xim_amb")
# order returned by get_pure_EB_modes
REF_ORDER = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")

K = 5
EXTRAP = 1


def smooth_xi(theta):
    """Smooth, cosmic-shear-like xi_pm (broken power laws)."""
    xip = 2.0e-5 * (theta / 10.0) ** -0.65 / (1.0 + (theta / 120.0) ** 2)
    xim = 6.0e-6 * (theta / 10.0) ** -1.35 / (1.0 + (theta / 200.0) ** 2)
    return xip, xim


# --------------------------------------------------------------------------
# Brute-force integration of the reference interpolant (independent check)
# --------------------------------------------------------------------------
_GL = np.polynomial.legendre.leggauss(40)


def bf_integral(g, x_start, h, x_end, x_a, x_b, k=K, e=EXTRAP):
    """int_{exp(x_a)}^{exp(x_b)} P[g](theta) dtheta by brute force.

    Uses a 40-node Gauss-Legendre rule on every cell of the interpolant, and
    the reference ``nb_interp1d`` object itself.  Shares no code with
    ``_quad_weights``.
    """
    nodes, wts = _GL
    itp = nb_interp1d(x_start, x_end, h, g, k, False, True, e, False)
    edges = [x_a]
    c = int(np.floor((x_a - x_start) / h)) + 1
    while x_start + c * h < x_b - 1e-13:
        edges.append(x_start + c * h)
        c += 1
    edges.append(x_b)
    edges = np.asarray(edges)
    lo, hi = edges[:-1, None], edges[1:, None]
    xq = (0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)).ravel()
    wq = (0.5 * (hi - lo) * wts).ravel()
    fq = itp.eval(np.ascontiguousarray(xq))
    return float(np.sum(wq * np.exp(xq) * fq))


def exact_pure_EB_modes(theta, xip, xim, tmin, tmax, pad_decade=1.0):
    """Reference transform, adaptive quadrature replaced by brute force.

    Mirrors ``_get_pure_EB_modes_serial`` line for line, except that every
    ``interp_quad`` call becomes a ``bf_integral`` call.  Slow, but independent
    of the operator machinery.
    """
    n = theta.size
    d = float(np.mean(np.diff(np.log(theta))))
    theta_bar = (tmax + tmin) / 2
    B = (tmax - tmin) / (tmax + tmin)

    theta_ext, i_low, i_high = extend_log_grid(theta, 0.0, pad_decade)
    xip_ext = pad_1D(theta_ext.size, i_low, i_high, xip)
    xim_ext = pad_1D(theta_ext.size, i_low, i_high, xim)
    tmax_xim = theta_ext[-1]
    tb_x = (tmax_xim + tmin) / 2
    B_x = (tmax_xim - tmin) / (tmax_xim + tmin)

    m0 = (theta >= tmin) & (theta <= tmax)
    g0 = theta[m0]
    d0 = float(np.mean(np.diff(np.log(g0))))
    m0e = (theta_ext >= tmin) & (theta_ext <= tmax_xim)
    g0e = theta_ext[m0e]
    d0e = float(np.mean(np.diff(np.log(g0e))))

    out = {name: np.full(n, np.nan) for name in ALL_OUTPUTS}
    for i in range(n):
        t = theta[i]

        m1 = (t < theta) & (theta <= tmax)
        n1 = int(m1.sum())
        if n1 == 0:
            int_p = 0.0
        elif n1 < K + 1:
            int_p = np.nan
        else:
            g1 = theta[m1]
            integ = np.ascontiguousarray(
                xim[m1] / g1 * (4.0 - 12.0 * t**2 / g1**2)
            )
            int_p = bf_integral(
                integ, np.log(g1[0]), d, np.log(g1[-1]),
                np.log(t), np.log(tmax),
            )
        s_p = bf_integral(
            np.ascontiguousarray(
                g0 * xip[m0] * H_p(t, g0, theta_bar, B) / theta_bar**2
            ),
            np.log(g0[0]), d0, np.log(g0[-1]), np.log(tmin), np.log(tmax),
        )
        s_m = bf_integral(
            np.ascontiguousarray(xim[m0] / g0 * H_m(t, g0, theta_bar, B)),
            np.log(g0[0]), d0, np.log(g0[-1]), np.log(tmin), np.log(tmax),
        )

        m2 = (t > theta_ext) & (theta_ext >= tmin)
        n2 = int(m2.sum())
        if n2 == 0:
            int_m = 0.0
        elif n2 < K + 1:
            int_m = np.nan
        else:
            g2 = theta_ext[m2]
            integ = np.ascontiguousarray(
                g2 / t**2 * xip_ext[m2] * (4.0 - 12.0 * g2**2 / t**2)
            )
            int_m = bf_integral(
                integ, np.log(g2[0]), d, np.log(g2[-1]),
                np.log(tmin), np.log(t),
            )
        v_p = bf_integral(
            np.ascontiguousarray(
                g0e * xip_ext[m0e] * K_p(t, g0e, tb_x, B_x) / tb_x**2
            ),
            np.log(g0e[0]), d0e, np.log(g0e[-1]),
            np.log(tmin), np.log(tmax_xim),
        )
        v_m = bf_integral(
            np.ascontiguousarray(
                g0e * xim_ext[m0e] * K_m(t, g0e, tb_x, B_x) / tb_x**2
            ),
            np.log(g0e[0]), d0e, np.log(g0e[-1]),
            np.log(tmin), np.log(tmax_xim),
        )

        out["xip_E"][i] = 0.5 * (xip[i] + xim[i] + int_p) - 0.5 * (s_p + s_m)
        out["xip_B"][i] = 0.5 * (xip[i] - xim[i] - int_p) - 0.5 * (s_p - s_m)
        out["xip_amb"][i] = s_p
        out["xim_E"][i] = 0.5 * (xip[i] + xim[i] + int_m) - 0.5 * (v_p + v_m)
        out["xim_B"][i] = 0.5 * (xip[i] - xim[i] + int_m) - 0.5 * (v_p - v_m)
        out["xim_amb"][i] = v_m
    return out


# --------------------------------------------------------------------------
# 1. weights
# --------------------------------------------------------------------------
def check_quadrature_weights():
    """Weights vs. brute force, over a range of grids and supports."""
    print("\n=== 1. quadrature weights vs. brute-force interpolant ===")
    rng = np.random.default_rng(12345)
    worst = 0.0
    for n_grid in (40, 150, 600):
        theta = np.geomspace(0.5, 300.0, n_grid)
        x = np.log(theta)
        h = float(np.mean(np.diff(x)))
        g = np.ascontiguousarray(rng.standard_normal(n_grid))
        cases = {
            "grid endpoints": (0, n_grid - 1, x[0], x[-1]),
            "sub-support": (7, n_grid - 1, x[7], x[-1]),
            "extrapolate left": (7, n_grid - 1, x[6], x[-1]),
            "extrapolate right": (0, n_grid - 2, x[0], x[-1]),
            "interior partial": (
                0, n_grid - 1, x[3] + 0.4 * h, x[-4] - 0.3 * h,
            ),
        }
        for name, (j0, j1, x_a, x_b) in cases.items():
            gg = np.ascontiguousarray(g[j0 : j1 + 1])
            rule = _QuadRule(K, EXTRAP, 16)
            got = _quad_weights(rule, gg.size, x[j0], h, x_a, x_b) @ gg
            ref = bf_integral(gg, x[j0], h, x[j1], x_a, x_b)
            rel = abs(got - ref) / max(abs(ref), 1e-300)
            worst = max(worst, rel)
            print(f"  n={n_grid:4d}  {name:18s} rel = {rel:.2e}")
    print(f"  -> worst relative deviation: {worst:.3e}")
    return worst


# --------------------------------------------------------------------------
# 2. Gauss node count
# --------------------------------------------------------------------------
def check_gauss_convergence(n_grid=200):
    """The rule must not depend on the number of Gauss nodes."""
    print("\n=== 2. Gauss-Legendre node-count convergence ===")
    theta = np.geomspace(0.5, 300.0, n_grid)
    x = np.log(theta)
    h = float(np.mean(np.diff(x)))
    g = np.ascontiguousarray(np.random.default_rng(7).standard_normal(n_grid))
    ref = None
    for n_gauss in (6, 8, 12, 16, 24, 32):
        val = _quad_weights(
            _QuadRule(K, EXTRAP, n_gauss), n_grid, x[0], h, x[0], x[-1]
        ) @ g
        ref = val if ref is None else ref
        print(
            f"  n_gauss={n_gauss:3d}  I = {val:+.16e}   "
            f"rel = {(val - ref) / ref:+.2e}"
        )


# --------------------------------------------------------------------------
# 3/4. end-to-end
# --------------------------------------------------------------------------
def check_end_to_end(n_grid, eps_list=(1e-10, 1e-12)):
    """Full transform: operator, brute force and get_pure_EB_modes."""
    print(f"\n=== end-to-end, n_grid = {n_grid} ===")
    theta = np.geomspace(0.5, 300.0, n_grid)
    tmin, tmax = theta[0], theta[-1]
    xip, xim = smooth_xi(theta)
    xvec = np.concatenate([xip, xim])
    scale = {"xip": np.max(np.abs(xip)), "xim": np.max(np.abs(xim))}

    t0 = time.perf_counter()
    op = get_pure_EB_operator(theta, theta, tmin, tmax, outputs=ALL_OUTPUTS)
    t_build = time.perf_counter() - t0
    mats = op["matrices"]
    nbytes = sum(m.nbytes for m in mats.values())
    print(
        f"  build : {t_build:7.2f} s  ({len(mats)} matrices of shape "
        f"{mats['xip_E'].shape}, {nbytes / 1e6:.0f} MB)"
    )

    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        applied = {name: m @ xvec for name, m in mats.items()}
    t_apply = (time.perf_counter() - t0) / n_rep
    print(
        f"  apply : {t_apply * 1e3:7.2f} ms for all {len(mats)} outputs "
        f"({t_apply * 1e3 / len(mats):.2f} ms each)"
    )

    t0 = time.perf_counter()
    exact = exact_pure_EB_modes(theta, xip, xim, tmin, tmax)
    print(f"  brute-force reference: {time.perf_counter() - t0:.1f} s")

    refs = {}
    for eps in eps_list:
        t0 = time.perf_counter()
        out = get_pure_EB_modes(
            theta, xip, xim, theta, xip, xim, tmin, tmax,
            parallel=True, pad_xim=True, pad_theta_max_decade=1.0,
            interp_order=K, epsabs=eps, epsrel=eps,
        )
        refs[eps] = dict(zip(REF_ORDER, out, strict=True))
        print(
            f"  get_pure_EB_modes (eps={eps:.0e}, parallel): "
            f"{time.perf_counter() - t0:.1f} s"
        )

    print(
        "\n  A. operator vs. EXACT integral of the reference interpolant\n"
        "     name       n_edge   max|op-exact|/|exact|   "
        "max|op-exact|/max|xi|"
    )
    worst_rel = 0.0
    for name in ALL_OUTPUTS:
        ok = op["valid"][name] & np.isfinite(exact[name])
        sc = scale["xip" if name.startswith("xip") else "xim"]
        dev = np.abs(applied[name] - exact[name])
        rel = float(np.max((dev / np.abs(exact[name]))[ok]))
        worst_rel = max(worst_rel, rel)
        print(
            f"     {name:9s} {int((~op['valid'][name]).sum()):4d}      "
            f"{rel:.3e}              {float(np.max(dev[ok])) / sc:.3e}"
        )
    print(f"     -> worst interior relative deviation: {worst_rel:.3e}")

    print(
        "\n  B. get_pure_EB_modes vs. the same EXACT integral (dqags error)\n"
        "     name       eps        max|ref-exact|/|exact|  "
        "max|ref-exact|/max|xi|"
    )
    for name in ALL_OUTPUTS:
        for eps in eps_list:
            ok = op["valid"][name] & np.isfinite(exact[name])
            sc = scale["xip" if name.startswith("xip") else "xim"]
            dev = np.abs(refs[eps][name] - exact[name])
            print(
                f"     {name:9s} {eps:.0e}   "
                f"{float(np.max((dev / np.abs(exact[name]))[ok])):.3e}"
                f"               {float(np.max(dev[ok])) / sc:.3e}"
            )

    print("\n  C. edge rows (masked support holds < k+1 = 6 nodes)")
    for name in ("xip_E", "xim_E"):
        bad = np.nonzero(~op["valid"][name])[0]
        print(
            f"     {name}: rows {bad.tolist()} (theta = "
            f"{np.array2string(theta[bad], precision=4)}) -> "
            f"{applied[name][bad]}"
        )

    print("\n  D. structural identities")
    rp = (mats["xip_E"] + mats["xip_B"] + mats["xip_amb"]) @ xvec
    rm = (mats["xim_E"] - mats["xim_B"] + mats["xim_amb"]) @ xvec
    vp, vm = op["valid"]["xip_E"], op["valid"]["xim_E"]
    print(
        f"     xi_p = E + B + amb : max rel "
        f"{np.max(np.abs(rp[vp] / xip[vp] - 1)):.2e}"
    )
    print(
        f"     xi_m = E - B + amb : max rel "
        f"{np.max(np.abs(rm[vm] / xim[vm] - 1)):.2e}"
    )

    rng = np.random.default_rng(3)
    y1 = xvec + 1e-6 * rng.standard_normal(xvec.size)
    y2 = 1e-6 * rng.standard_normal(xvec.size)
    a, b = -0.37, 2.4
    lin = 0.0
    for name, m in mats.items():
        ok = op["valid"][name]
        resid = (m @ (a * y1 + b * y2) - a * (m @ y1) - b * (m @ y2))[ok]
        lin = max(
            lin, float(np.max(np.abs(resid)) / np.max(np.abs((m @ y1)[ok])))
        )
    print(
        f"     linearity |M(a y1 + b y2) - a M y1 - b M y2| : {lin:.2e} (rel)"
    )
    return t_build, t_apply, worst_rel


# --------------------------------------------------------------------------
# 5. continuum convergence
# --------------------------------------------------------------------------
def check_grid_refinement():
    """Refine the grid: the operator converges, dqags degrades.

    All grids share the point theta = 0.5 * sqrt(600) = 12.247', so the
    left-hand columns compare like with like.
    """
    print("\n=== grid refinement ===")
    print(
        "  Left : xi_m^amb at the common point theta = 12.247' — continuum\n"
        "         convergence of the operator.\n"
        "  Right: worst disagreement between get_pure_EB_modes and the\n"
        "         operator over ALL interior evaluation points.\n"
    )
    print(
        "   n_grid   xi_m^amb(12.247')   dqags-op there    "
        "max|dqags-op|/max|xi_m|   at theta"
    )
    for n_grid in (251, 501, 1001, 2001, 4001):
        theta = np.geomspace(0.5, 300.0, n_grid)
        tmin, tmax = theta[0], theta[-1]
        xip, xim = smooth_xi(theta)
        xvec = np.concatenate([xip, xim])
        i = (n_grid - 1) // 2
        op = get_pure_EB_operator(
            theta, theta, tmin, tmax, outputs=("xim_amb",)
        )
        val = op["matrices"]["xim_amb"] @ xvec
        ref = get_pure_EB_modes(
            theta, xip, xim, theta, xip, xim, tmin, tmax,
            parallel=True, pad_xim=True, pad_theta_max_decade=1.0,
            interp_order=K, epsabs=1e-12, epsrel=1e-12,
        )[5]
        dev = np.abs(ref - val) / np.max(np.abs(xim))
        j = int(np.argmax(dev))
        rel_i = (ref[i] - val[i]) / val[i]
        print(
            f"   {n_grid:6d}   {val[i]:.12e}   {rel_i:+.2e}"
            f"         {dev[j]:.3e}            {theta[j]:8.3f}"
        )


def check_dqags_sporadic(n_lo=995, n_hi=1012):
    """dqags failures are sporadic: most grids are fine, a few are not."""
    print("\n=== sporadic dqags failures (xi_m^amb, eps = 1e-12) ===")
    print(
        "   n_grid   max|dqags-op|/max|xi_m|     at theta      dqags       op"
    )
    for n_grid in range(n_lo, n_hi):
        theta = np.geomspace(0.5, 300.0, n_grid)
        tmin, tmax = theta[0], theta[-1]
        xip, xim = smooth_xi(theta)
        op = get_pure_EB_operator(
            theta, theta, tmin, tmax, outputs=("xim_amb",)
        )
        val = op["matrices"]["xim_amb"] @ np.concatenate([xip, xim])
        ref = get_pure_EB_modes(
            theta, xip, xim, theta, xip, xim, tmin, tmax,
            parallel=True, pad_xim=True, pad_theta_max_decade=1.0,
            interp_order=K, epsabs=1e-12, epsrel=1e-12,
        )[5]
        dev = np.abs(ref - val) / np.max(np.abs(xim))
        j = int(np.argmax(dev))
        flag = "  <== FAILURE" if dev[j] > 1e-8 else ""
        print(
            f"   {n_grid:6d}   {dev[j]:.3e}              {theta[j]:8.3f}   "
            f"{ref[j]:.6e} {val[j]:.6e}{flag}"
        )


def check_scaling(sizes=(2000, 5000, 10000)):
    """Build time / apply time / memory at pipeline-scale grids."""
    print("\n=== scaling (4 E/B matrices, theta_eval == theta) ===")
    print("   n_grid    build [s]    apply [ms]    memory [GB]")
    for n_grid in sizes:
        theta = np.geomspace(0.5, 300.0, n_grid)
        xip, xim = smooth_xi(theta)
        xvec = np.concatenate([xip, xim])
        t0 = time.perf_counter()
        op = get_pure_EB_operator(theta, theta, theta[0], theta[-1])
        t_build = time.perf_counter() - t0
        mats = op["matrices"]
        t0 = time.perf_counter()
        for _ in range(5):
            for m in mats.values():
                m @ xvec
        t_apply = (time.perf_counter() - t0) / 5
        print(
            f"   {n_grid:6d}    {t_build:8.2f}    {t_apply * 1e3:8.2f}    "
            f"{sum(m.nbytes for m in mats.values()) / 1e9:9.2f}"
        )
        del op, mats


def main():
    """Run every check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grid", type=int, default=1000)
    parser.add_argument("--n-grid-small", type=int, default=200)
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--skip-sporadic", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    args = parser.parse_args()

    check_quadrature_weights()
    check_gauss_convergence()
    check_end_to_end(args.n_grid_small)
    check_end_to_end(args.n_grid)
    if not args.skip_refinement:
        check_grid_refinement()
    if not args.skip_sporadic:
        check_dqags_sporadic()
    if not args.skip_scaling:
        check_scaling()


if __name__ == "__main__":
    main()
