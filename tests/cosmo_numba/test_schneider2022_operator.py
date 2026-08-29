"""Tests for the fixed-quadrature pure E/B operator.

Mirrors ``validation/validate_operator.py`` at coarse resolution so the suite
stays a few seconds.  The reference for correctness is the *exact* integral of
the degree-k interpolant that ``get_pure_EB_modes`` builds, obtained here by
brute-force Gauss-Legendre integration of ``nb_interp1d``; the adaptive
``dqags`` used by ``get_pure_EB_modes`` is only accurate to ~1e-6 relative on
grids this coarse, so it is compared against with a loose tolerance.
"""

import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_SCHNEIDER2022"] = "1"

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes
from cosmo_numba.B_modes.schneider2022_operator import (
    OUTPUT_NAMES,
    _QuadRule,
    _quad_weights,
    get_pure_EB_operator,
)
from cosmo_numba.math.interpolate.interpolate_1D import nb_interp1d

K = 5
EXTRAP = 1
N_GRID = 80

# order returned by get_pure_EB_modes
REF_ORDER = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")


def _brute_force_integral(g, x_start, h, x_end, x_a, x_b, n_gauss=40):
    """Integral of the reference interpolant, by composite Gauss-Legendre."""
    nodes, wts = np.polynomial.legendre.leggauss(n_gauss)
    itp = nb_interp1d(x_start, x_end, h, g, K, False, True, EXTRAP, False)
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


@pytest.fixture(scope="module")
def setup():
    """Coarse grid and a smooth xi_pm."""
    theta = np.geomspace(0.5, 300.0, N_GRID)
    xip = 2.0e-5 * (theta / 10.0) ** -0.65 / (1.0 + (theta / 120.0) ** 2)
    xim = 6.0e-6 * (theta / 10.0) ** -1.35 / (1.0 + (theta / 200.0) ** 2)
    return theta, xip, xim


def test_quadrature_weights_are_exact():
    """The fixed rule integrates the reference interpolant exactly."""
    theta = np.geomspace(0.5, 300.0, 60)
    x = np.log(theta)
    h = float(np.mean(np.diff(x)))
    g = np.ascontiguousarray(np.random.default_rng(1).standard_normal(60))
    rule = _QuadRule(K, EXTRAP, 16)

    cases = {
        "endpoints": (0, 59, x[0], x[-1]),
        "sub-support": (7, 59, x[7], x[-1]),
        "extrapolate left": (7, 59, x[6], x[-1]),
        "extrapolate right": (0, 58, x[0], x[-1]),
        "interior partial": (0, 59, x[3] + 0.4 * h, x[-4] - 0.3 * h),
    }
    for name, (j0, j1, x_a, x_b) in cases.items():
        gg = np.ascontiguousarray(g[j0 : j1 + 1])
        got = _quad_weights(rule, gg.size, x[j0], h, x_a, x_b) @ gg
        ref = _brute_force_integral(gg, x[j0], h, x[j1], x_a, x_b)
        assert_allclose(got, ref, rtol=1e-11, atol=0.0, err_msg=name)


def test_gauss_node_count_is_converged():
    """The rule must not depend on the number of Gauss nodes."""
    theta = np.geomspace(0.5, 300.0, 60)
    x = np.log(theta)
    h = float(np.mean(np.diff(x)))
    g = np.ascontiguousarray(np.random.default_rng(2).standard_normal(60))
    vals = [
        _quad_weights(_QuadRule(K, EXTRAP, ng), 60, x[0], h, x[0], x[-1]) @ g
        for ng in (8, 16, 32)
    ]
    assert_allclose(vals[1], vals[0], rtol=1e-13)
    assert_allclose(vals[2], vals[0], rtol=1e-13)


def test_operator_is_exactly_linear(setup):
    """M(a y1 + b y2) == a M y1 + b M y2 to round-off."""
    theta, xip, xim = setup
    op = get_pure_EB_operator(theta, theta, theta[0], theta[-1])
    rng = np.random.default_rng(4)
    y1 = rng.standard_normal(2 * theta.size)
    y2 = rng.standard_normal(2 * theta.size)
    a, b = -0.37, 2.4
    for name, m in op["matrices"].items():
        ok = op["valid"][name]
        lhs = (m @ (a * y1 + b * y2))[ok]
        rhs = (a * (m @ y1) + b * (m @ y2))[ok]
        assert_allclose(lhs, rhs, rtol=1e-12, atol=0.0, err_msg=name)


def test_reconstruction_identity(setup):
    """E + B + amb must return the input exactly (structural)."""
    theta, xip, xim = setup
    op = get_pure_EB_operator(
        theta, theta, theta[0], theta[-1], outputs=OUTPUT_NAMES
    )
    m = op["matrices"]
    xvec = np.concatenate([xip, xim])
    vp = op["valid"]["xip_E"]
    vm = op["valid"]["xim_E"]
    rec_p = ((m["xip_E"] + m["xip_B"] + m["xip_amb"]) @ xvec)[vp]
    rec_m = ((m["xim_E"] - m["xim_B"] + m["xim_amb"]) @ xvec)[vm]
    assert_allclose(rec_p, xip[vp], rtol=1e-13, atol=0.0)
    assert_allclose(rec_m, xim[vm], rtol=1e-13, atol=0.0)


def test_edge_policy(setup):
    """Rows whose masked support holds < k+1 nodes are NaN and flagged."""
    theta, xip, xim = setup
    op = get_pure_EB_operator(
        theta, theta, theta[0], theta[-1], outputs=OUTPUT_NAMES
    )
    xvec = np.concatenate([xip, xim])

    # xi_p: the integral runs over (t, tmax], so the top end is affected
    bad_p = np.nonzero(~op["valid"]["xip_E"])[0]
    assert bad_p.tolist() == list(range(N_GRID - 6, N_GRID - 1))
    # xi_m: the integral runs over [tmin, t), so the bottom end is affected
    bad_m = np.nonzero(~op["valid"]["xim_E"])[0]
    assert bad_m.tolist() == list(range(1, K + 1))

    for name in ("xip_E", "xip_B"):
        out = op["matrices"][name] @ xvec
        assert np.all(np.isnan(out[bad_p]))
        assert np.all(np.isfinite(out[op["valid"][name]]))
    for name in ("xim_E", "xim_B"):
        out = op["matrices"][name] @ xvec
        assert np.all(np.isnan(out[bad_m]))
        assert np.all(np.isfinite(out[op["valid"][name]]))

    # the ambiguous-mode integrals use the full range and are never affected
    assert np.all(op["valid"]["xip_amb"])
    assert np.all(op["valid"]["xim_amb"])


def test_operator_matches_reference(setup):
    """Operator vs. get_pure_EB_modes on a smooth input.

    ``dqags`` is only good to ~1e-6 relative on a grid this coarse (see
    ``validation/validate_operator.py``), so the tolerance here is loose; the
    tight statement lives in ``test_quadrature_weights_are_exact``.
    """
    theta, xip, xim = setup
    tmin, tmax = theta[0], theta[-1]
    ref = dict(
        zip(
            REF_ORDER,
            get_pure_EB_modes(
                theta, xip, xim, theta, xip, xim, tmin, tmax,
                parallel=False, pad_xim=True, pad_theta_max_decade=1.0,
                interp_order=K, epsabs=1e-12, epsrel=1e-12,
            ),
            strict=True,
        )
    )
    op = get_pure_EB_operator(
        theta, theta, tmin, tmax, outputs=OUTPUT_NAMES
    )
    xvec = np.concatenate([xip, xim])
    scale = {"xip": np.max(np.abs(xip)), "xim": np.max(np.abs(xim))}
    for name in OUTPUT_NAMES:
        ok = op["valid"][name]
        got = (op["matrices"][name] @ xvec)[ok]
        sc = scale["xip" if name.startswith("xip") else "xim"]
        assert_allclose(got, ref[name][ok], rtol=0.0, atol=1e-5 * sc)


def test_theta_eval_subset(setup):
    """theta_eval may be any subset of theta, in any position."""
    theta, xip, xim = setup
    tmin, tmax = theta[0], theta[-1]
    sel = np.array([10, 25, 40, 55])
    full = get_pure_EB_operator(theta, theta, tmin, tmax)
    part = get_pure_EB_operator(theta[sel], theta, tmin, tmax)
    for name in full["matrices"]:
        assert_allclose(
            part["matrices"][name], full["matrices"][name][sel], rtol=0.0
        )


def test_rejects_theta_eval_off_grid(setup):
    """theta_eval must sit on the grid."""
    theta, _, _ = setup
    with pytest.raises(ValueError):
        get_pure_EB_operator(
            theta * 1.01, theta, theta[0], theta[-1]
        )
