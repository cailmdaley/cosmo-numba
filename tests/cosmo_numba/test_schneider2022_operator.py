"""Tests for the fixed-quadrature pure E/B operator.

The reference for correctness is the exact integral of the interpolant that
`get_pure_EB_modes` builds, computed here by brute force: composite
Gauss-Legendre quadrature of `nb_interp1d` with a break at every grid cell.
The adaptive `dqags` path integrates the same interpolant only to its
tolerance, so it is compared against with a loose tolerance.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmo_numba.B_modes.schneider2022 import (
    PURE_EB_OUTPUTS,
    get_pure_EB_covariance,
    get_pure_EB_modes,
    get_pure_EB_operator,
)
from cosmo_numba.B_modes.schneider2022_nb import H_m, H_p, K_m, K_p
from cosmo_numba.B_modes.schneider2022_operator import (
    _QuadRule,
    _quad_weights,
)
from cosmo_numba.math.interpolate.interpolate_1D import nb_interp1d
from cosmo_numba.math.utils import extend_log_grid

DATA_DIR = os.path.join(Path(__file__).parent, "data")
K = 5


def brute_integral(f, x_start, x_end, h, a, b, n_gauss=24):
    """Integral over theta in [a, b] of the interpolant of `f` (log grid)."""
    if f.size < K + 1:
        return np.nan
    xa, xb = np.log(a), np.log(b)
    sign = 1.0
    if xb < xa:
        xa, xb, sign = xb, xa, -1.0
    itp = nb_interp1d(
        x_start, x_end, h, np.ascontiguousarray(f), K, False, True, 1, False
    )
    brk = np.concatenate([[xa, xb], x_start + h * np.arange(-1, f.size + 1)])
    brk = np.unique(brk[(brk >= xa) & (brk <= xb)])
    brk = brk[np.concatenate([[True], np.diff(brk) > 1e-12])]
    nodes, wts = np.polynomial.legendre.leggauss(n_gauss)
    lo, hi = brk[:-1, None], brk[1:, None]
    xq = (0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)).ravel()
    wq = (0.5 * (hi - lo) * wts).ravel()
    # Clamp strictly inside the extrapolation range: the boundary value is
    # the limit from inside the outermost cell.
    eps = 1e-12 * h
    xe = np.clip(xq, x_start - h + eps, x_end + h - eps)
    return sign * float(
        np.sum(wq * np.exp(xq) * itp.eval(np.ascontiguousarray(xe)))
    )


def brute_pure_EB(
    theta, xip, xim, theta_int, xip_int, xim_int, tmin, tmax, pad_xim
):
    """`get_pure_EB_modes` with every integral done by `brute_integral`."""
    d = np.mean(np.diff(np.log(theta_int)))
    tb, B = (tmax + tmin) / 2, (tmax - tmin) / (tmax + tmin)
    if pad_xim:
        theta_x = extend_log_grid(theta, 0.0, 1.0)[0]
        ti_x = extend_log_grid(theta_int, 0.0, 1.0)[0]
        xip_x = np.zeros(ti_x.size)
        xim_x = np.zeros(ti_x.size)
        xip_x[: theta_int.size] = xip_int
        xim_x[: theta_int.size] = xim_int
        tmax_x = theta_x[-1]
    else:
        ti_x, xip_x, xim_x, tmax_x = theta_int, xip_int, xim_int, tmax
    tbx, Bx = (tmax_x + tmin) / 2, (tmax_x - tmin) / (tmax_x + tmin)

    def window(grid, lo, hi):
        g = grid[(grid >= lo) & (grid <= hi)]
        x = np.log(g)
        return (grid >= lo) & (grid <= hi), g, x[0], x[-1], np.diff(x).mean()

    mS, gS, xs, xe, hS = window(theta_int, tmin, tmax)
    mV, gV, xsV, xeV, hV = window(ti_x, tmin, tmax_x)
    out = np.zeros((6, theta.size))
    for i, t in enumerate(theta):
        m = (t < theta_int) & (theta_int <= tmax)
        ip = 0.0
        if m.any():
            g = theta_int[m]
            f = xim_int[m] / g * (4 - 12 * t**2 / g**2)
            ip = brute_integral(f, np.log(g[0]), np.log(g[-1]), d, t, tmax)
        m = (t > ti_x) & (ti_x >= tmin)
        im = 0.0
        if m.any():
            g = ti_x[m]
            f = xip_x[m] * g / t**2 * (4 - 12 * g**2 / t**2)
            im = brute_integral(f, np.log(g[0]), np.log(g[-1]), d, tmin, t)
        Sp = brute_integral(
            gS * xip_int[mS] * H_p(t, gS, tb, B) / tb**2,
            xs,
            xe,
            hS,
            tmin,
            tmax,
        )
        Sm = brute_integral(
            xim_int[mS] * H_m(t, gS, tb, B) / gS, xs, xe, hS, tmin, tmax
        )
        Vp = brute_integral(
            gV * xip_x[mV] * K_p(t, gV, tbx, Bx) / tbx**2,
            xsV,
            xeV,
            hV,
            tmin,
            tmax_x,
        )
        Vm = brute_integral(
            gV * xim_x[mV] * K_m(t, gV, tbx, Bx) / tbx**2,
            xsV,
            xeV,
            hV,
            tmin,
            tmax_x,
        )
        p, q = xip[i], xim[i]
        out[:, i] = (
            0.5 * (p + q + ip) - 0.5 * (Sp + Sm),
            0.5 * (p + q + im) - 0.5 * (Vp + Vm),
            0.5 * (p - q - ip) - 0.5 * (Sp - Sm),
            0.5 * (p - q + im) - 0.5 * (Vp - Vm),
            Sp,
            Vm,
        )
    return out


def assert_array_nan_equal(a, b):
    """NaN at the same places."""
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


@pytest.fixture(scope="module")
def ccl():
    """CCL xi_pm on a 20-point evaluation grid and a 1000-point fine grid."""
    coarse = np.load(os.path.join(DATA_DIR, "ccl_xi_pm_0.5_250_20.npy"))
    fine = np.load(os.path.join(DATA_DIR, "ccl_xi_pm_0.5_800_1_000.npy"))
    return coarse, fine


def _geometry(name, ccl):
    (theta, xip, xim), (ti, xpi, xmi) = ccl
    if name == "two grids":
        # theta is off the theta_int nodes everywhere.
        return theta, xip, xim, ti, xpi, xmi, theta.min(), theta.max()
    if name == "single grid":
        t, p, m = ti[::10], xpi[::10], xmi[::10]
        return t, p, m, t, p, m, t[0], t[-1]
    if name == "bounds beyond grid":
        # tmin and tmax lie several cells outside theta_int, where the
        # interpolant is clamped to its boundary value.
        s = slice(100, 600, 4)
        return (
            theta[2:-2],
            xip[2:-2],
            xim[2:-2],
            ti[s],
            xpi[s],
            xmi[s],
            theta.min(),
            theta.max(),
        )
    raise ValueError(name)


def test_quad_weights_are_exact():
    """The fixed rule integrates the interpolant exactly, for any bounds."""
    theta = np.geomspace(1.0, 40.0, 60)
    x = np.log(theta)
    h = np.diff(x).mean()
    f = np.random.default_rng(1).standard_normal(60)
    rule = _QuadRule(K, 16)
    bounds = [
        (theta[0], theta[-1]),
        (theta[3] * 1.01, theta[-4] * 0.99),
        (theta[0] * np.exp(-0.5 * h), theta[-1] * np.exp(0.7 * h)),
        (0.5, 100.0),
        (theta[10] * 1.003, theta[10] * 1.004),
        (theta[-4], theta[2]),
    ]
    for a, b in bounds:
        w = _quad_weights(rule, 60, x[0], x[-1], h, np.log(a), np.log(b))
        ref = brute_integral(f, x[0], x[-1], h, a, b)
        assert_allclose(w @ f, ref, rtol=1e-11, err_msg=f"[{a}, {b}]")


@pytest.mark.parametrize(
    "geometry", ["two grids", "single grid", "bounds beyond grid"]
)
@pytest.mark.parametrize("pad_xim", [True, False])
def test_operator_matches_brute_force(ccl, geometry, pad_xim):
    """Every output equals the exact integral of the reference interpolant."""
    args = _geometry(geometry, ccl)
    ref = brute_pure_EB(*args, pad_xim=pad_xim)
    got = get_pure_EB_modes(*args, pad_xim=pad_xim, quadrature="fixed")
    for j, (g, r) in enumerate(zip(got, ref, strict=True)):
        assert_array_nan_equal(g, r)
        scale = np.max(np.abs(args[1] if j % 2 == 0 else args[2]))
        assert_allclose(g, r, rtol=0, atol=1e-9 * scale, equal_nan=True)


def test_fixed_matches_adaptive(ccl):
    """On a smooth input both quadratures agree to the dqags tolerance."""
    args = _geometry("two grids", ccl)
    fixed = get_pure_EB_modes(*args, quadrature="fixed")
    adaptive = get_pure_EB_modes(*args, epsabs=1e-12, epsrel=1e-12)
    for j, (f, a) in enumerate(zip(fixed, adaptive, strict=True)):
        scale = np.max(np.abs(args[1] if j % 2 == 0 else args[2]))
        assert_allclose(f, a, rtol=0, atol=1e-6 * scale)


def test_reconstruction_identity(ccl):
    """E + B + amb returns the input exactly."""
    args = _geometry("two grids", ccl)
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = get_pure_EB_modes(
        *args, quadrature="fixed"
    )
    assert_allclose(xip_E + xip_B + xip_amb, args[1], rtol=1e-12, atol=0)
    assert_allclose(xim_E - xim_B + xim_amb, args[2], rtol=1e-12, atol=0)


def test_operator_is_linear_and_converged(ccl):
    """Matrix form equals the call, and does not depend on n_gauss."""
    theta, xip, xim, ti, xpi, xmi, tmin, tmax = _geometry("two grids", ccl)
    ops = get_pure_EB_operator(theta, ti, tmin, tmax)
    ops_hi = get_pure_EB_operator(theta, ti, tmin, tmax, n_gauss=32)
    rng = np.random.default_rng(3)
    y = [rng.standard_normal(2 * theta.size + 2 * ti.size) for _ in range(2)]

    def call(v):
        n = theta.size
        return get_pure_EB_modes(
            theta,
            v[:n],
            v[n : 2 * n],
            ti,
            v[2 * n : 2 * n + ti.size],
            v[2 * n + ti.size :],
            tmin,
            tmax,
            quadrature="fixed",
        )

    for op, op_hi, c1, c2, c12 in zip(
        ops,
        ops_hi,
        call(y[0]),
        call(y[1]),
        call(2 * y[0] - y[1]),
        strict=True,
    ):
        assert op.shape == (theta.size, 2 * theta.size + 2 * ti.size)
        assert_allclose(op_hi, op, rtol=1e-12, atol=1e-14 * np.abs(op).max())
        assert_allclose(op @ y[0], c1, rtol=1e-14)
        assert_allclose(c12, 2 * c1 - c2, rtol=1e-10, atol=1e-12)


def test_local_from_int(ccl):
    """Single-vector form for theta on the theta_int nodes."""
    _, (ti, xpi, xmi) = ccl
    idx = np.arange(40, 900, 37)
    theta = ti[idx]
    tmin, tmax = ti[0], ti[-1]
    full = get_pure_EB_operator(theta, ti, tmin, tmax)
    single = get_pure_EB_operator(theta, ti, tmin, tmax, local_from_int=True)
    v_full = np.concatenate([xpi[idx], xmi[idx], xpi, xmi])
    v_single = np.concatenate([xpi, xmi])
    scale = np.abs(v_single).max()
    for f, s in zip(full, single, strict=True):
        assert s.shape == (theta.size, 2 * ti.size)
        assert_allclose(s @ v_single, f @ v_full, rtol=0, atol=1e-13 * scale)
    with pytest.raises(ValueError, match="node of theta_int"):
        get_pure_EB_operator(
            theta * 1.001, ti, tmin, tmax, local_from_int=True
        )


def test_edge_policy(ccl):
    """Too few nodes in a theta-dependent support give NaN, never garbage."""
    theta, xip, xim, *_ = _geometry("single grid", ccl)
    out = get_pure_EB_modes(
        theta,
        xip,
        xim,
        theta,
        xip,
        xim,
        theta[0],
        theta[-1],
        quadrature="fixed",
    )
    n = theta.size
    # (t, tmax] runs out of nodes at the top, [tmin, t) at the bottom; an
    # empty support (t at the bound) is a zero-width integral.
    bad_p = list(range(n - K - 1, n - 1))
    bad_m = list(range(1, K + 1))
    for j in (0, 2):
        assert np.flatnonzero(np.isnan(out[j])).tolist() == bad_p
    for j in (1, 3):
        assert np.flatnonzero(np.isnan(out[j])).tolist() == bad_m
    for j in range(6):
        finite = out[j][np.isfinite(out[j])]
        assert np.all(np.abs(finite) < 1e2 * np.abs(xip).max())
    assert np.all(np.isfinite(out[4])) and np.all(np.isfinite(out[5]))


def test_unknown_quadrature(ccl):
    """Only 'adaptive' and 'fixed' are accepted."""
    args = _geometry("two grids", ccl)
    with pytest.raises(ValueError, match="quadrature"):
        get_pure_EB_modes(*args, quadrature="gauss")


@pytest.fixture(scope="module")
def cov_setup():
    """Coarse grid, smooth xi_pm and a correlated Gaussian xi_pm covariance."""
    theta_int = np.geomspace(1.0, 200.0, 50)
    xip = 2e-5 * (theta_int / 10.0) ** -0.65
    xim = 6e-6 * (theta_int / 10.0) ** -1.35
    x = np.log(theta_int)
    corr = np.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.3) ** 2)
    sig = np.concatenate([0.3 * xip, 0.3 * xim])
    blocks = np.block([[corr, 0.5 * corr], [0.5 * corr, corr]])
    cov = sig[:, None] * (blocks + 1e-3 * np.eye(100)) * sig[None, :]
    theta = theta_int[8:42:3]
    return theta, theta_int, xip, xim, cov


def test_covariance_matches_monte_carlo(cov_setup):
    """M C M^T matches the scatter of Gaussian draws through the transform.

    The draws are transformed with the two-vector operator (local term from
    the evaluation-grid values), independently of the single-vector operator
    the covariance is built from.
    """
    theta, ti, xip, xim, cov = cov_setup
    tmin, tmax = ti[0], ti[-1]
    op = get_pure_EB_operator(theta, ti, tmin, tmax, local_from_int=True)
    analytic = get_pure_EB_covariance(op, cov, outputs=PURE_EB_OUTPUTS)

    n_draw = 20000
    rng = np.random.default_rng(5)
    draws = rng.multivariate_normal(np.concatenate([xip, xim]), cov, n_draw)
    idx = np.searchsorted(ti, theta)
    n = ti.size
    data = np.hstack([draws[:, idx], draws[:, n + idx], draws])
    op4 = get_pure_EB_operator(theta, ti, tmin, tmax)
    modes = np.hstack([data @ m.T for m in op4])
    empirical = np.cov(modes, rowvar=False)

    sd = np.sqrt(np.diag(analytic))
    assert_allclose(np.sqrt(np.diag(empirical)), sd, rtol=0.05)
    # Correlation coefficients agree to a few times 1/sqrt(n_draw).
    r_an = analytic / np.outer(sd, sd)
    r_mc = empirical / np.outer(sd, sd)
    assert np.max(np.abs(r_mc - r_an)) < 5 / np.sqrt(n_draw)


def test_covariance_structure(cov_setup):
    """Symmetric PSD, documented block order, and every input form."""
    theta, ti, xip, xim, cov = cov_setup
    tmin, tmax = ti[0], ti[-1]
    n = theta.size
    op = get_pure_EB_operator(theta, ti, tmin, tmax, local_from_int=True)
    full = get_pure_EB_covariance(op, cov, outputs=PURE_EB_OUTPUTS)
    assert full.shape == (6 * n, 6 * n)
    assert_allclose(full, full.T, rtol=0, atol=0)
    eig = np.linalg.eigvalsh(full)
    assert eig.min() > -1e-10 * eig.max()

    # Blocks follow the order of `outputs`.
    sub = get_pure_EB_covariance(op, cov, outputs=("xim_B", "xip_E"))
    i_mB = PURE_EB_OUTPUTS.index("xim_B")
    i_pE = PURE_EB_OUTPUTS.index("xip_E")
    assert_allclose(
        sub[:n, n:], full[i_mB * n : (i_mB + 1) * n, i_pE * n : (i_pE + 1) * n]
    )
    assert_allclose(sub[:n, n:], op[i_mB] @ cov @ op[i_pE].T)

    # Two-vector form: the same covariance once the evaluation-grid values
    # are the integration-grid values at the same nodes.
    op4 = get_pure_EB_operator(theta, ti, tmin, tmax)
    idx = np.searchsorted(ti, theta)
    sel = np.zeros((2 * n + 2 * ti.size, 2 * ti.size))
    sel[np.arange(n), idx] = 1
    sel[n + np.arange(n), ti.size + idx] = 1
    sel[2 * n :, :] = np.eye(2 * ti.size)
    assert_allclose(
        get_pure_EB_covariance(
            op4, sel @ cov @ sel.T, outputs=PURE_EB_OUTPUTS
        ),
        full,
        rtol=1e-12,
        atol=1e-12 * np.abs(full).max(),
    )


def test_covariance_rejects_bad_input(cov_setup):
    """Unknown names, wrong shapes and undefined rows raise."""
    theta, ti, xip, xim, cov = cov_setup
    op = get_pure_EB_operator(theta, ti, ti[0], ti[-1], local_from_int=True)
    with pytest.raises(ValueError, match="unknown outputs"):
        get_pure_EB_covariance(op, cov, outputs=("xip_EE",))
    with pytest.raises(ValueError, match="shape"):
        get_pure_EB_covariance(op, cov[:-2, :-2])
    edge = get_pure_EB_operator(
        ti[-3:], ti, ti[0], ti[-1], local_from_int=True
    )
    with pytest.raises(ValueError, match="undefined"):
        get_pure_EB_covariance(edge, cov)
