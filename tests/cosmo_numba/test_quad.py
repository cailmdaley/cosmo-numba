import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_QUAD"] = "1"

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmo_numba.math.integrate.quad import interp_quad, interp_quad_weights
from cosmo_numba.math.interpolate.interpolate_1D import nb_interp1d


def test_quad_typical():
    """
    This test comes from Scipy testing suite.
    The value of atol is set to 1e-10 instead of err because there is also the
    error coming from the interpolation.
    """

    def myfunc(x):  # Bessel function integrand
        n = 2
        z = 1.8
        return np.cos(n * x - z * np.sin(x)) / np.pi

    x_start = 1e-4
    x_end = np.pi
    step = 0.001

    int_atol = int_rtol = 1.5e-8

    res, err, success = interp_quad(
        x_start,
        x_end,
        step,
        myfunc(np.arange(x_start, x_end, step)),
        0.0,
        np.pi,
        k=5,
        periodic=False,
        extrap_dist=1.0,
        log_interp=False,
        epsabs=int_atol,
        epsrel=int_rtol,
    )

    assert success
    assert_allclose(res, 0.30614353532540296487, atol=max(err, 1e-10), rtol=0)


def test_quad_singular():
    """
    This test comes from Scipy testing suite.
    The value of atol is set to 1e-10 instead of err because there is also the
    error coming from the interpolation.
    """

    def myfunc(x):
        if 0 < x < 2.5:
            return np.sin(x)
        elif 2.5 <= x <= 5.0:
            return np.exp(-x)
        else:
            return 0.0

    x_start = 0.0
    x_end = 10.0
    step = 0.001
    res, err, success = interp_quad(
        x_start,
        x_end,
        step,
        np.array([myfunc(x) for x in np.arange(x_start, x_end, step)]),
        x_start,
        x_end,
        k=3,
        periodic=False,
        extrap_dist=0.0,
        log_interp=False,
        epsabs=1.5e-8,
        epsrel=1.5e-8,
    )

    assert success
    assert_allclose(
        res,
        1 - np.cos(2.5) + np.exp(-2.5) - np.exp(-5.0),
        atol=max(err, 1e-10),
        rtol=0,
    )


def test_quad_log_spacing():
    """
    This test is for log-spaced grids.
    """

    def myfunc(x):
        return x ** (0.3) * np.exp(-x / 100.0)

    x_start = 1e-3
    x_end = 1e3
    n_points_log = 10_000
    xx_log = np.logspace(np.log10(x_start), np.log10(x_end), n_points_log)
    dln = np.median(np.diff(np.log(xx_log)))
    res, err, success = interp_quad(
        np.log(x_start),
        np.log(x_end),
        dln,
        myfunc(xx_log),
        1,
        1e3,
        k=5,
        periodic=False,
        extrap_dist=1.0,
        log_interp=True,
        epsabs=1.5e-8,
        epsrel=1.5e-8,
    )

    assert success
    assert_allclose(res, 356.48754262594207, atol=max(err, 1e-10), rtol=0)


def brute_integral(fx, x_start, x_end, h, a, b, k, e, log_interp):
    """Integral of the interpolant of `fx` from `a` to `b`, by brute force.

    Composite Gauss-Legendre quadrature of `nb_interp1d`, with a break at
    every cell edge and clamping bound, so that each piece is smooth.
    """
    itp = nb_interp1d(x_start, x_end, h, fx, k, False, True, e, False)
    xa, xb = (np.log(a), np.log(b)) if log_interp else (a, b)
    sign = 1.0
    if xb < xa:
        xa, xb, sign = xb, xa, -1.0
    edges = x_start + h * np.arange(-e, fx.size + e)
    brk = np.concatenate([edges, [itp.lb, itp.ub, xa, xb]])
    brk = np.unique(brk[(brk >= xa) & (brk <= xb)])
    nodes, wts = np.polynomial.legendre.leggauss(30)
    lo, hi = brk[:-1, None], brk[1:, None]
    xq = (0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)).ravel()
    wq = (0.5 * (hi - lo) * wts).ravel()
    if log_interp:
        wq = wq * np.exp(xq)
    return sign * np.sum(wq * itp.eval(xq))


@pytest.mark.parametrize("log_interp", [True, False])
@pytest.mark.parametrize("k, e", [(1, 0), (3, 0), (5, 1), (9, 2)])
def test_quad_weights_are_exact(log_interp, k, e):
    """
    The fixed rule integrates the interpolant exactly, for any bounds.
    """
    n = 60
    if log_interp:
        theta = np.geomspace(1.0, 40.0, n)
        x = np.log(theta)
    else:
        x = theta = np.linspace(1.0, 40.0, n)
    h = np.mean(np.diff(x))
    fx = np.random.default_rng(1).standard_normal(n)
    up = np.exp(h) if log_interp else h
    bounds = [
        (theta[0], theta[-1]),
        (theta[3] * 1.01, theta[-4] * 0.99),
        # Inside the extrapolated cells, and beyond the clamping bounds.
        (theta[0] - 0.5 * up, theta[-1] + 0.7 * up),
        (0.5, 100.0),
        # Within one cell, and reversed.
        (theta[10] * 1.003, theta[10] * 1.004),
        (theta[-4], theta[2]),
    ]
    for a, b in bounds:
        w = interp_quad_weights(
            x[0], x[-1], h, n, a, b, k, True, e, log_interp
        )
        ref = brute_integral(fx, x[0], x[-1], h, a, b, k, e, log_interp)
        assert_allclose(w @ fx, ref, rtol=1e-12, atol=1e-12, err_msg=(a, b))


def test_quad_weights_match_interp_quad():
    """
    On a smooth function, interp_quad converges to the same integral.
    """
    x = np.log(np.geomspace(1e-3, 1e3, 2000))
    dln = np.mean(np.diff(x))
    fx = np.exp(x) ** 0.3 * np.exp(-np.exp(x) / 100.0)
    w = interp_quad_weights(
        x[0], x[-1], dln, x.size, 1.0, 1e3, k=5, extrap_dist=1, log_interp=True
    )
    res, err, _ = interp_quad(
        x[0],
        x[-1],
        dln,
        fx,
        1.0,
        1e3,
        k=5,
        periodic=False,
        padding=True,
        extrap_dist=1.0,
        log_interp=True,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    assert_allclose(w @ fx, res, rtol=1e-9)


def test_quad_weights_undetermined():
    """
    Fewer than k + 1 samples leave the interpolant undetermined.
    """
    w = interp_quad_weights(0.0, 0.4, 0.1, 5, 0.0, 0.4, k=5, extrap_dist=1)
    assert w.shape == (5,)
    assert np.all(np.isnan(w))


def test_quad_weights_unsupported_order():
    """
    Even and out-of-range orders have no stencil.
    """
    for k in (0, 2, 4, 11):
        with pytest.raises(ValueError):
            interp_quad_weights(0.0, 1.0, 0.1, 11, 0.0, 1.0, k=k)
