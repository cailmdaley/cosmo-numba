import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_QUAD"] = "1"

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmo_numba.math.integrate.quad import interp_quad


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


@pytest.mark.parametrize("k", [1, 3, 5])
def test_quad_undersupported_is_nan(k):
    """
    With 1 to k samples the degree-k interpolant is undetermined, so its
    integral is NaN.
    """
    h = 0.1
    for n in range(1, k + 1):
        res = interp_quad(
            0.0,
            (n - 1) * h,
            h,
            1.0 + np.arange(n, dtype=np.float64),
            -0.5 * h,
            (n - 0.5) * h,
            k=k,
            periodic=False,
            padding=True,
            extrap_dist=1,
            log_interp=False,
        )[0]
        assert np.isnan(res), f"n={n}: {res}"
