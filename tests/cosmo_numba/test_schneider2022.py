import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_SCHNEIDER2022"] = "1"

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

DATA_DIR = os.path.join(Path(__file__).parent, "data")


def test_pure_eb():
    """
    Test the pure E/B-mode decomposition from Schneider et al. (2022).
    The test make use of pre-computed xi_pm from CCL.
    """
    # Load pre-computed xi_pm from CCL
    theta, xip_model, xim_model = np.load(
        os.path.join(
            DATA_DIR,
            "ccl_xi_pm_0.5_250_20.npy",
        ),
    )
    theta_int, xip_model_int, xim_model_int = np.load(
        os.path.join(
            DATA_DIR,
            "ccl_xi_pm_0.5_800_1_000.npy",
        ),
    )

    # Serial computation
    (
        xip_E_model,
        xim_E_model,
        xip_B_model,
        xim_B_model,
        xip_amb_model,
        xim_amb_model,
    ) = get_pure_EB_modes(
        theta,
        xip_model,
        xim_model,
        theta_int,
        xip_model_int,
        xim_model_int,
        np.min(theta),
        np.max(theta),
        parallel=False,
        pad_xim=True,
        interp_order=5,
    )

    xip_reconstructed = xip_E_model + xip_B_model + xip_amb_model
    xim_reconstructed = xim_E_model - xim_B_model + xim_amb_model

    assert xip_reconstructed.size == xip_model.size
    assert xim_reconstructed.size == xim_model.size

    assert_allclose(
        xip_reconstructed,
        xip_model,
        atol=1e-12,
        rtol=1e-12,
    )
    assert_allclose(
        xim_reconstructed,
        xim_model,
        atol=1e-12,
        rtol=1e-12,
    )

    # Parallel computation
    (
        xip_E_model,
        xim_E_model,
        xip_B_model,
        xim_B_model,
        xip_amb_model,
        xim_amb_model,
    ) = get_pure_EB_modes(
        theta,
        xip_model,
        xim_model,
        theta_int,
        xip_model_int,
        xim_model_int,
        np.min(theta),
        np.max(theta),
        parallel=True,
        pad_xim=True,
        interp_order=5,
    )
    xip_reconstructed = xip_E_model + xip_B_model + xip_amb_model
    xim_reconstructed = xim_E_model - xim_B_model + xim_amb_model
    assert_allclose(
        xip_reconstructed,
        xip_model,
        atol=1e-12,
        rtol=1e-12,
    )
    assert_allclose(
        xim_reconstructed,
        xim_model,
        atol=1e-12,
        rtol=1e-12,
    )

    # Check no padding
    (
        xip_E_model_no_pad,
        xim_E_model_no_pad,
        xip_B_model_no_pad,
        xim_B_model_no_pad,
        xip_amb_model_no_pad,
        xim_amb_model_no_pad,
    ) = get_pure_EB_modes(
        theta,
        xip_model,
        xim_model,
        theta_int,
        xip_model_int,
        xim_model_int,
        np.min(theta),
        np.max(theta),
        parallel=True,
        pad_xim=False,
        interp_order=5,
    )
    xip_reconstructed_no_pad = (
        xip_E_model_no_pad + xip_B_model_no_pad + xip_amb_model_no_pad
    )
    xim_reconstructed_no_pad = (
        xim_E_model_no_pad - xim_B_model_no_pad + xim_amb_model_no_pad
    )
    assert_allclose(
        xip_reconstructed_no_pad,
        xip_model,
        atol=1e-12,
        rtol=1e-12,
    )
    assert_allclose(
        xim_reconstructed_no_pad,
        xim_model,
        atol=1e-12,
        rtol=1e-12,
    )

    # Without padding the decomposition on [tmin, tmax] is exact, so the
    # pure-E model leaves only interpolation error in the xim B mode
    assert np.max(np.abs(xim_B_model_no_pad / xim_model)) < 1e-3

    # Check that B-mode on xip are unchanged
    assert_array_equal(
        xip_B_model,
        xip_B_model_no_pad,
    )


def _pure_eb(theta, theta_int, xipm_int, tmin, tmax):
    """Pure E/B modes at `theta` (nodes of `theta_int`) from one vector.

    The local terms at `theta` are read from the same `xipm_int =
    [xip_int, xim_int]` vector that is integrated, so the transform sees
    one consistent data vector. Returns an array of shape (6, len(theta)).
    """
    n = len(theta_int)
    xip_int, xim_int = xipm_int[:n], xipm_int[n:]
    idx = np.searchsorted(theta_int, theta)
    return np.array(
        get_pure_EB_modes(
            theta,
            xip_int[idx],
            xim_int[idx],
            theta_int,
            xip_int,
            xim_int,
            tmin,
            tmax,
            parallel=False,
            pad_xim=True,
            interp_order=5,
        )
    )


@pytest.mark.xfail(
    strict=True,
    reason="the adaptive quadrature (dqags) chooses its subdivision from the "
    "data, so the modes are not linear in xi_pm and noise rectifies",
)
def test_pure_eb_linearity():
    """
    Test that the pure E/B transform is linear in the input xi_pm.

    The pure E/B modes are linear functionals of xi_pm (Schneider et al.
    2022), and the code integrates a local polynomial interpolant that is
    itself linear in the data. Linearity is what makes the covariance of
    the modes M C M^T for a xi_pm covariance C, and what keeps noise in
    xi_pm from rectifying into a mean B mode on B-mode-free data.

    With theory xi_pm (CCL, pure E) and one Gaussian noise draw n from the
    CosmoCov LSST covariance scaled by alpha (UNIONS-like noise), the
    antithetic residual

        r = T(xi + n) + T(xi - n) - 2 T(xi)

    vanishes to round-off for a linear T: it isolates the even part of
    the transform's response to the noise. It is compared, per output bin,
    to the noise scale given by the RMS over the draws of the odd part
    (T(xi + n) - T(xi - n)) / 2. A quadrature on a fixed partition passes
    by about six orders of magnitude.
    """
    theta_ccl, xip_ccl, xim_ccl = np.load(
        os.path.join(DATA_DIR, "ccl_xi_pm_0.5_250_10_000.npy"),
    )
    cov = np.load(
        os.path.join(DATA_DIR, "cosmocov_cov_lsst_xipm_0.5_250_1000.npy"),
    )

    # Grid of the CosmoCov covariance (as in test_cosebis.py)
    theta_ = np.logspace(np.log10(0.5), np.log10(250.0), 1_000 + 1)
    theta_int = np.mean([theta_[:-1], theta_[1:]], axis=0)
    xipm = np.concatenate(
        [
            np.interp(np.log(theta_int), np.log(theta_ccl), xip_ccl),
            np.interp(np.log(theta_int), np.log(theta_ccl), xim_ccl),
        ]
    )

    # 20 nodes of theta_int, log-spaced inside the grid
    idx = np.round(np.linspace(0, len(theta_int) - 1, 22)).astype(int)
    theta = theta_int[idx[1:-1]]
    tmin, tmax = theta[0], theta[-1]

    # sqrt(diag C_UNIONS / diag C_LSST) in the shape-noise regime, theta < 10'
    alpha = 12.6
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(42)
    T0 = _pure_eb(theta, theta_int, xipm, tmin, tmax)
    odd, even = [], []
    for _ in range(4):
        noise = alpha * chol @ rng.standard_normal(len(xipm))
        T_plus = _pure_eb(theta, theta_int, xipm + noise, tmin, tmax)
        T_minus = _pure_eb(theta, theta_int, xipm - noise, tmin, tmax)
        odd.append(0.5 * (T_plus - T_minus))
        even.append(T_plus + T_minus - 2 * T0)
    sigma = np.sqrt(np.mean(np.square(odd), axis=0))
    residual = np.max(np.abs(np.array(even) / sigma), axis=(0, 2))

    names = ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
    report = ", ".join(
        f"{name}={value:.2e}"
        for name, value in zip(names, residual, strict=True)
    )
    assert np.all(residual < 1e-6), (
        f"max |T(xi+n) + T(xi-n) - 2T(xi)| / sigma per output: {report}"
    )


def test_pure_eb_no_leakage_narrow_window():
    """
    Test that pure-E input gives no B modes on a narrow, unpadded window.

    xi_+ is a Gaussian, exp(-x) with x = theta^2 / (2 S^2), and xi_- is its
    pure-E partner,

        xi_-(t) = xi_+(t) + int_0^t dp p / t^2 xi_+(p) (4 - 12 p^2 / t^2),

    which has the closed form

        xi_- = exp(-x) + 2 (1 - exp(-x)) / x
               - 6 (1 - (1 + x) exp(-x)) / x^2.

    With pad_xim=False the xi_- B mode depends on the xi_+ -> xi_- kernel
    K_+ (Eq. 51) over the whole window, so this checks its argument order.
    """
    S = 20.0
    tmin, tmax = 12.0, 83.0

    def xipm(theta):
        x = theta**2 / (2 * S**2)
        e = np.exp(-x)
        xim = e + 2 * (1 - e) / x - 6 * (1 - (1 + x) * e) / x**2
        return e, xim

    log_edges = np.linspace(np.log(tmin), np.log(tmax), 401)
    theta_int = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
    theta = np.geomspace(15.0, 70.0, 6)

    xip, xim = xipm(theta)
    xip_int, xim_int = xipm(theta_int)
    modes = get_pure_EB_modes(
        theta,
        xip,
        xim,
        theta_int,
        xip_int,
        xim_int,
        tmin,
        tmax,
        parallel=False,
        pad_xim=False,
    )

    # Peak |xi_+| is 1: B modes are leakage relative to the signal.
    assert_allclose(modes[2], 0, atol=1e-6)
    assert_allclose(modes[3], 0, atol=1e-6)
