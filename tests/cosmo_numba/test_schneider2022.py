import os

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_SCHNEIDER2022"] = "1"

from pathlib import Path

import numpy as np
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
