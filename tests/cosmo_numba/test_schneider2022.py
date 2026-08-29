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

    # Check that padding improve B-mode on xim
    norm_pad = np.linalg.norm(xim_B_model)
    norm_no_pad = np.linalg.norm(xim_B_model_no_pad)
    assert norm_pad < norm_no_pad

    # Check that B-mode on xip are unchanged
    assert_array_equal(
        xip_B_model,
        xip_B_model_no_pad,
    )


def test_underdetermined_edge_spline_is_nan():
    """
    An evaluation point whose masked integration support holds fewer than
    `interp_order + 1` nodes cannot define the spline. The integral must come
    back as NaN rather than as the large finite value the ghost-cell
    extrapolation used to produce.
    """
    interp_order = 5

    theta_int = np.geomspace(0.5, 250.0, 100)
    tmin, tmax = theta_int[0], theta_int[-1]
    xip_int = 1e-4 * (theta_int / tmin) ** -1.5
    xim_int = 1e-4 * (theta_int / tmin) ** -2.0

    # Only the last 3 integration nodes lie above this edge point, fewer than
    # the interp_order + 1 = 6 needed by the quintic spline.
    t_edge = theta_int[-4] * (1 + 1e-6)
    n_sup = np.sum((t_edge < theta_int) & (theta_int <= tmax))
    assert n_sup < interp_order + 1

    theta = np.array([theta_int[50], t_edge])
    xip = 1e-4 * (theta / tmin) ** -1.5
    xim = 1e-4 * (theta / tmin) ** -2.0

    out = get_pure_EB_modes(
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
        interp_order=interp_order,
    )
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = out

    # The unsupported edge point fails loudly...
    assert np.isnan(xip_E[-1])
    assert np.isnan(xip_B[-1])

    # ...while the well-supported point stays finite and sane, and nothing
    # anywhere comes back as finite garbage.
    assert np.isfinite(xip_E[0])
    for arr in out:
        finite = arr[np.isfinite(arr)]
        assert np.all(np.abs(finite) < 1e10)


def test_zero_width_support_stays_zero():
    """
    An empty integration support (n_sup == 0) is a zero-width interval, not an
    underdetermined spline: it must keep returning 0.0, not NaN.
    """
    theta_int = np.geomspace(0.5, 250.0, 1000)
    tmin, tmax = theta_int[0], theta_int[-1]
    xip_int = 1e-4 * (theta_int / tmin) ** -1.5
    xim_int = 1e-4 * (theta_int / tmin) ** -2.0

    # t == tmax for the xi_+ integral and t == tmin for the xi_- integral both
    # give an empty mask.
    theta = np.array([tmin, tmax])
    xip = 1e-4 * (theta / tmin) ** -1.5
    xim = 1e-4 * (theta / tmin) ** -2.0

    out = get_pure_EB_modes(
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
        interp_order=5,
    )
    for arr in out:
        assert np.all(np.isfinite(arr))
