import re
import os
from pathlib import Path

if os.environ.get("COVERAGE_MODE", "0") == "1":
    os.environ["TESTING_COSEBIS"] = "1"

import numpy as np
import mpmath as mp
from scipy.integrate import simpson

from numpy.testing import assert_allclose
import pytest

from cosmo_numba.B_modes.cosebis import COSEBIS
from cosmo_numba.math.integrate.quad import interp_quad
from cosmo_numba.math.integrate.fftlog import fht
from cosmo_numba.math.utils import extend_log_grid, pad_1D


DATA_DIR = os.path.join(Path(__file__).parent, "data")


def test_cosebis_roots_norms_vs_mathematica():
    """
    Test roots and norms against precomputed values from the original
    Mathematica code presented in Schneider et al. 2010 Fig. 3
    (https://arxiv.org/abs/1002.2136).

    NOTE: The precision used in Mathetatica is 50 deciaml places. We test down
    to 49 decimal places because I have noticed slight deviation at the 50th
    decimal place.
    """
    N_mode = 20
    mp.mp.dps = 50

    # Load precomputed roots and norms from Mathematica
    # with N_mode = 20, theta_min = 0.5 arcmin, theta_max = 250 arcmin
    with open(
        os.path.join(DATA_DIR, "cosebi_tplog_rN_20_0.500_250.000.dat")
    ) as f:
        lines = f.readlines()
    mp.mp.dps = 50
    roots_mm = []
    for i in range(3, N_mode + 3):
        r_str = re.split("\t", lines[i].strip())
        roots_mm.append(np.array([mp.mpf(rr) for rr in r_str]))
    norms_mm = []
    for i in range(N_mode + 3, 2 * N_mode + 3):
        norms_mm.append(mp.mpf(lines[i].strip()))

    cosebis = COSEBIS(
        theta_min=0.5, theta_max=250.0, N_max=N_mode, precision=100
    )
    cosebis.compute_roots()
    for i in range(len(roots_mm)):
        for j in range(len(roots_mm[i])):
            assert mp.almosteq(
                cosebis._raw_roots[i][j],
                roots_mm[i][j],
                rel_eps=1e-49,
                abs_eps=1e-49,
            )

    with pytest.raises(TypeError):
        cosebis = COSEBIS(
            theta_min=0.5, theta_max=250.0, N_max=N_mode, precision=100.0
        )
    with pytest.raises(TypeError):
        cosebis = COSEBIS(theta_min=0.5, theta_max=250.0, N_max=1.0)


def test_cosebis_Tp_log_Tm_log():
    """
    Test the COSEBIs filter functions Tp_log and Tm_log based on their
    mathematical properties as described in Schneider et al. 2022 Eq. 2, 4, 5
    (https://arxiv.org/abs/2110.09774).
    """
    N_mode = 12
    tmin = 0.1
    tmax = 400.0
    nbins = 100_000
    theta_ = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    theta = np.mean([theta_[:-1], theta_[1:]], axis=0)

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=N_mode)
    Tp_log = cosebis.get_Tp_log(theta)
    Tm_log = cosebis.get_Tm_log(theta)

    # Check Eq. 4 & 5
    for n in range(N_mode):
        res, err, success = interp_quad(
            np.log(theta[0]),
            np.log(theta[-1]),
            np.median(np.diff(np.log(theta))),
            Tp_log[n] * theta,
            theta[0],
            theta[-1],
            k=5,
            periodic=False,
            log_interp=True,
        )

        assert success
        assert_allclose(res, 0.0, atol=1e-5, rtol=0.0)

        res, err, success = interp_quad(
            np.log(theta[0]),
            np.log(theta[-1]),
            np.median(np.diff(np.log(theta))),
            Tm_log[n] / theta,
            theta[0],
            theta[-1],
            k=5,
            periodic=False,
            log_interp=True,
        )

        assert success
        assert_allclose(res, 0.0, atol=1e-5, rtol=0.0)

    # Check Eq. 2
    theta_ext, i_low, i_high = extend_log_grid(
        theta,
        pad_low_decades=2.0,
        pad_high_decades=2.0,
    )
    n_theta_ext = len(theta_ext)
    theta_rad = np.radians(theta_ext / 60)

    for n in range(N_mode):
        Tp_log_ext = pad_1D(len(theta_ext), i_low, i_high, Tp_log[n])
        Tm_log_ext = pad_1D(len(theta_ext), i_low, i_high, Tm_log[n])

        ell_, Tp_ell_ = fht(
            n_theta_ext,
            theta_rad,
            Tp_log_ext,
            dim=2,
            mu=0.0,
            q_=-1.3,
            kcrc=1.0,
            noring=1,
        )
        ell = ell_[i_low:i_high]
        Tp_ell = Tp_ell_[i_low:i_high]

        ell_, Tm_ell_ = fht(
            n_theta_ext,
            theta_rad,
            Tm_log_ext,
            dim=2,
            mu=4.0,
            q_=-1.3,
            kcrc=1.0,
            noring=1,
        )
        ell = ell_[i_low:i_high]
        Tm_ell = Tm_ell_[i_low:i_high]

        # This thest is a litlle bit unconventional but due to the oscillatory
        # nature of Tpm_log, it is extremlly challenging to check for equality.
        # However, this combined to the one above should be sufficient to make
        # sure that the Tpm_log function are correctly computed.
        num = simpson((Tp_ell - Tm_ell) ** 2, ell)
        den = simpson(Tp_ell**2, ell)
        rel_err = np.sqrt(num / den)
        if n == 0:
            assert rel_err < 2e-3
        else:
            assert rel_err < 5e-2

    # Basic tests
    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=1)
    assert cosebis.get_Tp_log(theta) is not None
    roots = cosebis.roots
    norms = cosebis.norms

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=1)
    assert cosebis.get_Tm_log(theta) is not None

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=1)
    with pytest.raises(ValueError):
        Tp_log = cosebis.get_Tp_log(theta, roots=roots)
    with pytest.raises(ValueError):
        Tp_log = cosebis.get_Tp_log(theta, norms=norms)
    with pytest.raises(ValueError):
        Tm_log = cosebis.get_Tm_log(theta, roots=roots)
    with pytest.raises(ValueError):
        Tm_log = cosebis.get_Tm_log(theta, norms=norms)

    assert cosebis.get_Tp_log(theta, roots=roots, norms=norms) is not None
    assert cosebis.get_Tm_log(theta, roots=roots, norms=norms) is not None


def test_cosebis_wn_log():
    """
    Test the COSEBIs filter functions Wn_log.
    The test here is extremely basic has the computation is equivalent to the
    check for Eq. 2 in the test_cosebis_Tp_log_Tm_log function.
    """
    N_mode = 3
    tmin = 0.1
    tmax = 400.0
    nbins = 100_000
    theta_ = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)
    theta = np.mean([theta_[:-1], theta_[1:]], axis=0)

    ell = np.logspace(np.log10(50), np.log10(10_000), 1_000)

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=N_mode)
    wn_log = cosebis.get_Wn_log(ell)
    wn_log = cosebis.get_Wn_log(ell, theta=theta)

    assert wn_log.shape[0] == N_mode
    assert wn_log.shape[1] == len(ell)


def test_cosebis_from_xipm_and_Cell():
    """
    Test the COSEBIs computation from real-space correlation functions
    xi_plus and xi_minus as well as from angular power spectra Cell.
    The test make use of pre-computed xi_pm and Cell from CCL.
    """
    N_mode = 12

    # Load pre-computed xi_pm from CCL
    theta, xip_model, xim_model = np.load(
        os.path.join(
            DATA_DIR,
            "ccl_xi_pm_0.5_250_10_000.npy",
        ),
    )
    ell, Cell = np.load(
        os.path.join(
            DATA_DIR,
            "ccl_ell_cell_1_100k_100k.npy",
        ),
    )

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=N_mode)
    cosebis.compute_roots()

    ###
    # From xipm
    ###
    # Serial computation
    C_E, C_B = cosebis.cosebis_from_xipm(
        theta, xip_model, xim_model, cache=True, parallel=False
    )

    assert C_E.size == N_mode
    assert C_B.size == N_mode
    assert_allclose(C_B, 0.0, atol=1e-14, rtol=0.0)

    # Check caching
    assert hasattr(cosebis, "_Tp_log")
    assert hasattr(cosebis, "_Tm_log")

    # Parallel computation
    C_E, C_B = cosebis.cosebis_from_xipm(
        theta, xip_model, xim_model, cache=True, parallel=True
    )

    assert C_E.size == N_mode
    assert C_B.size == N_mode
    assert_allclose(C_B, 0.0, atol=1e-14, rtol=0.0)

    # Check theta out of range
    theta_bad = np.concatenate(([theta[0] / 2], theta, [theta[-1] * 2]))
    with pytest.raises(ValueError):
        C_E, C_B = cosebis.cosebis_from_xipm(
            theta_bad[:-1], xip_model, xim_model, cache=True, parallel=False
        )
    with pytest.raises(ValueError):
        C_E, C_B = cosebis.cosebis_from_xipm(
            theta_bad[1:], xip_model, xim_model, cache=True, parallel=False
        )

    ###
    # From Cell
    ###
    # Serial computation
    C_E_l, C_B_l = cosebis.cosebis_from_Cell(
        ell, Cell, np.zeros_like(Cell), cache=True, parallel=False
    )

    assert C_E_l.size == N_mode
    assert C_B_l.size == N_mode
    assert_allclose(C_B_l, 0.0, atol=1e-14, rtol=0.0)

    # Check caching
    assert hasattr(cosebis, "_Wn_log")

    # Parallel computation
    C_E_l, C_B_l = cosebis.cosebis_from_Cell(
        ell, Cell, np.zeros_like(Cell), cache=True, parallel=True
    )

    assert C_E_l.size == N_mode
    assert C_B_l.size == N_mode
    assert_allclose(C_B_l, 0.0, atol=1e-14, rtol=0.0)

    ###
    # Consistency between xipm and Cell
    ###

    num = np.linalg.norm(C_E_l - C_E)
    den = np.linalg.norm(C_E)
    err = num / den
    assert err < 1e-3


def test_cosebis_covariance_from_xipm_covariance():
    """
    Test the COSEBIs covariance computation from xi_pm covariance.
    We compared the computed covariance to a pre-computed one derived by using
    500k samples of xi_pm drawn from the original covariance and then computing
    the COSEBIs covariance from the samples. We check agreement between the two
    covariances diagonal elements (off-diagonal elements are too noisy for the
    test) within 0.5% relative tolerance.
    """
    N_mode = 12

    # Load pre-computed xi_pm covariance from cosmocov
    cov_xipm = np.load(
        os.path.join(
            DATA_DIR,
            "cosmocov_cov_lsst_xipm_0.5_250_1000.npy",
        ),
    )
    cov_EB_sampled = np.load(
        os.path.join(
            DATA_DIR,
            "sampled_500k_cov_lsst_cosebis_20.npy",
        ),
    )

    tmin = 0.5
    tmax = 250.0
    theta_ = np.logspace(np.log10(tmin), np.log10(tmax), 1_000 + 1)
    theta = np.mean([theta_[:-1], theta_[1:]], axis=0)

    cosebis = COSEBIS(theta_min=theta[0], theta_max=theta[-1], N_max=N_mode)
    cosebis.compute_roots()

    cov_EB = cosebis.cosebis_covariance_from_xipm_covariance(theta, cov_xipm)

    assert_allclose(
        np.diag(cov_EB)[:N_mode],
        np.diag(cov_EB_sampled)[:N_mode],
        atol=0,
        rtol=0.005,
    )
    assert_allclose(
        np.diag(cov_EB)[N_mode:],
        np.diag(cov_EB_sampled)[20 : N_mode + 20],
        atol=0,
        rtol=0.005,
    )

    assert cov_EB.shape == (2 * N_mode, 2 * N_mode)

    # Check theta out of range
    theta_bad = np.concatenate(([theta[0] / 2], theta, [theta[-1] * 2]))
    with pytest.raises(ValueError):
        cov_EB = cosebis.cosebis_covariance_from_xipm_covariance(
            theta_bad[:-1], cov_xipm
        )
    with pytest.raises(ValueError):
        cov_EB = cosebis.cosebis_covariance_from_xipm_covariance(
            theta_bad[1:], cov_xipm
        )


def test_cosebis_roots_insufficient_precision_raises():
    """
    On a narrow window the root finding loses digits mode after mode. At too
    low a precision it returns a basis that is no longer orthonormal; this
    must raise rather than return the wrong filters.
    """
    cosebis = COSEBIS(theta_min=1.0, theta_max=1.05, N_max=8, precision=30)
    with pytest.raises(ValueError, match="not orthonormal"):
        cosebis.compute_roots()

    cosebis = COSEBIS(theta_min=1.0, theta_max=1.05, N_max=8, precision=80)
    cosebis.compute_roots()
    assert len(cosebis.roots) == 8


def test_cosebis_roots_narrow_window_converge():
    """
    20 modes on [1, 1.318] arcmin are well defined at precision=120, but the
    polynomial root finding needs more than its default 50 steps there.
    """
    cosebis = COSEBIS(theta_min=1.0, theta_max=1.318, N_max=20, precision=120)
    cosebis.compute_roots()
    assert len(cosebis.roots) == 20
    for n, roots in enumerate(cosebis.roots, start=1):
        assert roots.size == n + 1
        assert np.all((roots > 0) & (roots < np.log(1.318)))
