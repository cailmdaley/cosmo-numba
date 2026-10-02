"""E/B decomposition

Compute E-/B-modes decomposition based on Schneider et al. 2022
(https://arxiv.org/abs/2110.09774)

Author: Axel Guinot

"""

import os

import numba as nb
import numpy as np

from ..math.integrate.quad import interp_quad

from ..math.utils import extend_log_grid, pad_1D


# This allows having coverage
IN_COVERAGE = False
if (
    os.environ.get("TESTING_SCHNEIDER2022", "0") == "1"
    and os.environ.get("COVERAGE_MODE", "0") == "1"
):
    nb.config.DISABLE_JIT = 1
    IN_COVERAGE = True


@nb.njit
def H_p(t, t_int, theta_bar, B):
    """H_p

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 46.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        H_plus
    """
    return (
        1
        / (8 * B**3)
        * (
            4 * B**2
            + 3
            * ((t / theta_bar) ** 2 - 1 - B**2)
            * ((t_int / theta_bar) ** 2 - 1 - B**2)
        )
    )


@nb.njit
def H_m(t, t_int, theta_bar, B):
    """H_m

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 47.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        H_minus
    """
    return (
        (1 - B) ** 2
        / (8 * B**3)
        * (
            3
            * (1 - B) ** 2
            * ((1 + B) ** 4 - (1 + 4 * B + B**2) * (t / theta_bar) ** 2)
            * (t_int / theta_bar) ** (-2)
            + (
                3 * (1 + B) ** 2 * (t / theta_bar) ** 2
                - (3 + 6 * B + 14 * B**2 + 6 * B**3 + 3 * B**4)
            )
        )
    )


@nb.njit
def K_p(t, t_int, theta_bar, B):
    """K_p

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 51.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        K_plus
    """
    return (theta_bar / t) ** 2 * H_m(t_int, t, theta_bar, B)


@nb.njit
def K_m(t, t_int, theta_bar, B):
    """K_m

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 54.

    Parameters
    ----------
    t : float64
        theta (fancy)
    t_int : numpy.ndarray(float64)
        theta on which the integral is performed
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.

    Returns
    -------
    numpy.ndarray(float64)
        K_minus
    """
    return (
        (1 - B**2) ** 2
        * theta_bar**4
        / (t**2 * t_int**2)
        * H_p(
            (1 - B**2) * theta_bar**2 / t,
            (1 - B**2) * theta_bar**2 / t_int,
            theta_bar,
            B,
        )
    )


@nb.njit
def get_S(
    theta_int,
    xip_int,
    xim_int,
    t,
    tmin,
    tmax,
    theta_bar,
    B,
    interp_order=5,
    epsabs=1e-10,
    epsrel=1e-10,
):
    """get_S

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 44 for
    S_plus and Eq. 45 for S_minus.

    Parameters
    ----------
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    t : float64
        theta (fancy).
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    interp_order : int
        interpolation order used in the integrals
    epsabs : float64
        absolute error tolerance used in the integrals
    epsrel : float64
        relative error tolerance used in the integrals

    Returns
    -------
    tuple(float64, float64)
        S_plus, S_minus
    """

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    S_p_integrand = (
        1
        / theta_bar**2
        * theta_int
        * xip_int
        * H_p(t, theta_int, theta_bar, B)
    )
    S_p_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        S_p_integrand,
        tmin,
        tmax,
        k=interp_order,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    S_m_integrand = 1 / theta_int * xim_int * H_m(t, theta_int, theta_bar, B)
    S_m_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        S_m_integrand,
        tmin,
        tmax,
        k=interp_order,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    return S_p_int, S_m_int


@nb.njit
def get_V(
    theta_int,
    xip_int,
    xim_int,
    t,
    tmin,
    tmax,
    theta_bar,
    B,
    interp_order=5,
    epsabs=1e-10,
    epsrel=1e-10,
):
    """get_V

    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 50 for
    V_plus and Eq. 53 for V_minus.

    Parameters
    ----------
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    t : float64
        theta (fancy).
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    theta_bar : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    B : float64
        See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 7.
    interp_order : int
        interpolation order used in the integrals
    epsabs : float64
        absolute error tolerance used in the integrals
    epsrel : float64
        relative error tolerance used in the integrals

    Returns
    -------
    tuple(float64, float64)
        V_plus, V_minus
    """

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    V_p_integrand = (
        1
        / theta_bar**2
        * theta_int
        * xip_int
        * K_p(t, theta_int, theta_bar, B)
    )
    V_p_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        V_p_integrand,
        tmin,
        tmax,
        k=interp_order,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    V_m_integrand = (
        1
        / theta_bar**2
        * theta_int
        * xim_int
        * K_m(t, theta_int, theta_bar, B)
    )
    V_m_int = interp_quad(
        np.log(theta_int[0]),
        np.log(theta_int[-1]),
        d_theta_int,
        V_m_integrand,
        tmin,
        tmax,
        k=interp_order,
        padding=True,
        extrap_dist=1,
        log_interp=True,
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    return V_p_int, V_m_int


@nb.njit
def _get_xip_EB(
    t,
    theta_bar,
    B,
    theta_int,
    d_theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    interp_order,
    epsabs,
    epsrel,
):
    m_int = (t < theta_int) & (theta_int <= tmax)
    if sum(m_int) == 0:
        int_tmp = 0.0
    else:
        integrand = (
            1
            / theta_int[m_int]
            * xim_int[m_int]
            * (4 - 12 * t**2 / theta_int[m_int] ** 2)
        )

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            t,
            tmax,
            k=interp_order,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]

    m_int = (theta_int >= tmin) & (theta_int <= tmax)
    S_p_int, S_m_int = get_S(
        theta_int[m_int],
        xip_int[m_int],
        xim_int[m_int],
        t,
        tmin,
        tmax,
        theta_bar,
        B,
        interp_order=interp_order,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return int_tmp, S_p_int, S_m_int


@nb.njit
def _get_xim_EB(
    t,
    theta_bar,
    B,
    theta_int,
    d_theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    interp_order,
    epsabs,
    epsrel,
):
    m_int = (t > theta_int) & (theta_int >= tmin)
    if sum(m_int) == 0:
        int_tmp = 0.0
    else:
        integrand = (
            theta_int[m_int]
            / t**2
            * xip_int[m_int]
            * (4 - 12 * theta_int[m_int] ** 2 / t**2)
        )

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            tmin,
            t,
            k=interp_order,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]

    m_int = (theta_int >= tmin) & (theta_int <= tmax)
    V_p_int, V_m_int = get_V(
        theta_int[m_int],
        xip_int[m_int],
        xim_int[m_int],
        t,
        tmin,
        tmax,
        theta_bar,
        B,
        interp_order=interp_order,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return int_tmp, V_p_int, V_m_int


@nb.njit
def _get_pure_EB_modes_serial(
    theta,
    xip,
    xim,
    theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    epsabs=1e-10,
    epsrel=1e-10,
):
    """get_pure_EB_modes

    Computes pure xi E-/B-modes decomposition.
    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 42-43 for
    xi_plus and Eq. 55-56 for xi_minus.

    Parameters
    ----------
    theta : numpy.ndarray(float64)
        theta in arcmin
    xip : numpy.ndarray(float64)
        xi_plus
    xim : numpy.ndarray(float64)
        xi_minus
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    pad_xim : bool
        If True, pads xim and related arrays to avoid edge effects when
        computing xi_minus E/B-modes
    pad_theta_max_decade : float64
        Number of decades to pad xim and related arrays to avoid edge effects
        when computing xi_minus E/B-modes
    interp_order : int
        interpolation order used in the integrals
    epsabs : float64
        absolute error tolerance used in the integrals
    epsrel : float64
        relative error tolerance used in the integrals

    Returns
    -------
    tuple(float64, float64, float64, float64, float64, float64)
        xi_plus_E, xi_minus_E, xi_amb_E, xi_plus_B, xi_minus_B, xi_amb_B
    """

    n_theta = len(theta)

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    theta_bar = (tmax + tmin) / 2
    B = (tmax - tmin) / (tmax + tmin)

    if pad_xim:
        theta_xim, i_low, i_high = extend_log_grid(
            theta,
            pad_low_decades=0.0,
            pad_high_decades=pad_theta_max_decade,
        )
        xim_ext = pad_1D(len(theta_xim), i_low, i_high, xim)
        xip_ext = pad_1D(len(theta_xim), i_low, i_high, xip)
        theta_int_xim, i_low_int, i_high_int = extend_log_grid(
            theta_int,
            pad_low_decades=0.0,
            pad_high_decades=pad_theta_max_decade,
        )
        xim_int_ext = pad_1D(
            len(theta_int_xim), i_low_int, i_high_int, xim_int
        )
        xip_int_ext = pad_1D(
            len(theta_int_xim), i_low_int, i_high_int, xip_int
        )
        tmax_xim = theta_xim[-1]
    else:
        theta_xim = theta
        xim_ext = xim
        xip_ext = xip
        theta_int_xim = theta_int
        xim_int_ext = xim_int
        xip_int_ext = xip_int
        tmax_xim = tmax
    theta_bar_xim = (tmax_xim + tmin) / 2
    B_xim = (tmax_xim - tmin) / (tmax_xim + tmin)

    n_theta_loop = max(n_theta, len(theta_xim))
    xip_E = np.empty(n_theta, dtype=np.float64)
    xip_B = np.empty(n_theta, dtype=np.float64)
    xip_amb = np.empty(n_theta, dtype=np.float64)
    xim_E = np.empty(n_theta_loop, dtype=np.float64)
    xim_B = np.empty(n_theta_loop, dtype=np.float64)
    xim_amb = np.empty(n_theta_loop, dtype=np.float64)

    for i in nb.prange(n_theta_loop):
        t = theta_xim[i]

        # xi_p
        if i < n_theta:
            int_tmp, S_p_int, S_m_int = _get_xip_EB(
                t,
                theta_bar,
                B,
                theta_int,
                d_theta_int,
                xip_int,
                xim_int,
                tmin,
                tmax,
                interp_order,
                epsabs,
                epsrel,
            )

            xip_E[i] = 0.5 * (xip[i] + xim[i] + int_tmp) - 0.5 * (
                S_p_int + S_m_int
            )
            xip_B[i] = 0.5 * (xip[i] - xim[i] - int_tmp) - 0.5 * (
                S_p_int - S_m_int
            )
            xip_amb[i] = S_p_int

        # xi_m
        int_tmp, V_p_int, V_m_int = _get_xim_EB(
            t,
            theta_bar_xim,
            B_xim,
            theta_int_xim,
            d_theta_int,
            xip_int_ext,
            xim_int_ext,
            tmin,
            tmax_xim,
            interp_order,
            epsabs,
            epsrel,
        )

        xim_E[i] = 0.5 * (xip_ext[i] + xim_ext[i] + int_tmp) - 0.5 * (
            V_p_int + V_m_int
        )
        xim_B[i] = 0.5 * (xip_ext[i] - xim_ext[i] + int_tmp) - 0.5 * (
            V_p_int - V_m_int
        )
        xim_amb[i] = V_m_int

    return (
        xip_E,
        xim_E[:n_theta],
        xip_B,
        xim_B[:n_theta],
        xip_amb,
        xim_amb[:n_theta],
    )


if IN_COVERAGE:  # pragma: no cover
    _get_pure_EB_modes_parallel = _get_pure_EB_modes_serial
else:  # pragma: no cover
    _get_pure_EB_modes_parallel = nb.njit(
        parallel=True,
    )(_get_pure_EB_modes_serial.py_func)


@nb.njit(
    # parallel=True,
)
def get_CNPT_EB_modes(
    theta, xip, xim, theta_int, xip_int, xim_int, tmin, tmax
):  # pragma: no cover
    """get_pure_EB_modes

    Computes the CNPT xi E-/B-modes decomposition.
    See Schneider et al. 2022 (https://arxiv.org/abs/2110.09774) Eq. 59.

    Parameters
    ----------
    theta : numpy.ndarray(float64)
        theta in arcmin
    xip : numpy.ndarray(float64)
        xi_plus
    xim : numpy.ndarray(float64)
        xi_minus
    theta_int : numpy.ndarray(float64)
        theta used to cimpute the integrals in arcmin
    xip_int : numpy.ndarray(float64)
        xi_plus used to compute the integrals
    xim_int : numpy.ndarray(float64)
        xi_minus used to compute the integrals
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals

    Returns
    -------
    tuple(float64, float64, float64, float64)
        xi_plus_E, xi_minus_E, xi_plus_B, xi_minus_B
    """

    n_theta = len(theta)
    xip_E = np.empty(n_theta, dtype=np.float64)
    xip_B = np.empty(n_theta, dtype=np.float64)
    xim_E = np.empty(n_theta, dtype=np.float64)
    xim_B = np.empty(n_theta, dtype=np.float64)

    d_theta_int = np.mean(np.diff(np.log(theta_int)))

    for i in nb.prange(n_theta):
        t = theta[i]

        # xi_p
        m_int = (t < theta_int) & (theta_int < tmax)
        integrand = (
            1
            / theta_int[m_int]
            * xim_int[m_int]
            * (4 - 12 * t**2 / theta_int[m_int] ** 2)
        )

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            t,
            tmax,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10,
            epsrel=1e-10,
        )[0]

        xip_E[i] = 0.5 * (xip[i] + xim[i] + int_tmp)
        xip_B[i] = 0.5 * (xip[i] - xim[i] - int_tmp)

        # xi_m
        m_int = (t > theta_int) & (theta_int > tmin)
        integrand = (
            theta_int[m_int]
            / t**2
            * xip_int[m_int]
            * (4 - 12 * theta_int[m_int] ** 2 / t**2)
        )

        int_tmp = interp_quad(
            np.log(theta_int[m_int][0]),
            np.log(theta_int[m_int][-1]),
            d_theta_int,
            integrand,
            tmin,
            t,
            k=5,
            padding=True,
            extrap_dist=1,
            log_interp=True,
            epsabs=1e-10,
            epsrel=1e-10,
        )[0]

        xim_E[i] = 0.5 * (xip[i] + xim[i] + int_tmp)
        xim_B[i] = 0.5 * (xip[i] - xim[i] + int_tmp)

    return xip_E, xim_E, xip_B, xim_B
