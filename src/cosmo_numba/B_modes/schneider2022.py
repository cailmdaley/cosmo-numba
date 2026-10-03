"""E/B decomposition

Compute E-/B-modes decomposition based on Schneider et al. 2022
(https://arxiv.org/abs/2110.09774)

Author: Axel Guinot

"""

import numpy as np

from .schneider2022_nb import (
    _get_pure_EB_modes_parallel,
    _get_pure_EB_modes_serial,
)
from .schneider2022_operator import (
    PURE_EB_OUTPUTS,
    _apply,
    get_pure_EB_covariance,
    get_pure_EB_operator,
)

__all__ = [
    "PURE_EB_OUTPUTS",
    "get_pure_EB_covariance",
    "get_pure_EB_modes",
    "get_pure_EB_operator",
]


def get_pure_EB_modes(
    theta,
    xip,
    xim,
    theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    parallel=False,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    epsabs=1e-10,
    epsrel=1e-10,
    quadrature="fixed",
):
    """get_pure_EB_modes

    Computes the pure E-/B-mode decomposition of Schneider et al. 2022
    (https://arxiv.org/abs/2110.09774), Eqs. 42-56.

    The integrals are those of a degree-`interp_order` interpolant of the
    integrands sampled on `theta_int`.

    With `quadrature="fixed"` (the default), each grid cell of the
    interpolant is integrated exactly by Gauss-Legendre quadrature, through
    the matrices of `get_pure_EB_operator`. The modes are then exactly linear
    in the inputs, and the same matrices propagate a covariance of the inputs
    (`get_pure_EB_covariance`). Where the support of a theta-dependent
    integral holds 1 to `interp_order` nodes of `theta_int`, the outputs at
    that theta are NaN. A NaN or inf input makes NaN only the outputs whose
    integrals or local terms use it. `parallel`, `epsabs` and `epsrel` are
    not used.

    With `quadrature="adaptive"`, the interpolant is integrated by QUADPACK's
    adaptive `dqags` to the tolerances `epsabs`, `epsrel`, serially or in
    parallel. Its subdivision depends on the data, so the modes are linear
    in the inputs only to that tolerance.

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
    parallel : bool
        If True, runs the adaptive quadrature in parallel.
    pad_xim : bool
        If True, pads xim and related arrays to avoid edge effects when
        computing xi_minus E/B-modes.
    pad_theta_max_decade : float64
        Number of decades to pad xim and related arrays to avoid edge effects
        when computing xi_minus E/B-modes.
    interp_order : int
        interpolation order used in the integrals
    epsabs : float64
        absolute error tolerance of the adaptive quadrature
    epsrel : float64
        relative error tolerance of the adaptive quadrature
    quadrature : str
        "fixed" or "adaptive", see above.

    Returns
    -------
    tuple(numpy.ndarray(float64), ...)
        xi_plus_E, xi_minus_E, xi_plus_B, xi_minus_B, xi_plus_amb,
        xi_minus_amb
    """

    if quadrature == "fixed":
        ops = get_pure_EB_operator(
            theta,
            theta_int,
            tmin,
            tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
        )
        data = np.concatenate([xip, xim, xip_int, xim_int])
        return tuple(_apply(op, data) for op in ops)
    elif quadrature != "adaptive":
        raise ValueError(
            f"quadrature must be 'adaptive' or 'fixed', got {quadrature!r}"
        )

    if parallel:
        return _get_pure_EB_modes_parallel(
            theta,
            xip,
            xim,
            theta_int,
            xip_int,
            xim_int,
            tmin,
            tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
            epsabs=epsabs,
            epsrel=epsrel,
        )
    else:
        return _get_pure_EB_modes_serial(
            theta,
            xip,
            xim,
            theta_int,
            xip_int,
            xim_int,
            tmin,
            tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
            epsabs=epsabs,
            epsrel=epsrel,
        )
