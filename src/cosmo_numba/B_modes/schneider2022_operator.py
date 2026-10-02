"""E/B decomposition: fixed-quadrature operator

Linear-operator form of the pure E/B decomposition of Schneider et al. 2022
(https://arxiv.org/abs/2110.09774) as implemented in `schneider2022_nb`.

Each integral of `schneider2022_nb` integrates the interpolant of a sampled
integrand `xi * K`, a correlation function times a kernel, with the adaptive
`interp_quad`. Its fixed-rule counterpart `interp_quad_weights` integrates the
same interpolant exactly, through weights `w` that depend only on the grid and
the bounds, so that the integral is `w @ (xi * K) = (w * K) @ xi`. Every
output of `get_pure_EB_modes` is then a matrix-vector product, and the
transform is exactly linear: covariances propagate as `M C M^T`, and noise
cannot rectify into a bias through the quadrature.

Author: Cail Daley

"""

import numpy as np

from ..math.integrate.quad import interp_quad_weights
from ..math.utils import extend_log_grid
from .schneider2022_nb import H_m, H_p, K_m, K_p

# Order of the outputs of get_pure_EB_modes and get_pure_EB_operator.
PURE_EB_OUTPUTS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")


def _weights(nodes, lo, hi, h, k):
    """Weights of one `interp_quad` call of `schneider2022_nb`.

    The integral over theta in `[lo, hi]` of the interpolant of samples on
    `nodes`, with step `h` in log theta; empty for an empty support.
    """
    if nodes.size == 0:
        return np.zeros(0)
    x = np.log(nodes)
    return interp_quad_weights(
        x[0],
        x[-1],
        h,
        nodes.size,
        lo,
        hi,
        k=k,
        padding=True,
        extrap_dist=1,
        log_interp=True,
    )


def _node_index(theta, theta_int):
    """Index of the node of `theta_int` equal to each `theta`, or raise."""
    log_int = np.log(theta_int)
    log_t = np.log(theta)
    i = np.clip(np.searchsorted(log_int, log_t), 1, theta_int.size - 1)
    i = np.where(
        np.abs(log_int[i - 1] - log_t) <= np.abs(log_int[i] - log_t), i - 1, i
    )
    if not np.allclose(theta_int[i], theta, rtol=1e-10, atol=0.0):
        raise ValueError(
            "local_from_int=True requires every theta to be a node of "
            "theta_int"
        )
    return i


def get_pure_EB_operator(
    theta,
    theta_int,
    tmin,
    tmax,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    local_from_int=False,
):
    """get_pure_EB_operator

    Matrix form of `get_pure_EB_modes` with fixed (non-adaptive) quadrature.

    For any `theta` and log-spaced `theta_int`, the returned matrices `M`
    satisfy

        get_pure_EB_modes(theta, xip, xim, theta_int, xip_int, xim_int, ...,
                          quadrature="fixed")[j]
        == M[j] @ np.concatenate([xip, xim, xip_int, xim_int])

    The integrals are those of the degree-`interp_order` interpolant that the
    adaptive quadrature integrates, computed exactly (to round-off) by
    `interp_quad_weights`. `theta` may lie anywhere, on or off the nodes of
    `theta_int`, and integration limits outside the sampled range follow the
    interpolator's extrapolation and clamping.

    Where the support of a `theta`-dependent integral (`(theta, tmax]` for
    xi_plus, `[tmin, theta)` for xi_minus) holds 1 to `interp_order` nodes,
    the interpolant is undetermined and the corresponding matrix entries are
    NaN, so the outputs at those `theta` are NaN. An empty support
    contributes zero, as in the adaptive path.

    Parameters
    ----------
    theta : numpy.ndarray(float64)
        theta in arcmin at which the modes are evaluated
    theta_int : numpy.ndarray(float64)
        log-spaced theta used to compute the integrals in arcmin
    tmin : float64
        lower bound used for theta in the integrals
    tmax : float64
        upper bound used for theta in the integrals
    pad_xim : bool
        If True, the xi_minus integrals run beyond `tmax`, over a grid
        extended with zero xi_plus and xi_minus, as in `get_pure_EB_modes`.
    pad_theta_max_decade : float64
        Number of decades of that extension.
    interp_order : int
        interpolation order used in the integrals
    local_from_int : bool
        If False, the local `0.5 * (xi_plus +/- xi_minus)` term is taken from
        `xip`, `xim` at `theta` and each matrix acts on
        `concatenate([xip, xim, xip_int, xim_int])`. If True, every `theta`
        must be a node of `theta_int`, the local term is read from `xip_int`,
        `xim_int` at that node, and each matrix acts on
        `concatenate([xip_int, xim_int])`. This single-vector form is the
        one to use for propagating a covariance of `(xip_int, xim_int)`.

    Returns
    -------
    tuple(numpy.ndarray(float64), ...)
        xi_plus_E, xi_minus_E, xi_plus_B, xi_minus_B, xi_plus_amb,
        xi_minus_amb operators (the order of `get_pure_EB_modes`), each of
        shape `(len(theta), 2 * len(theta) + 2 * len(theta_int))`, or
        `(len(theta), 2 * len(theta_int))` if `local_from_int`.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    theta_int = np.ascontiguousarray(theta_int, dtype=np.float64)
    tmin = float(tmin)
    tmax = float(tmax)
    k = int(interp_order)
    n = theta.size
    n_int = theta_int.size
    h = np.mean(np.diff(np.log(theta_int)))
    theta_bar = (tmax + tmin) / 2
    B = (tmax - tmin) / (tmax + tmin)

    # xi_minus branch: the integration grid continued by zero-valued nodes,
    # and its window, as in _get_pure_EB_modes_serial.
    if pad_xim:
        theta_xim = extend_log_grid(theta, 0.0, pad_theta_max_decade)[0]
        theta_int_xim = extend_log_grid(
            theta_int, 0.0, pad_theta_max_decade
        )[0]
        tmax_xim = theta_xim[-1]
    else:
        theta_int_xim = theta_int
        tmax_xim = tmax
    theta_bar_xim = (tmax_xim + tmin) / 2
    B_xim = (tmax_xim - tmin) / (tmax_xim + tmin)

    # The integrals as weights on the integration grid. Supports that depend
    # on theta (Eqs. 42-43, 55-56) are taken row by row.
    I_p = np.zeros((n, n_int))
    I_m = np.zeros((n, theta_int_xim.size))
    for r, t in enumerate(theta):
        m = (t < theta_int) & (theta_int <= tmax)
        g = theta_int[m]
        I_p[r, m] = _weights(g, t, tmax, h, k) / g * (4 - 12 * t**2 / g**2)
        m = (t > theta_int_xim) & (theta_int_xim >= tmin)
        g = theta_int_xim[m]
        I_m[r, m] = (
            _weights(g, tmin, t, h, k) * g / t**2 * (4 - 12 * g**2 / t**2)
        )

    # S (Eqs. 44-45) and V (Eqs. 50, 53) run over a fixed window, whose step
    # schneider2022_nb takes from the window's own nodes.
    def window(grid, hi):
        m = (grid >= tmin) & (grid <= hi)
        g = grid[m]
        return m, g, _weights(g, tmin, hi, np.mean(np.diff(np.log(g))), k)

    t = theta[:, None]
    m, g, w = window(theta_int, tmax)
    S_p = np.zeros((n, n_int))
    S_m = np.zeros((n, n_int))
    S_p[:, m] = w * g * H_p(t, g, theta_bar, B) / theta_bar**2
    S_m[:, m] = w * H_m(t, g, theta_bar, B) / g

    m, g, w = window(theta_int_xim, tmax_xim)
    V_p = np.zeros_like(I_m)
    V_m = np.zeros_like(I_m)
    V_p[:, m] = w * g * K_p(t, g, theta_bar_xim, B_xim) / theta_bar_xim**2
    V_m[:, m] = w * g * K_m(t, g, theta_bar_xim, B_xim) / theta_bar_xim**2

    # Padded nodes hold zeros and carry no column.
    I_m, V_p, V_m = I_m[:, :n_int], V_p[:, :n_int], V_m[:, :n_int]

    zero = np.zeros((n, n_int))
    ops = [
        np.hstack([-0.5 * S_p, 0.5 * (I_p - S_m)]),
        np.hstack([0.5 * (I_m - V_p), -0.5 * V_m]),
        np.hstack([-0.5 * S_p, -0.5 * (I_p - S_m)]),
        np.hstack([0.5 * (I_m - V_p), 0.5 * V_m]),
        np.hstack([S_p, zero]),
        np.hstack([zero, V_m]),
    ]

    # Local term 0.5 * (xi_plus +/- xi_minus) at theta, in E and B.
    if local_from_int:
        i_loc = _node_index(theta, theta_int)
        col_p, col_m = i_loc, n_int + i_loc
    else:
        ops = [np.hstack([np.zeros((n, 2 * n)), op]) for op in ops]
        col_p, col_m = np.arange(n), n + np.arange(n)
    rows = np.arange(n)
    for op, sign in zip(ops[:4], (1, 1, -1, -1), strict=True):
        op[rows, col_p] += 0.5
        op[rows, col_m] += 0.5 * sign
    return tuple(ops)


def get_pure_EB_covariance(
    operator, cov, outputs=("xip_E", "xim_E", "xip_B", "xim_B")
):
    """get_pure_EB_covariance

    Covariance of pure E/B modes, `M C M^T`, from the covariance `C` of the
    correlation functions they are computed from. `C` can come from any
    source (analytic, jackknife, mocks); with the fixed-quadrature operator
    the propagation is exact.

    Parameters
    ----------
    operator : tuple(numpy.ndarray(float64), ...)
        Matrices returned by `get_pure_EB_operator`.
    cov : numpy.ndarray(float64)
        Covariance of the vector the matrices act on:
        `concatenate([xip_int, xim_int])` for an operator built with
        `local_from_int=True`, `concatenate([xip, xim, xip_int, xim_int])`
        otherwise.
    outputs : sequence of str
        Outputs to include, among `PURE_EB_OUTPUTS`
        (`"xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"`).

    Returns
    -------
    numpy.ndarray(float64)
        Joint covariance of the outputs stacked in the order of `outputs`:
        block `(i, j)`, of shape `(len(theta), len(theta))`, is the
        covariance of `outputs[i]` with `outputs[j]`.
    """
    unknown = [name for name in outputs if name not in PURE_EB_OUTPUTS]
    if unknown:
        raise ValueError(
            f"unknown outputs {unknown}, expected names in {PURE_EB_OUTPUTS}"
        )
    M = np.concatenate(
        [operator[PURE_EB_OUTPUTS.index(name)] for name in outputs]
    )
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (M.shape[1], M.shape[1]):
        raise ValueError(
            f"cov has shape {cov.shape}, the operator acts on vectors of "
            f"length {M.shape[1]}"
        )
    bad = np.flatnonzero(~np.isfinite(M).all(axis=1))
    if bad.size:
        n_theta = operator[0].shape[0]
        where = [(outputs[i // n_theta], int(i % n_theta)) for i in bad]
        raise ValueError(
            "the operator is undefined (NaN) where the integration support "
            f"is too small, at (output, theta index): {where}"
        )
    out = M @ cov @ M.T
    return 0.5 * (out + out.T)
