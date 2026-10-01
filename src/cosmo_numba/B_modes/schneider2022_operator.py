"""E/B decomposition: fixed-quadrature operator

Linear-operator form of the pure E/B decomposition of Schneider et al. 2022
(https://arxiv.org/abs/2110.09774) as implemented in `schneider2022_nb`.

The reference implementation lays a degree-`k` local-Taylor interpolant (in
log theta) through the integrand samples and integrates it with the adaptive
QUADPACK routine `dqags`. The interpolant is linear in the samples and is a
polynomial of degree `k` in `x = log(theta)` on each grid cell, so after the
substitution `theta = exp(x)` every integral reduces to cell-wise integrals
of `poly(x) * exp(x)`, which Gauss-Legendre quadrature evaluates exactly. The
resulting weights depend on the grids, the bounds and the kernels but not on
the data, so each output of `get_pure_EB_modes` becomes a matrix-vector
product. Unlike the adaptive rule, whose subdivision depends on the sampled
values, the fixed rule makes the transform exactly linear: covariances can be
propagated as `M C M^T`, and noise cannot rectify into a bias through the
quadrature.

The stencil functions of the interpolant and its ghost-cell extrapolation
coefficients are read off `interpolate_1D` itself (by interpolating unit
vectors), so the operator follows the reference interpolation scheme by
construction.

Author: Cail Daley

"""

import numpy as np

from ..math.interpolate.interpolate_1D import _extrapolate1d, nb_interp1d
from ..math.utils import extend_log_grid
from .schneider2022_nb import H_m, H_p, K_m, K_p

# Order of the outputs of get_pure_EB_modes and get_pure_EB_operator.
PURE_EB_OUTPUTS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")

# Extrapolation distance used by every interp_quad call in schneider2022_nb.
_EXTRAP_DIST = 1

_STENCIL_CACHE = {}
_GHOST_CACHE = {}


def _stencil_poly(k, e):
    """Polynomial coefficients of the `k + 1` stencil functions.

    For a point in cell `c` of a uniform grid, the reference interpolator
    returns `sum_i asx_i(r) * f[c - k // 2 + i]` with `r` the position inside
    the cell, in `[-1/2, 1/2)`. Each `asx_i` is a degree-`k` polynomial,
    recovered here from `k + 1` evaluations of the interpolator on unit
    vectors.

    Returns
    -------
    numpy.ndarray(float64)
        Shape `(k + 1, k + 1)`; `coef[p, i]` multiplies `r**p` in `asx_i`.
    """
    key = (k, e)
    if key not in _STENCIL_CACHE:
        n_ref = 4 * (k + 1) + 4
        c = n_ref // 2
        j = np.arange(k + 1)
        r = 0.5 * np.cos((2 * j + 1) * np.pi / (2 * (k + 1)))
        xq = np.ascontiguousarray(c + r + 0.5)
        vals = np.empty((k + 1, k + 1))
        for i in range(k + 1):
            f = np.zeros(n_ref)
            f[c - k // 2 + i] = 1.0
            itp = nb_interp1d(
                0.0, n_ref - 1.0, 1.0, f, k, False, True, e, False
            )
            vals[:, i] = itp.eval(xq)
        _STENCIL_CACHE[key] = np.linalg.solve(
            np.vander(r, k + 1, increasing=True), vals
        )
    return _STENCIL_CACHE[key]


def _ghost_coeffs(k, e):
    """Ghost-cell extrapolation coefficients of `_extrapolate1d`.

    The interpolator pads the samples with `o = k // 2 + e` ghost values on
    each side. Each one is a fixed linear combination of the `k + 1` real
    samples nearest to that side.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        `(left, right)`, each of shape `(o, k + 1)`: `left[p, m]` is the
        coefficient of `f[m]` in ghost `p`, `right[q, m]` that of
        `f[n - k - 1 + m]` in the `q`-th ghost after the last sample.
    """
    key = (k, e)
    if key not in _GHOST_CACHE:
        o = k // 2 + e
        left = np.empty((o, k + 1))
        right = np.empty((o, k + 1))
        for m in range(k + 1):
            f = np.zeros(k + 1)
            f[m] = 1.0
            fb, _ = _extrapolate1d(f, k, False, True, e)
            left[:, m] = fb[:o]
            right[:, m] = fb[k + 1 + o :]
        _GHOST_CACHE[key] = (left, right)
    return _GHOST_CACHE[key]


class _QuadRule:
    """Interpolation and Gauss-Legendre data for degree `k`."""

    def __init__(self, k, n_gauss, e=_EXTRAP_DIST):
        self.k = k
        self.e = e
        self.o = k // 2 + e
        self.coef = _stencil_poly(k, e)
        self.left, self.right = _ghost_coeffs(k, e)
        nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
        self.u = 0.5 * (nodes + 1.0)
        self.w = 0.5 * weights
        self.asx_full = self.asx(self.u - 0.5)

    def asx(self, r):
        """Stencil functions at in-cell positions `r`, shape (len(r), k+1)."""
        return np.vander(np.atleast_1d(r), self.k + 1, increasing=True) @ (
            self.coef
        )

    def cell(self, h, s0, s1):
        """`int_{s0}^{s1} asx_i(s - 1/2) exp(s h) ds` for one grid cell."""
        if s0 == 0.0 and s1 == 1.0:
            return (self.w * np.exp(self.u * h)) @ self.asx_full
        s = s0 + (s1 - s0) * self.u
        return ((s1 - s0) * self.w * np.exp(s * h)) @ self.asx(s - 0.5)


def _unpad(rule, w_pad, n_node):
    """Fold weights on the ghost-padded samples back onto the real ones."""
    o = rule.o
    k = rule.k
    w = w_pad[o : o + n_node].copy()
    w[: k + 1] += rule.left.T @ w_pad[:o]
    w[n_node - k - 1 :] += rule.right.T @ w_pad[n_node + o :]
    return w


def _add_stencil(rule, w_pad, c, values):
    """Add `values` (one per stencil node) onto the stencil of cell `c`."""
    lo = c + rule.o - rule.k // 2
    w_pad[lo : lo + rule.k + 1] += values


def _quad_weights(rule, n_node, x_start, x_end, h, x_a, x_b):
    """Weights reproducing one `interp_quad` call of `schneider2022_nb`.

    Returns `w` such that `w @ g` equals the integral over theta from
    `exp(x_a)` to `exp(x_b)` of the interpolant of the samples `g` placed at
    `x = x_start + h * arange(n_node)` (with `x_end` the last sample, as
    passed to `interp_quad`). Beyond `e` cells outside the samples the
    interpolator holds its boundary value, taken here as the limit from inside
    the outermost cell. Fewer than `k + 1` samples leave the interpolant
    undetermined: the weights are then NaN.
    """
    if x_b < x_a:
        return -_quad_weights(rule, n_node, x_start, x_end, h, x_b, x_a)
    k = rule.k
    e = rule.e
    if n_node < k + 1:
        return np.full(n_node, np.nan)

    lb = x_start - e * h
    ub = x_end + e * h
    w_pad = np.zeros(n_node + 2 * rule.o)

    # Interpolated range, cell by cell in u = (x - x_start) / h.
    u_a = (max(x_a, lb) - x_start) / h
    u_b = (min(x_b, ub) - x_start) / h
    if u_b > u_a:
        # Clamped to the cells the interpolator can reach, which absorbs
        # round-off in (x_end - x_start) / h.
        c_lo = max(int(np.floor(u_a)), -e)
        c_hi = min(max(int(np.ceil(u_b)) - 1, c_lo), n_node - 2 + e)
        if c_hi - c_lo >= 2:
            # Interior cells are whole: same integral up to exp(x_c).
            cells = np.arange(c_lo + 1, c_hi)
            scale = h * np.exp(x_start + cells * h)
            full = rule.cell(h, 0.0, 1.0)
            lo = c_lo + 1 + rule.o - k // 2
            for i in range(k + 1):
                w_pad[lo + i : lo + i + cells.size] += full[i] * scale
        for c in {c_lo, c_hi}:
            s0 = max(u_a - c, 0.0)
            s1 = min(u_b - c, 1.0)
            if s1 > s0:
                _add_stencil(
                    rule,
                    w_pad,
                    c,
                    h * np.exp(x_start + c * h) * rule.cell(h, s0, s1),
                )

    # Clamped range outside [lb, ub]: constant boundary value.
    for x_clip, length in (
        (lb, np.exp(min(x_b, lb)) - np.exp(x_a)),
        (ub, np.exp(x_b) - np.exp(max(x_a, ub))),
    ):
        if length > 0.0:
            u = (x_clip - x_start) / h
            c = min(max(int(np.floor(u)), -e), n_node - 2 + e)
            _add_stencil(rule, w_pad, c, length * rule.asx(u - c - 0.5)[0])

    return _unpad(rule, w_pad, n_node)


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
    n_gauss=16,
):
    """get_pure_EB_operator

    Matrix form of `get_pure_EB_modes` with fixed (non-adaptive) quadrature.

    For any `theta` and log-spaced `theta_int`, the returned matrices `M`
    satisfy

        get_pure_EB_modes(theta, xip, xim, theta_int, xip_int, xim_int, ...,
                          quadrature="fixed")[j]
        == M[j] @ np.concatenate([xip, xim, xip_int, xim_int])

    The integrals are those of the same degree-`interp_order` interpolant
    that the adaptive quadrature integrates, evaluated exactly (to round-off)
    by Gauss-Legendre quadrature on each grid cell. `theta` may lie anywhere,
    on or off the nodes of `theta_int`; integration limits outside the
    sampled range follow the interpolator's extrapolation and clamping.

    Where the support of a `theta`-dependent integral (`(theta, tmax]` for
    xi_plus, `[tmin, theta)` for xi_minus) holds fewer than
    `interp_order + 1` nodes, the interpolant is undetermined and the
    corresponding matrix entries are NaN, so the outputs at those `theta`
    are NaN. An empty support contributes zero, as in the reference.

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
        If True, pads xim and related arrays to avoid edge effects when
        computing xi_minus E/B-modes (as in `get_pure_EB_modes`).
    pad_theta_max_decade : float64
        Number of decades to pad xim and related arrays.
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
    n_gauss : int
        Gauss-Legendre nodes per grid cell. The integrand on a cell is a
        polynomial of degree `interp_order` times an exponential, integrated
        to round-off by the default.

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
    rule = _QuadRule(k, n_gauss)

    n = theta.size
    n_int = theta_int.size
    if local_from_int:
        i_loc = _node_index(theta, theta_int)
        col_p, col_m = i_loc, n_int + i_loc
        off_p, off_m = 0, n_int
        n_col = 2 * n_int
    else:
        col_p, col_m = np.arange(n), n + np.arange(n)
        off_p, off_m = 2 * n, 2 * n + n_int
        n_col = 2 * n + 2 * n_int

    d_theta_int = np.mean(np.diff(np.log(theta_int)))
    theta_bar = (tmax + tmin) / 2
    B = (tmax - tmin) / (tmax + tmin)

    # xi_minus branch: zero-padded integration grid and its window, built
    # exactly as in _get_pure_EB_modes_serial.
    if pad_xim:
        theta_xim = extend_log_grid(theta, 0.0, float(pad_theta_max_decade))[0]
        theta_int_xim = extend_log_grid(
            theta_int, 0.0, float(pad_theta_max_decade)
        )[0]
        tmax_xim = theta_xim[-1]
    else:
        theta_int_xim = theta_int
        tmax_xim = tmax
    theta_bar_xim = (tmax_xim + tmin) / 2
    B_xim = (tmax_xim - tmin) / (tmax_xim + tmin)
    # Padded nodes hold zeros and therefore carry no column.
    real_xim = np.arange(theta_int_xim.size) < n_int

    def fixed_support(grid, lo, hi):
        m = (grid >= lo) & (grid <= hi)
        g = grid[m]
        x = np.log(g)
        w = _quad_weights(
            rule,
            g.size,
            x[0],
            x[-1],
            np.mean(np.diff(x)),
            np.log(lo),
            np.log(hi),
        )
        return m, g, w

    m_S, g_S, w_S = fixed_support(theta_int, tmin, tmax)
    m_V, g_V, w_V = fixed_support(theta_int_xim, tmin, tmax_xim)
    keep_V = real_xim[m_V]
    idx_S = np.nonzero(m_S)[0]
    idx_V = np.nonzero(m_V)[0][keep_V]

    ops = tuple(np.zeros((n, n_col)) for _ in range(6))
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = ops

    for r in range(n):
        t = theta[r]

        # xi_plus: Eqs. 42-45
        row_int = np.zeros(n_int)
        m = (t < theta_int) & (theta_int <= tmax)
        if m.any():
            g = theta_int[m]
            row_int[m] = _quad_weights(
                rule,
                g.size,
                np.log(g[0]),
                np.log(g[-1]),
                d_theta_int,
                np.log(t),
                np.log(tmax),
            ) * (1 / g * (4 - 12 * t**2 / g**2))
        row_Sp = np.zeros(n_int)
        row_Sm = np.zeros(n_int)
        row_Sp[idx_S] = w_S * g_S * H_p(t, g_S, theta_bar, B) / theta_bar**2
        row_Sm[idx_S] = w_S * H_m(t, g_S, theta_bar, B) / g_S

        sl_p = slice(off_p, off_p + n_int)
        sl_m = slice(off_m, off_m + n_int)
        xip_E[r, sl_p] = -0.5 * row_Sp
        xip_E[r, sl_m] = 0.5 * row_int - 0.5 * row_Sm
        xip_B[r, sl_p] = -0.5 * row_Sp
        xip_B[r, sl_m] = -0.5 * row_int + 0.5 * row_Sm
        xip_amb[r, sl_p] = row_Sp

        # xi_minus: Eqs. 50-56
        row_int = np.zeros(n_int)
        m = (t > theta_int_xim) & (theta_int_xim >= tmin)
        if m.any():
            g = theta_int_xim[m]
            w = _quad_weights(
                rule,
                g.size,
                np.log(g[0]),
                np.log(g[-1]),
                d_theta_int,
                np.log(tmin),
                np.log(t),
            ) * (g / t**2 * (4 - 12 * g**2 / t**2))
            row_int[np.nonzero(m)[0][real_xim[m]]] = w[real_xim[m]]
        row_Vp = np.zeros(n_int)
        row_Vm = np.zeros(n_int)
        row_Vp[idx_V] = (
            w_V * g_V * K_p(t, g_V, theta_bar_xim, B_xim) / theta_bar_xim**2
        )[keep_V]
        row_Vm[idx_V] = (
            w_V * g_V * K_m(t, g_V, theta_bar_xim, B_xim) / theta_bar_xim**2
        )[keep_V]

        xim_E[r, sl_p] = 0.5 * row_int - 0.5 * row_Vp
        xim_E[r, sl_m] = -0.5 * row_Vm
        xim_B[r, sl_p] = 0.5 * row_int - 0.5 * row_Vp
        xim_B[r, sl_m] = 0.5 * row_Vm
        xim_amb[r, sl_m] = row_Vm

    # Local term 0.5 * (xi_plus +/- xi_minus) at theta.
    rows = np.arange(n)
    for op in (xip_E, xim_E):
        op[rows, col_p] += 0.5
        op[rows, col_m] += 0.5
    for op in (xip_B, xim_B):
        op[rows, col_p] += 0.5
        op[rows, col_m] -= 0.5

    return ops


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


def get_bin_averaged_operator(operator, theta, bin_edges, weights=None):
    """get_bin_averaged_operator

    Average an operator evaluated on a fine grid into coarse angular bins.

    Each output in bin `b` is the weighted mean of its values at the `theta`
    falling in `[bin_edges[b], bin_edges[b + 1])`, so the averaged operator is
    `P @ M` with `P[b, i] = weights[i] / sum(weights in b)`. The usual setup
    evaluates the operator at the fine-grid nodes inside the bins, with
    `local_from_int=True` and `[tmin, tmax]` covering the fine grid, so that
    every evaluation point has integration support on both sides.

    With `weights` set to the pair weights of the fine bins (TreeCorr's
    `weight` column, the sum of w_i * w_j over pairs) and fine bins nested in
    the coarse ones, the averaged local term `0.5 * (xi_+ +/- xi_-)` is the
    coarse-bin correlation function itself, as TreeCorr would measure it.
    Plain pair counts give the same only for an unweighted catalogue.

    Parameters
    ----------
    operator : tuple(numpy.ndarray(float64), ...)
        Matrices returned by `get_pure_EB_operator`, evaluated at `theta`.
    theta : numpy.ndarray(float64)
        theta in arcmin at which the operator is evaluated
    bin_edges : numpy.ndarray(float64)
        Increasing edges of the coarse bins in arcmin.
    weights : numpy.ndarray(float64), optional
        Non-negative weight of each `theta`. Uniform if None.

    Returns
    -------
    tuple(numpy.ndarray(float64), ...)
        Averaged matrices, in the order of `operator`, each of shape
        `(len(bin_edges) - 1, operator[0].shape[1])`. A NaN row of the
        operator with non-zero weight makes its bin NaN.
    """
    theta = np.asarray(theta, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if weights is None:
        weights = np.ones(theta.size)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != theta.shape or np.any(weights < 0):
        raise ValueError("weights must be non-negative, one per theta")
    if operator[0].shape[0] != theta.size:
        raise ValueError("the operator must have one row per theta")

    n_bin = bin_edges.size - 1
    b = np.searchsorted(bin_edges, theta, side="right") - 1
    used = (b >= 0) & (b < n_bin) & (weights > 0)
    P = np.zeros((n_bin, theta.size))
    P[b[used], np.flatnonzero(used)] = weights[used]
    norm = P.sum(axis=1)
    if np.any(norm == 0):
        raise ValueError(
            f"bins {np.flatnonzero(norm == 0).tolist()} contain no theta "
            "with non-zero weight"
        )
    P = P[:, used] / norm[:, None]
    return tuple(P @ op[used] for op in operator)
