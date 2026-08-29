"""Fixed-quadrature operator form of the Schneider et al. (2022) pure E/B
decomposition.

`cosmo_numba.B_modes.schneider2022.get_pure_EB_modes` evaluates the pure E/B
transform by building a degree-``k`` local-Taylor interpolant through the
*integrand* samples and handing it to QUADPACK's adaptive ``dqags``.  The
interpolant is linear in the sampled function values, so the transform that the
estimator *intends* is linear in ``(xi_p, xi_m)``.  The adaptive quadrature,
however, chooses its subdivision from the sampled values themselves, which
makes the realised map weakly non-linear: on noisy data the sub-tolerance
quadrature error is correlated with the noise and rectifies into a spurious
signal.

This module replaces the adaptive step by a *fixed* quadrature rule.  The
interpolant is piecewise polynomial of degree ``k`` on the cells of the
(log-uniform) integration grid, so each integral

    I(t) = int_a^b P[g](theta) dtheta ,   g_j = c_j(t) * xi_j

is evaluated exactly (to machine precision) by Gauss-Legendre quadrature
applied cell by cell, after the substitution ``theta = exp(x)``.  The resulting
weights depend only on the grid geometry and the kernels, never on the data, so
the whole decomposition collapses to a matrix product

    output = M @ [xi_p(theta); xi_m(theta)] .

The cardinal (stencil) functions and the ghost-cell extrapolation coefficients
are extracted numerically from the very routines used by the reference
implementation (``cosmo_numba.math.interpolate.interpolate_1D``), so the
operator reproduces the reference transform rather than an independent
re-derivation of it.

Author: Cail Daley
"""

import numpy as np

from ..math.interpolate.interpolate_1D import _extrapolate1d, nb_interp1d
from ..math.utils import extend_log_grid
from .schneider2022_nb import H_m, H_p, K_m, K_p

__all__ = ["get_pure_EB_operator", "OUTPUT_NAMES"]


OUTPUT_NAMES = (
    "xip_E",
    "xip_B",
    "xim_E",
    "xim_B",
    "xip_amb",
    "xim_amb",
)

_DEFAULT_OUTPUTS = ("xip_E", "xip_B", "xim_E", "xim_B")

# Cache for the (k, extrap_dist) dependent stencil / ghost coefficients.
_STENCIL_CACHE = {}
_GHOST_CACHE = {}


# --------------------------------------------------------------------------
# Interpolation scheme introspection
# --------------------------------------------------------------------------
def _cardinal_values(ratx, k, extrap_dist):
    """Values of the ``k+1`` cardinal (stencil) functions of the reference
    interpolator.

    For a point falling in cell ``j0`` of a uniform grid, the reference
    interpolator returns ``sum_i asx_i(ratx) * f[j0 - k//2 + i]`` with
    ``ratx = (x - x_grid[j0]) / h - 0.5``.  Here we recover ``asx_i`` by
    evaluating the *actual* interpolator on unit vectors in the middle of a
    reference grid, so no interpolation coefficient is ever re-typed.

    Parameters
    ----------
    ratx : numpy.ndarray(float64)
        Local coordinates in ``[-0.5, 0.5]``.
    k : int
        Interpolation degree (1, 3, 5, 7 or 9).
    extrap_dist : int
        Extrapolation distance of the reference call, in grid cells.

    Returns
    -------
    numpy.ndarray(float64)
        Array of shape ``(len(ratx), k + 1)``.
    """
    ratx = np.ascontiguousarray(np.atleast_1d(ratx), dtype=np.float64)
    n_ref = 4 * (k + 1) + 4
    h = 1.0
    a = 0.0
    b = a + (n_ref - 1) * h
    j0 = n_ref // 2
    xq = np.ascontiguousarray(a + (j0 + ratx + 0.5) * h)

    out = np.empty((ratx.size, k + 1), dtype=np.float64)
    for i in range(k + 1):
        f = np.zeros(n_ref, dtype=np.float64)
        f[j0 - k // 2 + i] = 1.0
        itp = nb_interp1d(a, b, h, f, k, False, True, extrap_dist, False)
        out[:, i] = itp.eval(xq)
    return out


def _stencil_poly(k, extrap_dist):
    """Polynomial coefficients of the ``k+1`` cardinal functions.

    ``asx_i`` is a polynomial of degree ``k`` in ``ratx``; we recover its
    coefficients from ``k+1`` Chebyshev samples of the reference interpolator.

    Returns
    -------
    numpy.ndarray(float64)
        Array of shape ``(k + 1, k + 1)``: ``coef[p, i]`` multiplies
        ``ratx**p`` in ``asx_i``.
    """
    key = (k, extrap_dist)
    if key not in _STENCIL_CACHE:
        j = np.arange(k + 1)
        r = 0.5 * np.cos((2 * j + 1) * np.pi / (2 * (k + 1)))
        vals = _cardinal_values(r, k, extrap_dist)
        vander = np.vander(r, k + 1, increasing=True)
        _STENCIL_CACHE[key] = np.linalg.solve(vander, vals)
    return _STENCIL_CACHE[key]


def _asx(ratx, coef):
    """Evaluate the cardinal functions from their polynomial coefficients."""
    ratx = np.ascontiguousarray(np.atleast_1d(ratx), dtype=np.float64)
    return np.vander(ratx, coef.shape[0], increasing=True) @ coef


def _ghost_coeffs(k, extrap_dist):
    """Ghost-cell extrapolation coefficients of ``_extrapolate1d``.

    The reference interpolator pads the sampled array with ``o = k//2 +
    extrap_dist`` ghost cells on each side, filled by polynomial extrapolation.
    Each ghost value is a fixed linear combination of the ``k+1`` nearest real
    samples (valid whenever the support holds at least ``k+1`` nodes).

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        ``(left, right)`` of shape ``(o, k + 1)``.  ``left[p, m]`` is the
        coefficient of ``f[m]`` in ghost ``p``; ``right[q, m]`` the coefficient
        of ``f[M - (k + 1) + m]`` in ghost ``M + o + q``.
    """
    key = (k, extrap_dist)
    if key not in _GHOST_CACHE:
        o = k // 2 + extrap_dist
        n_min = k + 1
        left = np.empty((o, n_min), dtype=np.float64)
        right = np.empty((o, n_min), dtype=np.float64)
        for m in range(n_min):
            f = np.zeros(n_min, dtype=np.float64)
            f[m] = 1.0
            fb, _ = _extrapolate1d(f, k, False, True, extrap_dist)
            left[:, m] = fb[:o]
            right[:, m] = fb[n_min + o :]
        _GHOST_CACHE[key] = (left, right)
    return _GHOST_CACHE[key]


# --------------------------------------------------------------------------
# Fixed quadrature
# --------------------------------------------------------------------------
class _QuadRule:
    """Everything that depends only on ``(k, extrap_dist, n_gauss)``."""

    def __init__(self, k, extrap_dist, n_gauss):
        self.k = k
        self.extrap_dist = extrap_dist
        self.o = k // 2 + extrap_dist
        # padded index of the first stencil node of cell ``c`` is ``c + base``
        self.base = self.o - k // 2
        self.coef = _stencil_poly(k, extrap_dist)
        self.left, self.right = _ghost_coeffs(k, extrap_dist)

        nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
        self.u = 0.5 * (nodes + 1.0)  # nodes on [0, 1]
        self.w = 0.5 * weights  # weights summing to 1
        self.asx = _asx(self.u - 0.5, self.coef)  # (n_gauss, k+1)

    def full_cell(self, h):
        """``A_i = int_0^1 asx_i(u - 1/2) exp(u h) du``, cell width ``h``."""
        return (self.w * np.exp(self.u * h)) @ self.asx

    def partial_cell(self, h, s0, s1):
        """Same integral restricted to ``u in [s0, s1]``."""
        u = s0 + (s1 - s0) * self.u
        w = (s1 - s0) * self.w
        return (w * np.exp(u * h)) @ _asx(u - 0.5, self.coef)


def _quad_weights(rule, n_node, x_start, h, x_a, x_b, a_full=None):
    """Fixed-quadrature weights reproducing ``interp_quad``.

    Returns ``w`` such that ``w @ g`` equals

        int_{exp(x_a)}^{exp(x_b)} P[g](theta) dtheta

    where ``P[g]`` is the reference degree-``k`` interpolant of the samples
    ``g`` laid on the uniform log-grid ``x_start + h * arange(n_node)``.

    Parameters
    ----------
    rule : _QuadRule
        Precomputed stencil / Gauss-Legendre data.
    n_node : int
        Number of samples in the (masked) support.
    x_start, h : float
        Grid origin and step in ``log(theta)``.
    x_a, x_b : float
        Integration bounds in ``log(theta)``.
    a_full : numpy.ndarray, optional
        ``rule.full_cell(h)``, passed in when it is reused across many calls.

    Returns
    -------
    numpy.ndarray(float64)
        Weights of shape ``(n_node,)``.
    """
    k = rule.k
    o = rule.o
    e = rule.extrap_dist
    base = rule.base

    if n_node < k + 1:
        raise ValueError(
            f"support holds {n_node} nodes, degree-{k} interpolation needs "
            f"at least {k + 1}"
        )

    if a_full is None:
        a_full = rule.full_cell(h)

    # The reference interpolator clips its argument to [x_start - e h,
    # x_end + e h]; emulate that so round-off at the bounds cannot push us
    # outside the padded array.
    u_a = min(max((x_a - x_start) / h, -e), (n_node - 1) + e)
    u_b = min(max((x_b - x_start) / h, -e), (n_node - 1) + e)

    w_pad = np.zeros(n_node + 2 * o, dtype=np.float64)
    if u_b > u_a:
        c_lo = int(np.floor(u_a + 1e-12))
        c_hi = int(np.ceil(u_b - 1e-12)) - 1
        c_hi = max(c_hi, c_lo)

        # interior (full) cells: closed form, vectorised over cells
        if c_hi - c_lo >= 2:
            cells = np.arange(c_lo + 1, c_hi)
            scale = h * np.exp(x_start + cells * h)
            for i in range(k + 1):
                lo = c_lo + 1 + base + i
                w_pad[lo : lo + cells.size] += a_full[i] * scale

        # boundary cells: partial Gauss-Legendre
        for c in (c_lo, c_hi) if c_hi > c_lo else (c_lo,):
            s0 = max(u_a - c, 0.0)
            s1 = min(u_b - c, 1.0)
            if s1 <= s0:
                continue
            contrib = rule.partial_cell(h, s0, s1)
            w_pad[c + base : c + base + k + 1] += (
                h * np.exp(x_start + c * h) * contrib
            )

    # adjoint of the ghost-cell extrapolation
    w = w_pad[o : o + n_node].copy()
    w[: k + 1] += rule.left.T @ w_pad[:o]
    w[n_node - k - 1 :] += rule.right.T @ w_pad[n_node + o :]
    return w


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_pure_EB_operator(
    theta_eval,
    theta,
    tmin,
    tmax,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    extrap_dist=1,
    outputs=_DEFAULT_OUTPUTS,
    n_gauss=16,
):
    """Dense operator form of the pure E/B decomposition.

    Builds matrices ``M`` such that

        out = M @ np.concatenate([xip, xim])

    reproduces the corresponding output of

        get_pure_EB_modes(theta, xip, xim, theta, xip, xim, tmin, tmax,
                          pad_xim=..., pad_theta_max_decade=...,
                          interp_order=...)

    restricted to the rows ``theta_eval``.  Note that the reference function is
    called here with a *single* grid: ``theta`` serves both as the sampling
    grid of the data and as the integration grid (``theta_int``), which is
    how the UNIONS pipeline uses it.  The weights depend only on ``theta``,
    ``tmin``, ``tmax`` and the kernels, so the returned map is exactly linear.

    Stacking convention
    -------------------
    The input vector is ``[xi_p(theta) ; xi_m(theta)]`` of length
    ``2 * len(theta)``; column ``j < n`` is ``xi_p(theta[j])`` and column
    ``n + j`` is ``xi_m(theta[j])``.  Every returned matrix has shape
    ``(len(theta_eval), 2 * len(theta))``.

    Edge policy
    -----------
    Two of the six integrals are taken over a data-dependent sub-range of the
    grid: ``[t, tmax]`` for the ``xi_p`` transform and ``[tmin, t]`` for the
    ``xi_m`` one.  Close to the corresponding end of the range this support
    holds fewer than ``interp_order + 1`` nodes, and the reference
    implementation's spline is then underdetermined (``_extrapolate1d`` reads
    uninitialised ghost cells, i.e. returns garbage).  Rather than reproduce
    that, or silently substitute a different rule, those contributions are set
    to ``NaN`` and the affected rows are flagged in the returned ``valid``
    masks.  An empty support (exactly at the end point) is *not* affected: the
    reference sets the integral to zero there and so do we.

    Parameters
    ----------
    theta_eval : numpy.ndarray(float64)
        Output angular scales in arcmin.  Must be a subset of ``theta`` and lie
        within ``[tmin, tmax]``.
    theta : numpy.ndarray(float64)
        Log-spaced sampling/integration grid in arcmin.
    tmin, tmax : float64
        Bounds of the pure E/B range in arcmin.
    pad_xim : bool
        Zero-pad ``xi_m`` above ``tmax`` before the ``xi_m`` transform.
    pad_theta_max_decade : float64
        Number of decades of zero padding.
    interp_order : int
        Interpolation degree ``k`` (must match the reference call).
    extrap_dist : int
        Extrapolation distance used by the reference call (1 in
        ``schneider2022_nb``).
    outputs : sequence of str
        Which matrices to build; any of ``OUTPUT_NAMES``.  Each dense matrix
        costs ``8 * len(theta_eval) * 2 * len(theta)`` bytes.
    n_gauss : int
        Gauss-Legendre nodes per grid cell.  The integrand is a degree-``k``
        polynomial times ``exp(x)`` on each cell, so this is exact to machine
        precision for any ``n_gauss > (k + 1) / 2`` in practice.

    Returns
    -------
    dict
        ``matrices``  : dict name -> ndarray (n_eval, 2 n)
        ``valid``     : dict name -> bool ndarray (n_eval,)
        ``theta_eval``, ``theta``, ``eval_index``, ``tmin``, ``tmax``,
        ``stacking``.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    theta_eval = np.ascontiguousarray(theta_eval, dtype=np.float64)
    tmin = float(tmin)
    tmax = float(tmax)
    k = int(interp_order)

    unknown = set(outputs) - set(OUTPUT_NAMES)
    if unknown:
        raise ValueError(f"unknown outputs {sorted(unknown)}")

    n = theta.size
    # map theta_eval onto grid indices
    idx = np.searchsorted(theta, theta_eval)
    idx = np.clip(idx, 1, n - 1)
    left = np.abs(theta[idx - 1] - theta_eval)
    right = np.abs(theta[np.minimum(idx, n - 1)] - theta_eval)
    eval_index = np.where(left <= right, idx - 1, idx)
    if not np.allclose(theta[eval_index], theta_eval, rtol=1e-12, atol=0.0):
        raise ValueError("theta_eval must be a subset of theta")
    if theta_eval.min() < tmin or theta_eval.max() > tmax:
        raise ValueError("theta_eval must lie within [tmin, tmax]")
    n_eval = eval_index.size

    rule = _QuadRule(k, extrap_dist, n_gauss)

    d_theta = float(np.mean(np.diff(np.log(theta))))
    theta_bar = (tmax + tmin) / 2
    B = (tmax - tmin) / (tmax + tmin)

    if pad_xim:
        theta_ext, i_low, i_high = extend_log_grid(
            theta, 0.0, float(pad_theta_max_decade)
        )
        tmax_xim = theta_ext[-1]
    else:
        theta_ext, i_low = theta, 0
        tmax_xim = tmax
    theta_bar_xim = (tmax_xim + tmin) / 2
    B_xim = (tmax_xim - tmin) / (tmax_xim + tmin)
    # column index in the stacked vector for each node of the extended grid
    # (padded nodes carry a zero value and therefore no column)
    col_ext = np.arange(theta_ext.size) - i_low
    real_ext = (col_ext >= 0) & (col_ext < n)

    # ---- t-independent supports and weights -----------------------------
    m0 = (theta >= tmin) & (theta <= tmax)
    g0 = theta[m0]
    col0 = np.nonzero(m0)[0]
    h0 = float(np.mean(np.diff(np.log(g0))))
    w0 = _quad_weights(
        rule, g0.size, np.log(g0[0]), h0, np.log(tmin), np.log(tmax)
    )

    m0e = (theta_ext >= tmin) & (theta_ext <= tmax_xim)
    g0e = theta_ext[m0e]
    h0e = float(np.mean(np.diff(np.log(g0e))))
    w0e = _quad_weights(
        rule, g0e.size, np.log(g0e[0]), h0e, np.log(tmin), np.log(tmax_xim)
    )
    keep0e = real_ext[m0e]
    col0e = col_ext[m0e][keep0e]

    a_full = rule.full_cell(d_theta)

    # ---- accumulators ----------------------------------------------------
    need = set(outputs)
    need_xip = bool(need & {"xip_E", "xip_B", "xip_amb"})
    need_xim = bool(need & {"xim_E", "xim_B", "xim_amb"})

    shape = (n_eval, 2 * n)
    rows = {name: np.zeros(shape, dtype=np.float64) for name in outputs}
    valid = {name: np.ones(n_eval, dtype=bool) for name in outputs}

    row_int_p = np.empty(2 * n, dtype=np.float64)
    row_sp = np.empty(2 * n, dtype=np.float64)
    row_sm = np.empty(2 * n, dtype=np.float64)
    row_int_m = np.empty(2 * n, dtype=np.float64)
    row_vp = np.empty(2 * n, dtype=np.float64)
    row_vm = np.empty(2 * n, dtype=np.float64)

    log_tmin = np.log(tmin)
    log_tmax = np.log(tmax)

    for r in range(n_eval):
        ii = int(eval_index[r])
        t = float(theta[ii])

        if need_xip:
            # --- int_tmp: 0.5 * int_t^tmax dtheta' xi_m (4 - 12 t^2/th'^2)/th'
            row_int_p[:] = 0.0
            ok_p = True
            m1 = (t < theta) & (theta <= tmax)
            n1 = int(m1.sum())
            if n1 > 0:
                g1 = theta[m1]
                col1 = np.nonzero(m1)[0] + n
                cf1 = (1.0 / g1) * (4.0 - 12.0 * t**2 / g1**2)
                if n1 < k + 1:
                    row_int_p[col1] = np.nan
                    ok_p = False
                else:
                    w1 = _quad_weights(
                        rule,
                        n1,
                        np.log(g1[0]),
                        d_theta,
                        np.log(t),
                        log_tmax,
                        a_full=a_full,
                    )
                    row_int_p[col1] = w1 * cf1

            # --- S_plus / S_minus
            row_sp[:] = 0.0
            row_sm[:] = 0.0
            row_sp[col0] = w0 * (
                g0 * H_p(t, g0, theta_bar, B) / theta_bar**2
            )
            row_sm[col0 + n] = w0 * (H_m(t, g0, theta_bar, B) / g0)

            if "xip_E" in need:
                out = rows["xip_E"][r]
                out[:] = 0.5 * row_int_p - 0.5 * (row_sp + row_sm)
                out[ii] += 0.5
                out[ii + n] += 0.5
                valid["xip_E"][r] = ok_p
            if "xip_B" in need:
                out = rows["xip_B"][r]
                out[:] = -0.5 * row_int_p - 0.5 * (row_sp - row_sm)
                out[ii] += 0.5
                out[ii + n] -= 0.5
                valid["xip_B"][r] = ok_p
            if "xip_amb" in need:
                rows["xip_amb"][r] = row_sp

        if need_xim:
            # --- int_tmp: int_tmin^t dtheta' th'/t^2 xi_p (4 - 12 th'^2/t^2)
            row_int_m[:] = 0.0
            ok_m = True
            m2 = (t > theta_ext) & (theta_ext >= tmin)
            n2 = int(m2.sum())
            if n2 > 0:
                g2 = theta_ext[m2]
                keep2 = real_ext[m2]
                col2 = col_ext[m2][keep2]
                cf2 = (g2 / t**2) * (4.0 - 12.0 * g2**2 / t**2)
                if n2 < k + 1:
                    row_int_m[col2] = np.nan
                    ok_m = False
                else:
                    w2 = _quad_weights(
                        rule,
                        n2,
                        np.log(g2[0]),
                        d_theta,
                        log_tmin,
                        np.log(t),
                        a_full=a_full,
                    )
                    row_int_m[col2] = (w2 * cf2)[keep2]

            # --- V_plus / V_minus
            row_vp[:] = 0.0
            row_vm[:] = 0.0
            cvp = (
                g0e
                * K_p(t, g0e, theta_bar_xim, B_xim)
                / theta_bar_xim**2
            )
            cvm = (
                g0e
                * K_m(t, g0e, theta_bar_xim, B_xim)
                / theta_bar_xim**2
            )
            row_vp[col0e] = (w0e * cvp)[keep0e]
            row_vm[col0e + n] = (w0e * cvm)[keep0e]

            if "xim_E" in need:
                out = rows["xim_E"][r]
                out[:] = 0.5 * row_int_m - 0.5 * (row_vp + row_vm)
                out[ii] += 0.5
                out[ii + n] += 0.5
                valid["xim_E"][r] = ok_m
            if "xim_B" in need:
                out = rows["xim_B"][r]
                out[:] = 0.5 * row_int_m - 0.5 * (row_vp - row_vm)
                out[ii] += 0.5
                out[ii + n] -= 0.5
                valid["xim_B"][r] = ok_m
            if "xim_amb" in need:
                rows["xim_amb"][r] = row_vm

    return {
        "matrices": rows,
        "valid": valid,
        "theta_eval": theta_eval,
        "theta": theta,
        "eval_index": eval_index,
        "tmin": tmin,
        "tmax": tmax,
        "stacking": "[xi_p(theta); xi_m(theta)]",
    }
