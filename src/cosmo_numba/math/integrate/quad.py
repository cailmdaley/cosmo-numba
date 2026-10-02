"""Quad integration

Implementation of the quad integration based on NumbaQuadpack
https://github.com/Nicholaswogan/NumbaQuadpack. The dqags has been adapted to
use interpolation which allows the the integration to work on "any" functions.
It can also be called from within jitted functions. `interp_quad_weights`
integrates the same interpolant with a fixed rule, as weights on the samples.

Author: Axel Guinot

"""

import os

from itertools import product

import numba as nb
import numpy as np
from NumbaQuadpack import dqags

from ..interpolate.interpolate_1D import (
    _compute_bounds1,
    _extrapolate1d_adjoint,
    _locate,
    nb_interp1d_cfunc,
    spec_interp,
    stencil_weights,
)

# This allows having coverage
IN_COVERAGE = False
if (os.environ.get("TESTING_QUAD", "0") == "1") and os.environ.get(
    "COVERAGE_MODE", "0"
) == "1":
    IN_COVERAGE = True


INTERP_ADDRESS = nb.experimental.function_type._get_wrapper_address(
    nb_interp1d_cfunc,
    spec_interp,
)


def make_signature(output_sigs, *args):
    """
    Create function signatures for Numba JIT compilation.

    Parameters
    ----------
    output_sigs : list of str
        Signature of the output type.
    *args : list of list of str
        Each list contains possible types for each argument.

    Returns
    -------
    list of str
        List of function signatures.
    """
    output_comb = list(product(*args, repeat=1))
    sig_final = []
    for output_sig in output_sigs:
        for output in output_comb:
            sig_tmp = ", ".join(output)
            sig_tmp = f"({sig_tmp})"
            sig_tmp = output_sig + sig_tmp
            sig_final.append(sig_tmp)
    return sig_final


# Here we use a constructor for the signature due to the high number of
# optional parameters. It would make the code very hard to read otherwise.
# To see the full signature of the function, one can call
# `interp_quad.signature` from python.
spec_interp_quad = make_signature(
    ["Tuple((float64, float64, boolean))"],
    ["float64"],
    ["float64"],
    ["float64"],
    ["float64[::1]"],
    ["float64"],
    ["float64"],
    ["int64", "Omitted(3)"],
    ["boolean", "Omitted(False)"],
    ["boolean", "Omitted(True)"],
    ["int64", "Omitted(0)"],
    ["boolean", "Omitted(False)"],
    ["float64", "Omitted(1.49e-8)"],
    ["float64", "Omitted(1.49e-8)"],
)


@nb.njit(spec_interp_quad)
def interp_quad(
    x_start,
    x_end,
    x_step,
    fx,
    a,
    b,
    k=3,
    periodic=False,
    padding=True,
    extrap_dist=0,
    log_interp=False,
    epsabs=1.49e-8,
    epsrel=1.49e-8,
):
    """interp_quad

    Perform an interpolation of degree `k` of the provided array `fx` padded on
    the regular grid `[x_start:x_step:x_end]`. If `fx` is padded on a
    logarithmically spaced grid, one can set `log_interp=True`.

    NOTE: only works on definite bounds and regular grid either in real or log
    space.
    NOTE: For some reason when providing a signature, omitting some parameters
    return an error (`k`, `padding`). Also, not providing the `padding`
    parameter makes the computation longer.

    Parameters
    ----------
    x_start : float
        Grid start.
    x_end : float
        Grid end.
    x_step : _type_
        Grid step.
    fx : numpy.ndarray
        Padded function to integrate.
    a : float
        Lower bound of the integral.
    b : float
        Upper bound of the integral.
    k : int, optional
        Degree of interpolation, by default 3
    periodic : bool, optional
        See interp1d, by default False
    padding : bool, optional
        See interp1d, by default True
    extrap_dist : int, optional
        See interp1d, by default 0
    log_interp : bool, optional
        See interp1d, by default False
    epsabs : float, optional
        Absolute error tolerance, by default 1.49e-8
    epsrel : float, optional
        Relative error tolerance, by default 1.49e-8

    Returns
    -------
    tuple
        Tuple with result, abserr, success.
    """

    len_fx = len(fx)

    data = np.empty(len_fx + 9, dtype=np.float64)
    data[0] = np.float64(len_fx)
    for i in range(len_fx):
        data[i + 1] = fx[i]
    data[len_fx + 1] = x_start
    data[len_fx + 2] = x_end
    data[len_fx + 3] = x_step
    data[len_fx + 4] = np.float64(k)
    data[len_fx + 5] = np.float64(periodic)
    data[len_fx + 6] = np.float64(padding)
    data[len_fx + 7] = np.float64(extrap_dist)
    data[len_fx + 8] = np.float64(log_interp)

    res = dqags(
        INTERP_ADDRESS,
        a,
        b,
        data=data,
        epsabs=epsabs,
        epsrel=epsrel,
    )

    return res


# Gauss-Legendre rule used by `interp_quad_weights` on each interpolation
# cell.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


@nb.njit
def _cell_quad(s0, s1, h, k, log_interp):
    """Integral of the stencil weights over `[s0, s1]` in one cell.

    Returns `int_{s0}^{s1} asx(s - 1/2) ds`, times `exp(s * h)` in the
    integrand if `log_interp`, with `s` the position in the cell in units of
    the grid step `h` and `asx` the weights of `stencil_weights`.
    """
    out = np.zeros(k + 1)
    asx = np.empty(k + 1)
    half = 0.5 * (s1 - s0)
    for q in range(_GL_NODES.shape[0]):
        s = s0 + half * (1.0 + _GL_NODES[q])
        wq = half * _GL_WEIGHTS[q]
        if log_interp:
            wq *= np.exp(s * h)
        stencil_weights(s - 0.5, k, asx)
        out += wq * asx
    return out


@nb.njit
def interp_quad_weights(
    x_start,
    x_end,
    x_step,
    n,
    a,
    b,
    k=3,
    padding=True,
    extrap_dist=0,
    log_interp=False,
):
    """interp_quad_weights

    Fixed-rule counterpart of `interp_quad`: returns the weights `w` such that
    `w @ fx` is the integral from `a` to `b` of the interpolant that
    `interp_quad` integrates, for any `fx` of length `n`.

    On each grid cell the interpolant is a polynomial of degree `k` in `x`
    (in `log(x)` if `log_interp`, where the integrand gains a factor `x`).
    Each cell is integrated with a 16-point Gauss-Legendre rule, exact to
    round-off for any `k` and any grid step up to about one e-fold; beyond
    the clamping bounds of the interpolator the interpolant is constant and
    is integrated in closed form. The weights do not depend on `fx`, so,
    unlike the adaptive `interp_quad`, the integral is exactly linear in
    `fx`.

    Fewer than `k + 1` samples leave the interpolant undetermined: the
    weights are then NaN. Periodic interpolation is not supported.

    Parameters
    ----------
    x_start : float
        Grid start.
    x_end : float
        Grid end.
    x_step : float
        Grid step.
    n : int
        Number of samples.
    a : float
        Lower bound of the integral.
    b : float
        Upper bound of the integral.
    k : int, optional
        Degree of interpolation, by default 3
    padding : bool, optional
        See interp1d, by default True
    extrap_dist : int, optional
        See interp1d, by default 0
    log_interp : bool, optional
        See interp1d, by default False

    Returns
    -------
    numpy.ndarray
        Weights, of length `n`.
    """
    if n < k + 1:
        return np.full(n, np.nan)
    h = x_step
    o = k // 2 + extrap_dist if padding else 0
    lb, ub = _compute_bounds1(
        x_start, x_end, h, False, padding, extrap_dist, k
    )
    if log_interp:
        xa, xb = np.log(a), np.log(b)
    else:
        xa, xb = a, b
    sign = 1.0
    if xb < xa:
        xa, xb, sign = xb, xa, -1.0

    w = np.zeros(n + 2 * o)
    asx = np.empty(k + 1)

    # Beyond the clamping bounds, the value at the bound.
    for x_clip, lo, hi in ((lb, xa, min(xb, lb)), (ub, max(xa, ub), xb)):
        if hi > lo:
            ix, ratx = _locate(x_clip, x_start, h, n, k, False, o, lb, ub)
            stencil_weights(ratx, k, asx)
            i0 = ix + o - k // 2
            length = np.exp(hi) - np.exp(lo) if log_interp else hi - lo
            w[i0 : i0 + k + 1] += length * asx

    # In between, cell by cell, in the cells the interpolator evaluates in;
    # whole cells share one set of weights.
    lo, hi = max(xa, lb), min(xb, ub)
    if hi > lo:
        c_lo = _locate(lo, x_start, h, n, k, False, o, lb, ub)[0]
        c_hi = _locate(hi, x_start, h, n, k, False, o, lb, ub)[0]
        whole = _cell_quad(0.0, 1.0, h, k, log_interp)
        for ix in range(c_lo, c_hi + 1):
            s0 = (lo - x_start) / h - ix if ix == c_lo else 0.0
            s1 = (hi - x_start) / h - ix if ix == c_hi else 1.0
            if s0 == 0.0 and s1 == 1.0:
                cell = whole
            else:
                cell = _cell_quad(s0, s1, h, k, log_interp)
            scale = h * np.exp(x_start + ix * h) if log_interp else h
            i0 = ix + o - k // 2
            for i in range(k + 1):
                w[i0 + i] += scale * cell[i]

    return sign * _extrapolate1d_adjoint(w, k, o)
