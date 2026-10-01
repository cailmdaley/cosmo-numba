"""Interpolation in 1D

Implementation of the Akima spline interpolation
Taken from https://github.com/cgohlke/akima adapted to a class and port to
numba.

Implementation of 1D interpolation taken from
https://github.com/dbstein/fast_interp and adapted to a numba class that can be
used in jitted functions.. There is also a functionnal version to be used
within quad integration.

Author: Axel Guinot

"""

import os

import numpy as np
import numba as nb

from ..utils import numbadiff

# This allows having coverage
if (
    os.environ.get("TESTING_INTERP", "0") == "1"
    and os.environ.get("COVERAGE_MODE", "0") == "1"
):
    nb.config.DISABLE_JIT = 1


class _AkimaInterp1D:
    """
    Akima's interpolation in 1D.
    This implementation comes from https://github.com/cgohlke/akima ported to
    numba and adapted to a class.
    Extrapolation is set to 0.0 by default but can be changed with the
    `update_extrapolation` method.

    Parameters
    ----------
    x : array_like
        Data points, must be increasing.
    y : array_like
        Data values at data points x.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.left = 0.0
        self.right = 0.0

        self.n = len(x)
        if self.n < 3:
            raise ValueError("array too small")
        if self.n != len(y):
            raise ValueError("size of x-array must match data shape")

        dx = numbadiff(self.x)
        if np.any(dx <= 0.0):
            raise ValueError("x-axis not valid")

        m = numbadiff(self.y) / dx
        mm = 2.0 * m[0] - m[1]
        mmm = 2.0 * mm - m[0]
        mp = 2.0 * m[self.n - 2] - m[self.n - 3]
        mpp = 2.0 * mp - m[self.n - 2]

        mm = np.array([mm])
        mmm = np.array([mmm])
        mp = np.array([mp])
        mpp = np.array([mpp])

        m1 = np.concatenate((mmm, mm, m, mp, mpp))

        dm = np.abs(numbadiff(m1))
        f1 = dm[2 : self.n + 2]
        f2 = dm[0 : self.n]
        f12 = f1 + f2

        ids = np.nonzero(f12 > 1e-9 * np.max(f12))[0]
        self.b = m1[1 : self.n + 1]

        self.b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[
            ids
        ]
        self.c = (
            3.0 * m - 2.0 * self.b[0 : self.n - 1] - self.b[1 : self.n]
        ) / dx
        self.d = (
            self.b[0 : self.n - 1] + self.b[1 : self.n] - 2.0 * m
        ) / dx**2

    def update_extrapolation(self, left, right):
        """
        Update the values to return when extrapolating.
        We have to do it this way because numba jitclass do not support keyword
        arguments in methods.

        Parameters
        ----------
        left, right : float
            Values to return for when extrapolating.
        """
        self.left = left
        self.right = right

    def eval(self, xout):
        """
        Return the interpolated values at xout.

        Parameters
        ----------
        xout : array_like
            Points where to interpolate.

        Returns
        -------
        yout : ndarray
            Interpolated values at xout.
        """
        lm = xout < self.x[0]
        rm = xout > self.x[-1]
        totm = np.invert(lm | rm)

        yout = np.zeros_like(xout)

        # Do something better for extrapolation
        yout[lm] = self.left
        yout[rm] = self.right

        bins = np.digitize(xout[totm], self.x)
        bins = np.minimum(bins, self.n - 1) - 1
        bb = bins[0 : len(xout[totm])]
        wj = xout[totm] - self.x[bb]

        yout[totm] = (
            (wj * self.d[bb] + self.c[bb]) * wj + self.b[bb]
        ) * wj + self.y[bb]

        return yout


# We jit the class as it could be useful to have access to the ortiginal python
# class for non-jitted code.
spec_akima = [
    ("x", nb.float64[:]),
    ("y", nb.float64[:]),
    ("left", nb.float64),
    ("right", nb.float64),
    ("n", nb.int64),
    ("b", nb.float64[:]),
    ("c", nb.float64[:]),
    ("d", nb.float64[:]),
]

AkimaInterp1D = nb.experimental.jitclass(spec_akima)(_AkimaInterp1D)


@nb.njit(
    nb.void(
        nb.float64[:],  # f
        nb.float64[:],  # xout
        nb.float64[:],  # fout
        nb.float64,  # a
        nb.float64,  # h
        nb.int64,  # n
        nb.boolean,  # p
        nb.int64,  # o
        nb.float64,  # lb
        nb.float64,  # ub
    ),
    fastmath=True,
)
def _interp1d_k1(f, xout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb), ub)
        xx = xr - a
        ix = int(xx // h)
        ratx = xx / h - (ix + 0.5)
        asx = np.empty(2)
        asx[0] = 0.5 - ratx
        asx[1] = 0.5 + ratx
        ix += o
        fout[mi] = 0.0
        for i in range(2):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi] * asx[i]


@nb.njit(
    nb.void(
        nb.float64[:],  # f
        nb.float64[:],  # xout
        nb.float64[:],  # fout
        nb.float64,  # a
        nb.float64,  # h
        nb.int64,  # n
        nb.boolean,  # p
        nb.int64,  # o
        nb.float64,  # lb
        nb.float64,  # ub
    ),
    fastmath=True,
)
def _interp1d_k3(f, xout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb), ub)
        xx = xr - a
        ix = int(xx // h)
        ratx = xx / h - (ix + 0.5)
        asx = np.empty(4)
        asx[0] = -1/16 + ratx*( 1/24 + ratx*( 1/4 - ratx/6))  # noqa # fmt: skip
        asx[1] =  9/16 + ratx*( -9/8 + ratx*(-1/4 + ratx/2))  # noqa # fmt: skip
        asx[2] =  9/16 + ratx*(  9/8 + ratx*(-1/4 - ratx/2))  # noqa # fmt: skip
        asx[3] = -1/16 + ratx*(-1/24 + ratx*( 1/4 + ratx/6))  # noqa # fmt: skip
        ix += o - 1
        fout[mi] = 0.0
        for i in range(4):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi] * asx[i]


@nb.njit(
    nb.void(
        nb.float64[:],  # f
        nb.float64[:],  # xout
        nb.float64[:],  # fout
        nb.float64,  # a
        nb.float64,  # h
        nb.int64,  # n
        nb.boolean,  # p
        nb.int64,  # o
        nb.float64,  # lb
        nb.float64,  # ub
    ),
    fastmath=True,
)
def _interp1d_k5(f, xout, fout, a, h, n, p, o, lb, ub):
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb), ub)
        xx = xr - a
        ix = int(xx // h)
        ratx = xx / h - (ix + 0.5)
        asx = np.empty(6)
        asx[0] =   3/256 + ratx*(   -9/1920 + ratx*( -5/48/2 + ratx*(  1/8/6 + ratx*( 1/2/24 -  1/8/120*ratx))))  # noqa # fmt: skip # fmt: skip
        asx[1] = -25/256 + ratx*(  125/1920 + ratx*( 39/48/2 + ratx*(-13/8/6 + ratx*(-3/2/24 +  5/8/120*ratx))))  # noqa # fmt: skip
        asx[2] = 150/256 + ratx*(-2250/1920 + ratx*(-34/48/2 + ratx*( 34/8/6 + ratx*( 2/2/24 - 10/8/120*ratx))))  # noqa # fmt: skip
        asx[3] = 150/256 + ratx*( 2250/1920 + ratx*(-34/48/2 + ratx*(-34/8/6 + ratx*( 2/2/24 + 10/8/120*ratx))))  # noqa # fmt: skip
        asx[4] = -25/256 + ratx*( -125/1920 + ratx*( 39/48/2 + ratx*( 13/8/6 + ratx*(-3/2/24 -  5/8/120*ratx))))  # noqa # fmt: skip
        asx[5] =   3/256 + ratx*(    9/1920 + ratx*( -5/48/2 + ratx*( -1/8/6 + ratx*( 1/2/24 +  1/8/120*ratx))))  # noqa # fmt: skip
        ix += o - 2
        fout[mi] = 0.0
        for i in range(6):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi] * asx[i]


@nb.njit(
    nb.void(
        nb.float64[:],  # f
        nb.float64[:],  # xout
        nb.float64[:],  # fout
        nb.float64,  # a
        nb.float64,  # h
        nb.int64,  # n
        nb.boolean,  # p
        nb.int64,  # o
        nb.float64,  # lb
        nb.float64,  # ub
    ),
    fastmath=True,
)
def _interp1d_k7(f, xout, fout, a, h, n, p, o, lb, ub):  # pragma: no cover
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb), ub)
        xx = xr - a
        ix = int(xx // h)
        ratx = xx / h - (ix + 0.5)
        asx = np.empty(8)
        asx[0] =   -5/2048 + ratx*(     75/107520 + ratx*(  259/11520/2 + ratx*(  -37/1920/6 + ratx*(  -7/48/24 + ratx*(   5/24/120 + ratx*( 1/2/720 -  1/5040*ratx))))))  # noqa # fmt: skip
        asx[1] =   49/2048 + ratx*(  -1029/107520 + ratx*(-2495/11520/2 + ratx*(  499/1920/6 + ratx*(  59/48/24 + ratx*( -59/24/120 + ratx*(-5/2/720 +  7/5040*ratx))))))  # noqa # fmt: skip
        asx[2] = -245/2048 + ratx*(   8575/107520 + ratx*(11691/11520/2 + ratx*(-3897/1920/6 + ratx*(-135/48/24 + ratx*( 225/24/120 + ratx*( 9/2/720 - 21/5040*ratx))))))  # noqa # fmt: skip
        asx[3] = 1225/2048 + ratx*(-128625/107520 + ratx*(-9455/11520/2 + ratx*( 9455/1920/6 + ratx*(  83/48/24 + ratx*(-415/24/120 + ratx*(-5/2/720 + 35/5040*ratx))))))  # noqa # fmt: skip
        asx[4] = 1225/2048 + ratx*( 128625/107520 + ratx*(-9455/11520/2 + ratx*(-9455/1920/6 + ratx*(  83/48/24 + ratx*( 415/24/120 + ratx*(-5/2/720 - 35/5040*ratx))))))  # noqa # fmt: skip
        asx[5] = -245/2048 + ratx*(  -8575/107520 + ratx*(11691/11520/2 + ratx*( 3897/1920/6 + ratx*(-135/48/24 + ratx*(-225/24/120 + ratx*( 9/2/720 + 21/5040*ratx))))))  # noqa # fmt: skip
        asx[6] =   49/2048 + ratx*(   1029/107520 + ratx*(-2495/11520/2 + ratx*( -499/1920/6 + ratx*(  59/48/24 + ratx*(  59/24/120 + ratx*(-5/2/720 -  7/5040*ratx))))))  # noqa # fmt: skip
        asx[7] =   -5/2048 + ratx*(    -75/107520 + ratx*(  259/11520/2 + ratx*(   37/1920/6 + ratx*(  -7/48/24 + ratx*(  -5/24/120 + ratx*( 1/2/720 +  1/5040*ratx))))))  # noqa # fmt: skip
        ix += o - 3
        fout[mi] = 0.0
        for i in range(8):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi] * asx[i]


@nb.njit(
    nb.void(
        nb.float64[:],  # f
        nb.float64[:],  # xout
        nb.float64[:],  # fout
        nb.float64,  # a
        nb.float64,  # h
        nb.int64,  # n
        nb.boolean,  # p
        nb.int64,  # o
        nb.float64,  # lb
        nb.float64,  # ub
    ),
    fastmath=True,
)
def _interp1d_k9(f, xout, fout, a, h, n, p, o, lb, ub):  # pragma: no cover
    m = fout.shape[0]
    for mi in nb.prange(m):
        xr = min(max(xout[mi], lb), ub)
        xx = xr - a
        ix = int(xx // h)
        ratx = xx / h - (ix + 0.5)
        asx = np.empty(10)
        asx[0] =    35/65536 + ratx*(    -1225/10321920 + ratx*(  -3229/645120/2 + ratx*(    3229/967680/6 + ratx*(   141/3840/24 + ratx*(   -47/1152/120 + ratx*(  -3/16/720 + ratx*(    7/24/5040 + ratx*(  1/2/40320 -   1/362880*ratx))))))))  # noqa # fmt: skip
        asx[1] =  -405/65536 + ratx*(    18225/10321920 + ratx*(  37107/645120/2 + ratx*(  -47709/967680/6 + ratx*( -1547/3840/24 + ratx*(   663/1152/120 + ratx*(  29/16/720 + ratx*(  -87/24/5040 + ratx*( -7/2/40320 +   9/362880*ratx))))))))  # noqa # fmt: skip
        asx[2] =  2268/65536 + ratx*(  -142884/10321920 + ratx*(-204300/645120/2 + ratx*(  367740/967680/6 + ratx*(  7540/3840/24 + ratx*( -4524/1152/120 + ratx*(-100/16/720 + ratx*(  420/24/5040 + ratx*( 20/2/40320 -  36/362880*ratx))))))))  # noqa # fmt: skip
        asx[3] = -8820/65536 + ratx*(   926100/10321920 + ratx*( 745108/645120/2 + ratx*(-2235324/967680/6 + ratx*(-14748/3840/24 + ratx*( 14748/1152/120 + ratx*( 156/16/720 + ratx*(-1092/24/5040 + ratx*(-28/2/40320 +  84/362880*ratx))))))))  # noqa # fmt: skip
        asx[4] = 39690/65536 + ratx*(-12502350/10321920 + ratx*(-574686/645120/2 + ratx*( 5172174/967680/6 + ratx*(  8614/3840/24 + ratx*(-25842/1152/120 + ratx*( -82/16/720 + ratx*( 1722/24/5040 + ratx*( 14/2/40320 - 126/362880*ratx))))))))  # noqa # fmt: skip
        asx[5] = 39690/65536 + ratx*( 12502350/10321920 + ratx*(-574686/645120/2 + ratx*(-5172174/967680/6 + ratx*(  8614/3840/24 + ratx*( 25842/1152/120 + ratx*( -82/16/720 + ratx*(-1722/24/5040 + ratx*( 14/2/40320 + 126/362880*ratx))))))))  # noqa # fmt: skip
        asx[6] = -8820/65536 + ratx*(  -926100/10321920 + ratx*( 745108/645120/2 + ratx*( 2235324/967680/6 + ratx*(-14748/3840/24 + ratx*(-14748/1152/120 + ratx*( 156/16/720 + ratx*( 1092/24/5040 + ratx*(-28/2/40320 -  84/362880*ratx))))))))  # noqa # fmt: skip
        asx[7] =  2268/65536 + ratx*(   142884/10321920 + ratx*(-204300/645120/2 + ratx*( -367740/967680/6 + ratx*(  7540/3840/24 + ratx*(  4524/1152/120 + ratx*(-100/16/720 + ratx*( -420/24/5040 + ratx*( 20/2/40320 +  36/362880*ratx))))))))  # noqa # fmt: skip
        asx[8] =  -405/65536 + ratx*(   -18225/10321920 + ratx*(  37107/645120/2 + ratx*(   47709/967680/6 + ratx*( -1547/3840/24 + ratx*(  -663/1152/120 + ratx*(  29/16/720 + ratx*(   87/24/5040 + ratx*( -7/2/40320 -   9/362880*ratx))))))))  # noqa # fmt: skip
        asx[9] =    35/65536 + ratx*(     1225/10321920 + ratx*(  -3229/645120/2 + ratx*(   -3229/967680/6 + ratx*(   141/3840/24 + ratx*(    47/1152/120 + ratx*(  -3/16/720 + ratx*(   -7/24/5040 + ratx*(  1/2/40320 +   1/362880*ratx))))))))  # noqa # fmt: skip
        ix += o - 4
        fout[mi] = 0.0
        for i in range(10):
            ixi = (ix + i) % n if p else ix + i
            fout[mi] += f[ixi] * asx[i]


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64))(
        nb.float64,
        nb.float64,
        nb.float64,
        nb.boolean,
        nb.boolean,
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _compute_bounds1(a, b, h, p, c, e, k):
    if p:
        return -1e100, 1e100
    elif not c:
        d = h * (k // 2)
        return a + d, b - d
    else:
        d = e * h
        # Keep both clamped end points strictly inside the padded grid:
        # rounding at an end point can otherwise select a cell whose stencil
        # reads past the padded array.
        eps = 1e-12 * (h + abs(a) + abs(b))
        return a - d + eps, b + d - eps


@nb.njit(
    nb.void(
        nb.float64[:],
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def _extrapolate1d_x(f, k, o):
    for ix in range(o):
        il = o - ix - 1
        ih = f.shape[0] - (o - ix)
        if k == 1:
            f[il] = 2 * f[il + 1] - 1 * f[il + 2]
            f[ih] = 2 * f[ih - 1] - 1 * f[ih - 2]
        if k == 3:
            f[il] = 4 * f[il + 1] - 6 * f[il + 2] + 4 * f[il + 3] - f[il + 4]
            f[ih] = 4 * f[ih - 1] - 6 * f[ih - 2] + 4 * f[ih - 3] - f[ih - 4]
        if k == 5:
            f[il] = 6*f[il+1]-15*f[il+2]+20*f[il+3]-15*f[il+4]+6*f[il+5]-f[il+6]  # noqa # fmt: skip
            f[ih] = 6*f[ih-1]-15*f[ih-2]+20*f[ih-3]-15*f[ih-4]+6*f[ih-5]-f[ih-6]  # noqa # fmt: skip
        if k == 7:  # pragma: no cover
            f[il] = 8*f[il+1]-28*f[il+2]+56*f[il+3]-70*f[il+4]+56*f[il+5]-28*f[il+6]+8*f[il+7]-f[il+8]  # noqa # fmt: skip
            f[ih] = 8*f[ih-1]-28*f[ih-2]+56*f[ih-3]-70*f[ih-4]+56*f[ih-5]-28*f[ih-6]+8*f[ih-7]-f[ih-8]  # noqa # fmt: skip
        if k == 9:  # pragma: no cover
            f[il] = 10*f[il+1]-45*f[il+2]+120*f[il+3]-210*f[il+4]+252*f[il+5]-210*f[il+6]+120*f[il+7]-45*f[il+8]+10*f[il+9]-f[il+10]  # noqa # fmt: skip
            f[ih] = 10*f[ih-1]-45*f[ih-2]+120*f[ih-3]-210*f[ih-4]+252*f[ih-5]-210*f[ih-6]+120*f[ih-7]-45*f[ih-8]+10*f[ih-9]-f[ih-10]  # noqa # fmt: skip


@nb.njit(
    nb.void(
        nb.float64[:],
        nb.float64[:],
        nb.int64,
    ),
    fastmath=True,
)
def _fill1(f, fb, o):
    fb[o : o + f.shape[0]] = f


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.int64))(
        nb.float64[:],
        nb.int64,
        nb.boolean,
        nb.boolean,
        nb.int64,
    ),
    fastmath=True,
)
def _extrapolate1d(f, k, p, c, e):
    pad = (not p) and c
    if pad:
        o = (k // 2) + e
        fb = np.empty(f.shape[0] + 2 * o, dtype=f.dtype)
        _fill1(f, fb, o)
        _extrapolate1d_x(fb, k, o)
        return fb, o
    else:
        return f, 0


class _nb_interp1d:
    """
    1D Interpolator using local Taylor expansions
    This implementation comes from https://github.com/dbstein/fast_interp
    and adapted to a numba class that can be used in jitted functions

    Parameters
    ----------
    a : float
        The lower bound of the interpolation region
    b : float
        The upper bound of the interpolation region
    h : float
        The grid-spacing at which f is given
    f : ndarray
        Data to be interpolated
    k : int
        Order of local Taylor expansions (1, 3, 5, 7, or 9)
    p : bool, optional
        Whether the dimension is taken to be periodic (default is False)
    c : bool, optional
        Whether the array should be padded to allow accurate close eval
        (default is True)
    e : int, optional
        Extrapolation distance, how far to allow extrap, in units of h
        (needs to be an integer, default is 0)
    log_interp : bool, optional
        Whether to perform the interpolation in log-space (default is False)
    """

    def __init__(
        self,
        a,
        b,
        h,
        f,
        k,
        p=False,
        c=True,
        e=0,
        log_interp=False,
    ):
        self.a = a
        self.b = b
        self.h = h
        self.f = f
        self.k = k
        self.p = p
        self.c = c
        self.e = e
        self.log_interp = log_interp
        self.n = f.shape[0]
        _f, _o = _extrapolate1d(f, k, p, c, e)
        self._f = _f
        self._o = _o
        lb, ub = _compute_bounds1(a, b, h, p, c, e, k)
        self.lb = lb
        self.ub = ub

    def eval(self, xout):
        """
        Return the interpolated values at xout.

        Parameters
        ----------
        xout : array_like
            Points where to interpolate.

        Returns
        -------
        out : ndarray
            Interpolated values at xout.
        """

        m = int(np.prod(np.array(xout.shape)))

        self.xout = np.log(xout) if self.log_interp else xout
        self.out = np.empty(m, dtype=self.f.dtype)

        if self.k == 1:
            self._interp1d_k1()
        elif self.k == 3:
            self._interp1d_k3()
        elif self.k == 5:
            self._interp1d_k5()
        elif self.k == 7:  # pragma: no cover
            self._interp1d_k7()
        elif self.k == 9:  # pragma: no cover
            self._interp1d_k9()
        else:
            raise ValueError(f"No interpolation for k={self.k}")

        return self.out

    def _interp1d_k1(self):
        _interp1d_k1(
            self._f,
            self.xout,
            self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb,
            self.ub,
        )

    def _interp1d_k3(self):
        _interp1d_k3(
            self._f,
            self.xout,
            self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb,
            self.ub,
        )

    def _interp1d_k5(self):
        _interp1d_k5(
            self._f,
            self.xout,
            self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb,
            self.ub,
        )

    def _interp1d_k7(self):  # pragma: no cover
        _interp1d_k7(
            self._f,
            self.xout,
            self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb,
            self.ub,
        )

    def _interp1d_k9(self):  # pragma: no cover
        _interp1d_k9(
            self._f,
            self.xout,
            self.out,
            self.a,
            self.h,
            self.n,
            self.p,
            self._o,
            self.lb,
            self.ub,
        )


# We jit the class as it could be useful to have access to the ortiginal python
# class for non-jitted code.
spec_1d = [
    ("a", nb.float64),
    ("b", nb.float64),
    ("h", nb.float64),
    ("f", nb.float64[:]),
    ("k", nb.int64),
    ("p", nb.boolean),
    ("c", nb.boolean),
    ("e", nb.int64),
    ("log_interp", nb.boolean),
    ("n", nb.int64),
    ("_f", nb.float64[:]),
    ("_o", nb.int64),
    ("lb", nb.float64),
    ("ub", nb.float64),
    ("xout", nb.float64[:]),
    ("out", nb.float64[:]),
]


nb_interp1d = nb.experimental.jitclass(spec_1d)(_nb_interp1d)


#########
#########
# This functionnal form of nb_interp1d is meant to be use for integration
# purposes. I would recommand to use the class above in a general case.
spec_interp = nb.float64(
    nb.float64,
    nb.types.CPointer(nb.float64),
)


@nb.njit
def nb_interp1d_func(
    xout,
    data,
):
    """
    This function does the same thing as the class above but wrapped in a way
    that it can be called for integration purpopses.

    Here `data` contains all the information needed to make the interpolation.
    It is built following:
    - data[0] = len(f)
    - data[1 : len(f)] = f
    - data[1 + len(f) + 0] = a
    - data[1 + len(f) + 1] = b
    - data[1 + len(f) + 2] = h
    - data[1 + len(f) + 3] = k
    - data[1 + len(f) + 4] = p
    - data[1 + len(f) + 5] = c
    - data[1 + len(f) + 6] = e
    - data[1 + len(f) + 7] = log_interp

    with a, b, h, k, p, c, e defined in the class above.
    log_interp tells if `x` used for the interpolation is provided in log
    space. In that case we need to know to convert the input `xout` for the
    interpolation. This is useful for a case where your function
    logarithmically spaced.

    Parameters
    ----------
    xout : float
        Point where to interpolate.
    data : CPointer to float64
        Data needed to make the interpolation (see above).
    """

    # data is given as a Cpointer so we cannot use `.shape` or slicing which
    # why we to know the size in advance and we also use "C-style" assignement
    # with a for loop.
    len_f = int(data[0])
    f = np.empty(len_f, dtype=np.float64)
    for i in range(1, len_f + 1):
        f[i - 1] = data[i]
    a = data[1 + len_f + 0]
    b = data[1 + len_f + 1]
    h = data[1 + len_f + 2]
    k = np.int64(data[1 + len_f + 3])
    p = bool(data[1 + len_f + 4])
    c = bool(data[1 + len_f + 5])
    e = np.int64(data[1 + len_f + 6])
    log_interp = bool(data[1 + len_f + 7])

    n = f.shape[0]
    _f, _o = _extrapolate1d(f, k, p, c, e)
    lb, ub = _compute_bounds1(a, b, h, p, c, e, k)

    xout = np.array([np.log(xout)]) if log_interp else np.array([xout])

    m = int(np.prod(np.array(xout.shape)))

    out = np.empty(m, dtype=np.float64)

    if k == 1:
        _interp1d_k1(_f, xout, out, a, h, n, p, _o, lb, ub)
    elif k == 3:
        _interp1d_k3(_f, xout, out, a, h, n, p, _o, lb, ub)
    elif k == 5:
        _interp1d_k5(_f, xout, out, a, h, n, p, _o, lb, ub)
    elif k == 7:  # pragma: no cover
        _interp1d_k7(_f, xout, out, a, h, n, p, _o, lb, ub)
    elif k == 9:  # pragma: no cover
        _interp1d_k9(_f, xout, out, a, h, n, p, _o, lb, ub)

    return out[0]


if not nb.config.DISABLE_JIT:  # pragma: no cover
    nb_interp1d_cfunc = nb.cfunc(spec_interp)(nb_interp1d_func)
