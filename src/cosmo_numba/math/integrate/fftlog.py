"""FFTLog Numba

First imlplemented by Andrew Hamilton:

https://jila.colorado.edu/~ajsh/FFTLog/

Adapted from C in CCL:
https://github.com/LSSTDESC/CCL/blob/master/src/ccl_fftlog.c

Author: Axel Guinot

"""

import os

import numpy as np
import numba as nb

from ..utils import lngamma

# This allows having coverage
if (
    os.environ.get("TESTING_FFTLOG", "0") == "1"
    and os.environ.get("COVERAGE_MODE", "0") == "1"
):
    nb.config.DISABLE_JIT = 1


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64))(nb.float64, nb.float64),
    fastmath=True,
)
def lngamma_4(x, y):
    """
    Wrapper around lngamma to return real and imaginary parts separately.

    Parameters
    ----------
    x : float64
        real part
    y : float64
        imaginary part

    Returns
    -------
    float64, float64
        real and imaginary parts of lngamma(x + i y)
    """
    out = lngamma(x + y * 1j)
    return np.real(out), np.imag(out)


@nb.njit(
    nb.complex128(nb.float64, nb.float64),
    fastmath=True,
)
def polar(r, phi):
    """
    Create a complex number from polar coordinates.

    Parameters
    ----------
    r : float64
        modulus
    phi : float64
        phase

    Returns
    -------
    complex128
        complex number
    """
    res = r * np.cos(phi) + r * np.sin(phi) * 1j
    return res


@nb.njit(
    nb.float64(
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
    ),
    fastmath=True,
)
def goodkr(N, mu, q, L, kr):
    """
    Find a good value of kr to minimize ringing.

    Parameters
    ----------
    N : int64
        number of points
    mu : int64
        mu parameter
    q : int64
        q parameter
    L : int64
        L parameter
    kr : float64
        initial kr value
    Returns
    -------
    float64
        adjusted kr value
    """
    xp = (mu + 1 + q) / 2.0
    xm = (mu + 1 - q) / 2.0
    y = np.pi * N / (2.0 * L)
    lnr, argp = lngamma_4(xp, y)
    lnr, argm = lngamma_4(xm, y)
    arg = np.log(2 / kr) * N / L + (argp + argm) / np.pi
    iarg = np.round(arg)
    if arg != iarg:
        kr *= np.exp((arg - iarg) * L / N)
    return kr


@nb.njit(
    nb.void(
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.complex128[:],
    ),
    fastmath=True,
)
def compute_u_coeff(N, mu, q, L, kcrc, u):
    """
    Compute the u_m coefficients for FFTLog. These coefficients encode the
    Bessel function convolution in Fourier space.

    Parameters
    ----------
    N : int64
        Number of points in the input arrays.
    mu : int64
        Order of the Bessel function in the Hankel transform.
    q : float64
        Bias parameter controlling the power-law tilt.
    L : float64
        Logarithmic interval size.
    kcrc : float64
        Parameter controlling the mapping between k and r (usually set to 1.0).
    u : complex128[:]
        Output array to store the u_m coefficients.
    """
    y = np.pi / L
    k0r0 = kcrc * np.exp(-L)
    t = -2.0 * y * np.log(k0r0 / 2.0)

    if q == 0:
        x = (mu + 1) / 2.0
        for m in range(0, int(N / 2) + 1):
            lnr, phi = lngamma_4(x, m * y)
            u[m] = polar(1.0, m * t + 2 * phi)
    else:
        xp = (mu + 1 + q) / 2.0
        xm = (mu + 1 - q) / 2.0
        for m in range(0, int(N / 2) + 1):
            lnrp, phip = lngamma_4(xp, m * y)
            lnrm, phim = lngamma_4(xm, -m * y)
            u[m] = polar(
                np.exp(q * np.log(2.0) + lnrp - lnrm), m * t + phip - phim
            )

    for m in range(int(N / 2) + 1, N):
        u[m] = np.conj(u[N - m])
    if np.mod(N, 2) == 0:
        u[int(N / 2)] = np.real(u[int(N / 2)]) + 0.0 * 1j


@nb.njit(
    nb.types.Tuple(
        (
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
            nb.types.Array(nb.float64, 1, "C", False, aligned=True),
        )
    )(
        nb.int64,
        nb.float64[:],
        nb.float64[:],
        nb.int64,
        nb.int64,
        nb.float64,
        nb.int64,
        nb.int64,
    ),
    fastmath=True,
)
def fht(N, k, pk, dim, mu, q_, kcrc, noring):
    """
    Perform a FFTLog-based Hankel transform.

    Parameters
    ----------
    N : int64
        Number of points in the input arrays.
    k : float64[:]
        Wavenumber array
    pk : float64[:]
        Power spectrum array P(k).
    dim : int64
        Dimension of the transform
    mu : int64
        Order of the Bessel function in the Hankel transform.
    q_ : int64
        Bias parameter controlling the power-law tilt.
        Internally CCL compute epsilon = 0.5 * dim + q_ and uses epsilon as
        bias parameter.
    kcrc : float64
        Parameter controlling the mapping between k and r (usually set to 1.0).
    noring : int64
        If True, adjust kcrc to minimize ringing.

    Returns
    -------
    r : float64[:]
        Output distance array.
    xi : float64[:]
        Transformed correlation function array xi(r).
    """

    # Computed as epsilon in CCL
    q = 0.5 * dim + q_

    L = np.log(k[N - 1] / k[0]) * N / (N - 1.0)
    if noring:
        kcrc = goodkr(N, mu, q, L, kcrc)

    u = np.empty(N, dtype=np.complex128)
    compute_u_coeff(N, mu, q, L, kcrc, u)

    prefac_pk = np.power(k, dim / 2.0 - q)
    k0r0 = kcrc * np.exp(-L)
    r = np.zeros(N)
    r[0] = k0r0 / k[0]
    for n in range(1, N):
        r[n] = r[0] * np.exp(n * L / N)

    one_over_2pi_dhalf = pow(2 * np.pi, -dim / 2)
    prefac_xi = one_over_2pi_dhalf * np.power(r, -dim / 2 - q)

    a = np.asarray(prefac_pk * pk + 0.0 * 1j, dtype=np.complex128)
    # Use objmode for FFT calls - numba doesn't support np.fft with numpy 2.x
    with nb.objmode(b="complex128[:]"):
        b = np.fft.fft(a)
    b *= u
    with nb.objmode(b="complex128[:]"):
        b = np.fft.ifft(b)

    for n in range(0, int(N / 2)):
        tmp = b[n]
        b[n] = b[N - n - 1]
        b[N - n - 1] = tmp
    xi = prefac_xi * np.real(b)

    return r, xi
