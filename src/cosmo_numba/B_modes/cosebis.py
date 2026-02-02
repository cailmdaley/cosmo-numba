"""COSEBIs

Compute COSEBIs based on Schneider et al. 2010
(https://arxiv.org/abs/1002.2136).
This code include the computation of the roots and normalizations requiered to
compute T_plus_log. At the moment the code can only compute the "log" quantity.
The code has been tested against Mathematica for the roots/norm and against
nicaea for the COSEBIs E-/B-modes.

Author: Axel Guinot

"""

import numpy as np
import sympy as sym
import mpmath as mp
import numba as nb

from .cosebis_nb import (
    tp_n_log,
    tm_n_log,
    wn_log,
    get_xipm_cosebis_serial,
    get_xipm_cosebis_parallel,
    get_Cell_cosebis_serial,
    get_Cell_cosebis_parallel,
    get_cosebis_cov_from_xipm_cov,
)


class COSEBIS:
    """COSEBIs

    This class handles the computation of the COSEBIs E-/B-modes from both 2pt
    correlation functions and power spectra.
    The implementation is based on Schneider et al. 2010
    (https://arxiv.org/abs/1002.2136).
    The code computes the roots and normalizations required to compute the
    T_plus_log and T_minus_log adapted from the original mathematica code.

    Parameters
    ----------
    theta_min : float
        Minimum theta in arcmin.
    theta_max : float
        Maximum theta in arcmin.
    N_max : int
        Maximum mode for the COSEBIs.
    precision : int, optional
        Precision used in the sympy/mpmath calculations, by default 80.
    """

    def __init__(self, theta_min, theta_max, N_max, precision=80):
        if not isinstance(precision, int):
            raise TypeError("precision should be an integer.")
        self.precision = precision

        self.tmin = theta_min
        self.tmax = theta_max

        if not isinstance(N_max, int):
            raise TypeError("N_max should be an integer.")
        self.Nmax = N_max

    def compute_roots(self):
        """
        Initialize the COSEBIs by computing the roots and normalization.
        """
        with mp.workdps(self.precision):
            self._compute_roots()

    def _compute_roots(self):
        all_roots, all_norms = compute_roots_norms(
            self.Nmax,
            self.tmin,
            self.tmax,
            self.precision,
        )
        self._raw_roots = all_roots
        self._raw_norms = all_norms

        self.roots = nb.typed.List(
            [
                np.array([np.float64(roots_tmp) for roots_tmp in roots_n_tmp])
                for roots_n_tmp in all_roots
            ]
        )
        self.norms = np.array([np.float64(norm) for norm in all_norms])

    def get_Tp_log(self, theta, roots=None, norms=None):
        """
        Compute T_plus_log for an array of theta.

        Parameters
        ----------
        theta : array_like
            Theta values in arcmin where to compute T_plus_log.
        roots : list of array_like, optional
            List of arrays containing the roots for each mode. If None, the
            roots stored in the class are used, by default None.
        norms : list of float, optional
            List of normalizations for each mode. If None, the norms stored in
            the class are used, by default None.

        Returns
        -------
        Tp_log : array_like
            2D array of shape (Nmodes, len(theta)) of T_minus_log evaluated at
            the provided theta values.
        """
        if (roots is None) and (norms is None):
            if not hasattr(self, "roots"):
                self.compute_roots()
            roots = self.roots
            norms = self.norms
        elif (roots is None) ^ (norms is None):
            raise ValueError(
                "Both roots and norms should be provided or neither."
            )

        z = np.log(theta / self.tmin)
        Tp_log = tp_n_log(z, roots, norms)

        return Tp_log

    def get_Tm_log(self, theta, roots=None, norms=None):
        """
        Compute T_minus_log for an array of theta.

        Parameters
        ----------
        theta : array_like
            Theta values in arcmin where to compute T_minus_log.
        roots : list of array_like, optional
            List of arrays containing the roots for each mode. If None, the
            roots stored in the class are used, by default None.
        norms : list of float, optional
            List of normalizations for each mode. If None, the norms stored in
            the class are used, by default None.

        Returns
        -------
        Tm_log : array_like
            2D array of shape (Nmodes, len(theta)) of T_minus_log evaluated at
            the provided theta values.
        """
        if (roots is None) and (norms is None):
            if not hasattr(self, "roots"):
                self.compute_roots()
            roots = self.roots
            norms = self.norms
        elif (roots is None) ^ (norms is None):
            raise ValueError(
                "Both roots and norms should be provided or neither."
            )

        z = np.log(theta / self.tmin)
        Tm_log = tm_n_log(z, roots, norms)

        return Tm_log

    def get_Wn_log(
        self,
        ell,
        theta=None,
        q=-1.3,
        pad_low_decades=2.0,
        pad_high_decades=2.0,
    ):
        """
        Compute Wn_log using the FFTLog algorithm.
        For the computation to be accurate we need a finely binned theta grid.
        If not provided, a grid of 100,000 points between tmin and tmax is
        used. In addition, we need to pad the T_plus_log function to avoid edge
        effects in the FFTLog. Finally, the parameter `q` is the biasing
        parameter used in the FFTLog. After some testing, a value of -1.3 seems
        to provide good results, but with enough theta bins and padding, the
        results should be robust to the choice of `q`.

        Parameters
        ----------
        ell : array_like
            Ell values where to compute Wn_log.
        theta : array_like, optional
            Theta values in arcmin where to compute T_plus_log used in FFTLog.
            Has to be log-spaced and contain enough points O(100k). If None, a
            grid of 100,000 points between tmin and tmax is used, by default
            None.
        q : float, optional
            Biasing parameter used in the FFTLog, by default -1.3.
        pad_low_decades : float, optional
            Number of decades to pad below the minimum theta, by default 2.0.
        pad_high_decades : float, optional
            Number of decades to pad above the maximum theta, by default 2.0.

        Returns
        -------
        Wn_log : array_like
            2D array of shape (Nmodes, len(ell)) of Wn_log evaluated at the
            provided ell values.
        """
        if theta is None:
            theta_ = np.logspace(
                np.log10(self.tmin), np.log10(self.tmax), 100_000 + 1
            )
            theta = np.mean([theta_[:-1], theta_[1:]], axis=0)

        Tp_log = self.get_Tp_log(theta)

        theta_rad = np.deg2rad(theta / 60)

        Wn_log = wn_log(
            ell,
            theta_rad,
            Tp_log,
            q,
            pad_low_decades,
            pad_high_decades,
        )

        return Wn_log

    def cosebis_from_xipm(
        self,
        theta,
        xi_plus,
        xi_minus,
        cache=False,
        parallel=False,
    ):
        """
        Compute the COSEBIs E-/B-modes from real space 2pt correlation
        functions.

        Parameters
        ----------
        theta : array_like
            Theta values in arcmin where xi_plus and xi_minus are provided.
        xi_plus : array_like
            xi_plus values at the provided theta values.
        xi_minus : array_like
            xi_minus values at the provided theta values.
        cache : bool, optional
            If True, cache the T_plus_log and T_minus_log used in the
            computation for future use. This is particularly relevant for
            inference, where the theta range is fixed so the roots, norms and
            T_pm_log can be reused. By default False.
        parallel : bool, optional
            If True, use the parallel version of the computation. By default
            False.

        Returns
        -------
        C_E, C_B : array_like
            COSEBIs E- and B-modes computed from the provided xi_plus and
            xi_minus.
        """
        if (np.min(theta) < self.tmin) | (np.max(theta) > self.tmax):
            raise ValueError(
                "The range of theta for xipm is outside the range used to"
                " compute the roots and normalizations "
                f"[{self.tmin, self.tmax}]."
            )

        if cache & hasattr(self, "_Tp_log"):
            Tp_log = self._Tp_log
            Tm_log = self._Tm_log
        else:
            Tp_log = self.get_Tp_log(theta)
            Tm_log = self.get_Tm_log(theta)
            if cache:
                self._Tp_log = Tp_log
                self._Tm_log = Tm_log

        theta_rad = np.deg2rad(theta / 60)
        if not parallel:
            C_E, C_B = get_xipm_cosebis_serial(
                theta_rad,
                xi_plus,
                xi_minus,
                Tp_log,
                Tm_log,
                self.Nmax,
            )
        else:
            C_E, C_B = get_xipm_cosebis_parallel(
                theta_rad,
                xi_plus,
                xi_minus,
                Tp_log,
                Tm_log,
                self.Nmax,
            )

        return C_E, C_B

    def cosebis_from_Cell(
        self,
        ell,
        Cell_E,
        Cell_B,
        theta=None,
        q=-1.3,
        pad_low_decades=2.0,
        pad_high_decades=2.0,
        cache=False,
        parallel=False,
    ):
        """
        Compute the COSEBIs E-/B-modes from power spectra.

        Parameters
        ----------
        ell : array_like
            Ell values where Cell_E and Cell_B are provided.
        Cell_E : array_like
            E-mode power spectrum values at the provided ell values.
        Cell_B : array_like
            B-mode power spectrum values at the provided ell values.
        theta : array_like, optional
            Theta values in arcmin where to compute T_plus_log used in FFTLog.
            Has to be log-spaced and contain enough points O(100k). If None, a
            grid of 100,000 points between tmin and tmax is used, by default
            None.
        q : float, optional
            Biasing parameter used in the FFTLog, by default -1.3.
        pad_low_decades : float, optional
            Number of decades to pad below the minimum theta, by default 2.0.
        pad_high_decades : float, optional
            Number of decades to pad above the maximum theta, by default 2.0.
        cache : bool, optional
            If True, cache the Wn_log used in the computation for future use.
            This is particularly relevant for inference, where the theta range
            is fixed so the roots, norms and Wn_log can be reused. By default
            False.
        parallel : bool, optional
            If True, use the parallel version of the computation. By default
            False.

        Returns
        -------
        C_E, C_B : array_like
            COSEBIs E- and B-modes computed from the provided Cell_E and
            Cell_B.
        """
        if cache & hasattr(self, "_Wn_log"):
            Wn_log = self._Wn_log
        else:
            Wn_log = self.get_Wn_log(
                ell,
                theta,
                q,
                pad_low_decades,
                pad_high_decades,
            )
            if cache:
                self._Wn_log = Wn_log

        if not parallel:
            C_E, C_B = get_Cell_cosebis_serial(
                ell, Cell_E, Cell_B, Wn_log, self.Nmax
            )
        else:
            C_E, C_B = get_Cell_cosebis_parallel(
                ell, Cell_E, Cell_B, Wn_log, self.Nmax
            )

        return C_E, C_B

    def cosebis_covariance_from_xipm_covariance(
        self,
        theta,
        cov_xi_pm,
    ):
        """
        Compute the COSEBIs covariance from the real space covariance of the
        2pt correlation functions.

        Parameters
        ----------
        theta : array_like
            Theta values in arcmin where xi_plus and xi_minus are provided.
        cov_xi_pm : array_like
            Covariance matrix of xi_plus and xi_minus at the provided theta
            values. The shape has to be (2*len(theta), 2*len(theta)) where the
            first len(theta) indices correspond to xi_plus and the last
            len(theta) indices to xi_minus.

        Returns
        -------
        Cov_EB : array_like
            Covariance matrix of the COSEBIs E- and B-modes. The shape is
            (2*Nmax, 2*Nmax) where the first Nmax indices correspond to E-modes
            and the last Nmax indices to B-modes.
        """
        if (np.min(theta) < self.tmin) | (np.max(theta) > self.tmax):
            raise ValueError(
                "WARNING: The range of theta for xipm is outside the range "
                "used to compute the roots and normalizations "
                f"[{self.tmin, self.tmax}]."
            )

        Tp_log = self.get_Tp_log(theta)
        Tm_log = self.get_Tm_log(theta)

        theta_rad = np.deg2rad(theta / 60)
        Cov_EB = get_cosebis_cov_from_xipm_cov(
            theta_rad, cov_xi_pm, Tp_log, Tm_log, self.Nmax
        )

        return Cov_EB


def compute_roots_norms(Nmax, tmin, tmax, precision=80):
    """compute roots and norms

    Compute the roots, normalization and Tp_log used in the COSEBIs
    computation.
    This code is derived from the mathematica version proposed in Fig. 3 of
    Schneider et al. 2010 (https://arxiv.org/abs/1002.2136).
    The equations referenced in the code come from the paper above.

    Parameters
    ----------
    Nmax : int
        Maximum mode for the COSEBIs.
    tmin : float
        Minimum theta in arcmin.
    tmax : float
        Maximum theta in arcmin.
    precision : int, optional
        Precision used in the sympy calculations, by default 80.

    Returns
    -------
    all_roots : list of sympy.Array
        List of arrays containing the roots for each mode.
    all_norm : list of sympy.Float
        List of normalizations for each mode.
    """

    z = sym.Symbol("z")
    i = sym.Symbol("i")

    # Eq. 29 for theta = theta_max
    zm = sym.N(
        sym.log(
            sym.nsimplify(
                tmax / tmin,
                rational=True,
                tolerance=1e-20,
            )
        ),
        precision,
    )

    # Eq 32, k=[1,2]
    J = sym.Array(
        [
            [
                sym.re(
                    sym.lowergamma(j + 1, -(k + 1) * zm)
                    / ((-(k + 1)) ** (j + 1)),
                )
                for j in range(0, 2 * Nmax + 1 + 1)
            ]
            for k in range(0, 2)
        ]
    ).as_mutable()

    # Eq 32, k=4
    J4 = sym.Array(
        [
            sym.re(
                sym.lowergamma(j + 1, -4 * zm) / ((-4) ** (j + 1)),
            )
            for j in range(0, 2 * Nmax + 1 + 1)
        ]
    ).as_mutable()

    # Those arrays are not accessed during the first iteration, we
    # initialized them just to have a nice code.
    c = sym.Array([0]).as_mutable()
    c_old = sym.Array([0]).as_mutable()

    all_roots = []
    all_norm = []
    # tp_log = []
    for n in range(1, Nmax + 1):
        a = sym.Array([[0] * (n + 1)] * (n + 1)).as_mutable()
        for j in range(0, n + 1):
            a[n - 1, j] = J[2 - 1, j] / J[2 - 1, n + 1]
            a[n + 1 - 1, j] = J4[j] / J4[n + 1]
        b = sym.Array([0] * (n + 1)).as_mutable()
        b[n - 1,] = -1
        b[n + 1 - 1,] = -1

        for m in range(1, n):
            for j in range(0, n + 1):
                tmp_sum = 0
                for ii in range(0, m + 1 + 1):
                    tmp_sum += J[0, ii + j] * c[m - 1, ii]
                a[m - 1, j] = tmp_sum

        bb = sym.Array([0] * (n)).as_mutable()
        for m in range(1, n):
            tmp_sum = 0
            for ii in range(0, m + 1 + 1):
                tmp_sum += J[0, ii + n + 1] * c[m - 1, ii]
            bb[m - 1,] = -tmp_sum

        for m in range(1, n):
            for j in range(0, n + 1):
                a[m - 1, j] = a[m - 1, j] / bb[m - 1]
        for m in range(1, n):
            b[m - 1,] = sym.Float(1)

        A = sym.Matrix(
            [
                [a[i - 1, j] for j in range(0, n + 1)]
                for i in range(1, n + 1 + 1)
            ]
        )
        B = sym.Matrix([b[i - 1] for i in range(1, n + 1 + 1)])

        CC = sym.linsolve((A, B))

        c = sym.Array([[0] * (n + 1 + 1)] * (n)).as_mutable()
        for ii in range(n - 1):
            for j in range(n + 1):
                c[ii, j] = c_old[ii, j]

        for j in range(0, n + 1):
            c[n - 1, j] = CC.args[0][j]
        c[n - 1, n + 1] = 1
        c_old = c

        # Eq. 28
        tt = sym.summation(c[n - 1, i] * z ** (i), (i, 0, n + 1))

        # Increase maxsteps from default 50 to 200 to improve convergence
        # for high-order polynomials (modes 10+)
        roots = sym.nroots(tt, n=precision, maxsteps=200)

        r = sym.Array([sym.N(sym.re(roots[j]), 50) for j in range(0, n + 1)])
        # Eq. 36 (without normalization)
        t = sym.product(z - r[i,], (i, 0, n))

        # Eq. 31
        mp_integrand = sym.lambdify(z, sym.exp(z) * t**2, modules="mpmath")
        mp_res = mp.quad(mp_integrand, [0, zm])
        normgral = sym.sympify(mp_res)

        norm = sym.sqrt((sym.exp(zm) - 1) / normgral)

        all_roots.append(r)
        all_norm.append(sym.N(norm, 50))
        # tp_log.append(t * sym.N(norm, 50))

    return all_roots, all_norm
