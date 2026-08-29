# Route B: fixed-quadrature pure-E/B operator — results

`get_pure_EB_operator` (`src/cosmo_numba/B_modes/schneider2022_operator.py`)
returns dense matrices `M` such that

```
out = M @ np.concatenate([xi_p(theta), xi_m(theta)])
```

reproduces the corresponding output of

```
get_pure_EB_modes(theta, xip, xim, theta, xip, xim, tmin, tmax,
                  pad_xim=True, pad_theta_max_decade=1.0, interp_order=5)
```

on the rows `theta_eval`. Validation: `validation/validate_operator.py`.
Tests: `tests/cosmo_numba/test_schneider2022_operator.py` (8 tests, ~50 s,
dominated by numba JIT).

## Construction

`interp_quad` builds a degree-`k` local-Taylor interpolant `P[g]` through the
**integrand** samples `g` on the log-uniform grid, then hands `P[g]` to
QUADPACK's adaptive `dqags`. `P[g]` is linear in `g`, and `g_j = c_j(t) *
xi_j` with `c_j(t)` a pure kernel factor, so the intended transform is linear.
The operator makes it linear by construction:

- `P[g]` is a degree-`k` polynomial **on each cell** of the grid, so after the
  substitution `theta = exp(x)` each integral is `∫ poly(x) e^x dx` cell by
  cell — exact under Gauss–Legendre for any `n_gauss > (k+1)/2`. Default
  `n_gauss = 16`; whole cells use a closed form (`A_i(h)`, six numbers), only
  the two boundary cells need an on-the-fly rule.
- The cardinal (stencil) functions `asx_i(ratx)` and the ghost-cell
  extrapolation coefficients are **extracted numerically from
  `interpolate_1D`** (`nb_interp1d` on unit vectors, `_extrapolate1d` on unit
  vectors), not re-derived. Nothing about the interpolation scheme is retyped,
  so the operator tracks the reference by construction and generalises to any
  `interp_order` the reference supports.
- Ghost weights are folded back onto real nodes by the exact adjoint of
  `_extrapolate1d`.
- The four integrals over `[tmin, tmax]` (`S_±`, `V_±`) share one weight
  vector across all evaluation points; only the two `t`-dependent supports
  (`(t, tmax]` for `xi_p`, `[tmin, t)` for `xi_m`) need per-row weights.

**Stacking:** input `[xi_p(theta) ; xi_m(theta)]`, length `2 n`; column `j` is
`xi_p(theta[j])`, column `n + j` is `xi_m(theta[j])`. Every matrix has shape
`(len(theta_eval), 2 n)`. `theta_eval` must be a subset of `theta` inside
`[tmin, tmax]`. Outputs available: `xip_E`, `xip_B`, `xim_E`, `xim_B`,
`xip_amb`, `xim_amb` (first four by default). Returns
`{"matrices": {...}, "valid": {...}, "theta_eval", "theta", "eval_index",
"tmin", "tmax", "stacking"}`.

## Achieved agreement

The brief's target was "max abs relative deviation < 1e-6 vs
`get_pure_EB_modes` with tight dqags tolerances". **That target is not the
right one, and I did not use it** — see *Deviations* below. The reference's
adaptive quadrature is itself wrong by far more than 1e-6 on the grids the
pipeline uses, so the operator is validated against the **exact integral of
the reference interpolant**, computed by brute force (composite 40-node
Gauss–Legendre on `nb_interp1d` — no shared code with the operator).

**Operator vs. exact integral of the reference interpolant** (smooth broken
power-law `xi_pm`, `theta` log-spaced on `[0.5', 300']`, `tmin/tmax` = grid
ends, interior rows only):

| output | max &#124;op−exact&#124;/&#124;exact&#124;, n=200 | n=1000 | max &#124;op−exact&#124;/max&#124;xi&#124;, n=1000 |
|---|---|---|---|
| `xip_E`   | 7.6e-14 | 1.0e-12 | 3.1e-15 |
| `xip_B`   | 1.1e-13 | 3.7e-13 | 2.7e-15 |
| `xim_E`   | 2.5e-13 | 1.7e-12 | 1.3e-15 |
| `xim_B`   | 2.1e-12 | 3.1e-10 | 5.5e-16 |
| `xip_amb` | 3.9e-14 | 1.0e-13 | 3.0e-17 |
| `xim_amb` | 7.3e-16 | 1.6e-15 | 1.3e-15 |

Worst interior relative deviation **2.1e-12 (n=200)** / **3.1e-10 (n=1000)**;
relative to `max|xi|` it is **≤ 3e-15** everywhere, i.e. round-off. The one
larger relative number (`xim_B`, 3.1e-10) is a near-zero crossing.

Supporting checks:

- Weights vs. brute force, over grids of 40/150/600 points and five support
  geometries (grid endpoints, sub-support, extrapolate left, extrapolate
  right, interior partial cells): worst **1.3e-13** relative.
- `n_gauss ∈ {6, 8, 12, 16, 24, 32}` give the same integral to **8e-15** —
  the rule is converged, not tuned.
- `xi_p = E + B + amb` and `xi_m = E − B + amb` hold to **exactly 0.0**
  (structural, since the identity is built into the matrix rows).
- `M(a y1 + b y2) − a M y1 − b M y2` = **1e-14** relative — linearity is
  exact, which was the point of the exercise.

## The reference is the inaccurate one

Same comparison, `get_pure_EB_modes` vs. the same exact integral,
`max|ref−exact| / max|xi|`:

| output | n=200, eps=1e-10 | n=200, eps=1e-12 | n=1000, eps=1e-10 | n=1000, eps=1e-12 |
|---|---|---|---|---|
| `xip_E`   | 1.3e-08 | 1.8e-08 | 4.7e-08 | 5.7e-12 |
| `xim_E`   | 3.9e-09 | 1.2e-09 | 6.0e-05 | **5.3e-05** |
| `xim_B`   | 3.8e-09 | 1.2e-09 | 6.0e-05 | **5.3e-05** |
| `xim_amb` | 7.8e-09 | 2.4e-09 | 1.2e-04 | **1.1e-04** |

Tightening `epsabs`/`epsrel` from 1e-10 to 1e-12 does **not** fix it. On the
n=1000 grid the single worst evaluation point is `theta = 29.35'`, where

```
V_m   dqags(1e-12) = 2.685897e-07
      exact         = 2.323746e-07     ->  dqags is 15.6% wrong
      operator      = 2.323746e-07     ->  matches exact to 7e-16
```

**The failures are sporadic, not systematic.** Scanning `n_grid` from 995 to
1011 (worst `|dqags − op|` over all evaluation points, in units of
`max|xi_m|`):

```
995..999   1.3e-10 .. 1.7e-10
1000       1.06e-04   <== catastrophic, at theta = 29.35'
1001..1011 5.8e-11 .. 2.1e-10
```

One grid in seventeen carries an O(10%) error at one evaluation point;
the rest are fine at the 1e-10 level. That is the signature of an adaptive
integrator landing on a bad subdivision — the integrand it is fed is a
degree-5 interpolant with discontinuous derivatives at every cell boundary,
plus (for the `xi_m` path) a hard jump at `tmax` where `pad_xim` zero-pads,
which the interpolant turns into narrow ringing. Refining the grid makes the
ringing *narrower*, so the adaptive panels resolve it *worse*: on a smooth
input, dqags at n=200 is good to 1e-9 relative and at n=1000 it is not.

The fixed rule integrates that same interpolant exactly, and its value
converges smoothly under grid refinement (`xi_m^amb` at the common point
`theta = 12.247'`: 1.333351e-06, 1.333350e-06, 1.333349e-06, 1.333349e-06,
1.333348e-06 for n = 251, 501, 1001, 2001, 4001).

This matters beyond linearity: on the fiducial UNIONS grid the reference can
put an O(10%) error into an individual `theta` bin of `xi_m^E/B`, with no
warning and no sensitivity to the tolerance knob.

## Edge policy

Two of the six integrals run over a `t`-dependent sub-range: `(t, tmax]` for
the `xi_p` transform and `[tmin, t)` for the `xi_m` one. Near the
corresponding end of the range that support holds fewer than `k + 1 = 6`
nodes, and `_extrapolate1d` then fills its ghost cells from *uninitialised
memory* — the reference returns garbage there (the known upstream defect; not
fixed here).

**Policy: NaN, flagged.** Those quadrature contributions are set to `NaN`, so
`M @ xi` is `NaN` on the affected rows, and `result["valid"][name]` is `False`
there. Exactly 5 rows at each end:

- `xip_E`, `xip_B`: the 5 rows below `tmax` (support 1–5 nodes). The row *at*
  `tmax` has an empty support, where the reference sets the integral to zero —
  reproduced exactly, so that row is valid.
- `xim_E`, `xim_B`: the 5 rows above `tmin`, symmetrically.
- `xip_amb`, `xim_amb` integrate over the full range and are never affected.

I chose NaN over a reduced-order rule so that a silently-different transform
cannot leak into a covariance; callers slice with `valid`. A reduced-order
fallback would need `asx` coefficients for `k < 5` and would produce numbers
that match neither the reference nor the `k=5` transform.

## Timings and memory

Node n08, 8 cores, `theta_eval == theta`, 4 E/B matrices (`float64`):

| n_grid | build [s] | apply [ms], 4 outputs | memory [GB] |
|---|---|---|---|
| 1000  |  0.5 |   1.6 | 0.06 |
| 2000  |  1.2 |   6.8 | 0.26 |
| 5000  |  6.2 |  43.2 | 1.60 |
| 10000 | 19.6 | 176.1 | 6.40 |

Build is well under the "minutes" budget; application is milliseconds and
memory-bandwidth bound. Memory is `8 · n_eval · 2n` per matrix — 6.4 GB for
four matrices at n=10000, so pass `outputs=` to build only what is needed
(e.g. `("xim_E", "xim_B")` halves it). Restricting `theta_eval` to the data
bins rather than the full integration grid reduces it by the same factor.

For reference, `get_pure_EB_modes(parallel=True)` takes 2.0 s at n=1000 —
comparable to a single operator build, so the operator pays for itself the
first time it is reused (covariance propagation, many noise realisations).

## Deviations from the brief

1. **Validation target changed.** The brief asked for `< 1e-6` max relative
   deviation against `get_pure_EB_modes` at tight dqags tolerances. On the
   n=1000 grid the reference deviates from the exact integral of its own
   interpolant by up to 15.6% at isolated points *at* `eps = 1e-12`, so that
   target is unmeetable and would be the wrong thing to meet. I validated
   against the exact integral instead, computed by an independent brute-force
   integration, and report the dqags comparison as a diagnostic. The pytest
   comparison against `get_pure_EB_modes` is kept but with a deliberately
   loose tolerance (`1e-5 · max|xi|` at n=80, where dqags is still decent);
   the tight assertion lives in `test_quadrature_weights_are_exact`.

2. **Single grid.** `get_pure_EB_modes` accepts distinct `theta` and
   `theta_int`. The operator takes one grid (`theta_eval ⊆ theta`), matching
   the pipeline's usage and the brief's suggested signature. Supporting
   `theta ≠ theta_int` would require interpolating the direct `xip[i]`,
   `xim[i]` terms, adding a second interpolation operator for no pipeline
   benefit.

3. **Interpolation coefficients are extracted, not transcribed.** The brief
   suggested integrating B-spline basis functions or Lagrange cardinal
   functions analytically. Both would re-derive the scheme and risk drifting
   from `interpolate_1D`. Sampling the actual interpolator on unit vectors and
   fitting the (exactly degree-`k`) cardinal polynomials is simpler and
   guarantees fidelity; `test_quadrature_weights_are_exact` pins it.

## Follow-ups worth having

- The `pad_xim` zero-padding puts a step discontinuity in the `xi_m`
  integrand at `tmax`. The operator integrates the resulting ringing exactly,
  which is the right thing to do relative to the defined estimator, but the
  ringing is an artefact of padding a discontinuity with a degree-5
  interpolant. Tapering the pad, or padding with a power-law continuation,
  would be worth testing against the operator.
- The `< k+1` node defect in `_extrapolate1d` is real and silent (reads
  uninitialised memory). It belongs upstream; here it is only avoided.
