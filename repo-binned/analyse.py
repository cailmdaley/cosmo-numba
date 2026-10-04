"""Figure + numbers from cache/adapt_*.npz (run.py). Fixed arm = M' x on the same z.

    PYTHONPATH=../issue23-repo/src python3.12 analyse.py
"""
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as st

import run

names = ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
iB = [2, 3]
th = run.theta
M = run.operator()
sig1 = np.sqrt(np.einsum("jkl,lm,jkm->jk", M, run.cov, M))
T0 = M @ run.xipm  # exact transform of the noiseless input
T0a = run.T_ad(run.xipm)
lines = []


def out(s=""):
    print(s); lines.append(s)


def load(a):
    fs = sorted(glob.glob(f"{run.HERE}/cache/adapt_a{a:g}_c*.npz"))
    cs = [int(re.search(r"_c(\d+)", f).group(1)) for f in fs]
    Ta = np.concatenate([np.load(f)["T"] for f in fs])
    Tf = np.concatenate([np.einsum("jkl,nl->njk", M, run.xipm + a * run.noise(c)) for c in cs])
    return Ta, Tf


def stats(T, a):
    s = a * sig1; n = len(T)
    return (T.mean(0) - T0) / s, T.std(0, ddof=1) / s / np.sqrt(n), s / T.std(0, ddof=1)


def chi2(T, a):
    v = ((T - T0) / (a * sig1))[:, iB, :].reshape(len(T), -1)
    m = v.mean(0); n = len(v)
    Ma = np.concatenate([M[j] / sig1[j][:, None] for j in iB])
    p = len(m)
    emp = (n - p - 2) / (n - 1) * m @ np.linalg.solve(np.cov(v, rowvar=False) / n, m)
    return m @ np.linalg.solve(Ma @ run.cov @ Ma.T / n, m), p, emp


alphas = sorted({float(re.search(r"_a([\d.]+)_c", f).group(1)) for f in glob.glob(f"{run.HERE}/cache/*.npz")})
res = {a: load(a) for a in alphas}
out(f"# issue23-binned: repo data, Paper II binned convention\n")
out(f"alphas {alphas}; N {[len(res[a][0]) for a in alphas]}")
out(f"reporting θ [′]: {', '.join(f'{t:.2f}' for t in th)}; tmin/tmax {run.TMIN:.4f}, {run.TMAX:.3f}")
out("noiseless |T_adaptive(ξ) − M′ξ|/σ(α=1) max per output: " +
    ", ".join(f"{n}={v:.1e}" for n, v in zip(names, np.max(np.abs(T0a - T0) / sig1, axis=1))))

a = 12.6
Ta, Tf = res[a]; N = len(Ta)
out(f"\n## α = {a}, N = {N} (common random numbers; reference T(mean) = M′ξ̄)")
summ = {}
for arm, T in (("adaptive", Ta), ("fixed", Tf)):
    b, e, r = stats(T, a)
    summ[arm] = (b, e, r)
    z = np.abs((T - T0) / (a * sig1))
    c, p, ce = chi2(T, a)
    out(f"\n**{arm}**: nonfinite {int((~np.isfinite(T)).sum())}; B χ² vs T(mean) = {c:.1f}/{p} (p={st.chi2(p).sf(c):.2g}) with M′CM′ᵀ; {ce:.1f} with the empirical covariance (Hartlap); "
        f"draws with any B output >8σ: {np.mean(np.any(z[:, iB] > 8, axis=(1, 2))):.4f} "
        f"(any output: {np.mean(np.any(z > 8, axis=(1, 2))):.4f}); max |B−T(mean)| {z[:, iB].max():.1f}σ")
    out("\n| θ [′] | " + " | ".join(f"{names[j]} bias/σ" for j in iB) + " | " +
        " | ".join(f"{names[j]} σ_ana/σ_emp" for j in iB) + " |")
    out("|---|---|---|---|---|")
    for k in range(len(th)):
        out(f"| {th[k]:.2f} | " + " | ".join(f"{b[j,k]:+.3f}±{e[j,k]:.3f}" for j in iB) + " | " +
            " | ".join(f"{r[j,k]:.3f}" for j in iB) + " |")
    for j in range(6):
        k = np.argmax(np.abs(b[j]))
        out(f"- {names[j]}: max|bias| {b[j,k]:+.3f}±{e[j,k]:.3f} at {th[k]:.2f}′; σ_ana/σ_emp {r[j].min():.3f}–{r[j].max():.3f}")

if len(alphas) > 1:
    out("\n## α scan (first 1000 draws of the same z; bias/σ of the adaptive arm minus the paired fixed arm)")
    out("| output, θ | " + " | ".join(f"α={x:g}" for x in alphas) + " |")
    out("|---|" + "---|" * len(alphas))
    for j in iB:
        for k in range(0, 8, 2):
            row = []
            for x in alphas:
                Tax, Tfx = res[x]; n = 1000
                d = (Tax[:n, j, k] - Tfx[:n, j, k]) / (x * sig1[j, k])
                row.append(f"{d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(n):.3f}")
            out(f"| {names[j]} {th[k]:.2f}′ | " + " | ".join(row) + " |")
    out("\n| α | adaptive B χ² (1000 draws, 40 dof; analytic / empirical cov) | fixed | adaptive outlier rate >8σ (B) |")
    out("|---|---|---|---|")
    for x in alphas:
        Tax, Tfx = res[x]
        z = np.abs((Tax[:1000] - T0) / (x * sig1))[:, iB]
        out(f"| {x:g} | {chi2(Tax[:1000], x)[0]:.1f} / {chi2(Tax[:1000], x)[2]:.1f} | {chi2(Tfx[:1000], x)[0]:.1f} | {np.mean(np.any(z > 8, axis=(1, 2))):.4f} |")

# ---- figure
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "axes.linewidth": 0.8})
col = {2: "#D55E00", 3: "#0072B2"}
lab = {2: r"$\xi_+^B$", 3: r"$\xi_-^B$"}
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.2, 5.8), sharex=True, height_ratios=(1.6, 1))
ax0.axhspan(-2 / np.sqrt(N), 2 / np.sqrt(N), color="0.9", lw=0, zorder=0)
ax1.axhspan(1 - 2 / np.sqrt(2 * (N - 1)), 1 + 2 / np.sqrt(2 * (N - 1)), color="0.9", lw=0, zorder=0)
for arm, mk, sh in (("adaptive", "o", 1.0), ("fixed", "s", 1.0)):
    b, e, r = summ[arm]
    for j, dx in ((2, 0.97), (3, 1.03)):
        kw = dict(marker=mk, ms=5, ls="-" if arm == "adaptive" else "none", lw=0.9, color=col[j],
                  mfc=col[j] if arm == "adaptive" else "white", mew=1.2, capsize=0, elinewidth=1)
        ax0.errorbar(th * dx, b[j], e[j], **kw)
        ax1.errorbar(th * dx, r[j], r[j] / np.sqrt(2 * (N - 1)), **kw)
ax0.axhline(0, color="0.4", lw=0.8)
ax1.axhline(1, color="0.4", lw=0.8)
ax0.set_ylabel(r"$(\langle T\rangle - T(\bar\xi))\,/\,\sigma_{\rm ana}$")
ax1.set_ylabel(r"$\sigma_{\rm ana}/\sigma_{\rm emp}$")
ax1.set_xlabel(r"$\theta$ [arcmin]")
ax1.set_xscale("log")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=col[2], marker="o", ls="-", label=lab[2]),
     Line2D([], [], color=col[3], marker="o", ls="-", label=lab[3]),
     Line2D([], [], color="0.3", marker="o", ls="-", label="adaptive (dqags)"),
     Line2D([], [], color="0.3", marker="s", mfc="white", ls="none", label="exact (fixed)")]
ax0.legend(handles=h, frameon=False, ncol=2, loc="lower right", fontsize=9.5)
ax0.set_title(f"cosmo-numba test data, noise ×{a:g} (UNIONS-like), 20 binned bins, N = {N}",
              fontsize=9.5, color="0.3")
ax1.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{run.HERE}/fig_binned_bias.{ext}", dpi=160)
with open(f"{run.HERE}/results.md", "w") as f:
    f.write("\n".join(lines) + "\n")
