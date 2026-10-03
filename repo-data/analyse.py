"""Figures and numbers from cache/adapt_*.npz (run.py). Fixed arm = M x on the same z.

    PYTHONPATH=src python3.12 analyse.py
"""
import glob
import os
import re

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run

HERE = run.HERE
ALPHA_U = 12.6
names = ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
iB = [2, 3]
theta = run.theta
M = run.operator()
sig1 = np.sqrt(np.einsum("jkl,lm,jkm->jk", M, run.cov, M))  # propagated sigma at alpha = 1
T0f = M @ run.xipm
T0a = run.T_adaptive(run.xipm)

def load(alpha):
    fs = sorted(glob.glob(f"{HERE}/cache/adapt_a{alpha:g}_c*.npz"))
    cs = [int(re.search(r"_c(\d+)", f).group(1)) for f in fs]
    Ta = np.concatenate([np.load(f)["T"] for f in fs])
    Tf = np.concatenate([np.einsum("jkl,nl->njk", M, run.xipm + alpha * run.noise(c)) for c in cs])
    return Ta, Tf

alphas = sorted({float(re.search(r"_a([\d.]+)_c", f).group(1)) for f in glob.glob(f"{HERE}/cache/*.npz")})
res = {a: load(a) for a in alphas}
lines = []
def out(s=""):
    print(s); lines.append(s)

out(f"alphas: {alphas}; N per alpha: {[len(res[a][0]) for a in alphas]}")
out(f"T_adapt(xi) - T_fixed(xi), noiseless, in sigma(alpha=1): " +
    ", ".join(f"{n}={v:.1e}" for n, v in zip(names, np.max(np.abs(T0a - T0f) / sig1, axis=1))))
out("Reference for both arms: the exact transform of the noiseless input, M xi (= fixed path to round-off).")
T0a = T0f

def stats(T, T0, alpha):
    s = alpha * sig1
    n = len(T)
    b = (T.mean(0) - T0) / s
    e = T.std(0, ddof=1) / s / np.sqrt(n)
    r = T.std(0, ddof=1) / s
    return b, e, r

def chi2(T, T0, alpha, sel):
    v = ((T - T0) / (alpha * sig1))[:, sel, :].reshape(len(T), -1)
    m = v.mean(0); n, p = v.shape
    C = np.cov(v, rowvar=False) / n
    emp = (n - p - 2) / (n - 1) * m @ np.linalg.solve(C, m)
    Ma = np.concatenate([M[j] for j in sel]) / np.repeat(sig1[sel].ravel(), 1)[:, None]
    Ca = Ma @ run.cov @ Ma.T / n  # analytic cov of the mean, units of sigma
    return emp, p, m @ np.linalg.solve(Ca, m)

# ---- numbers at alpha_U and alpha = 1
for a in [x for x in (ALPHA_U, 1.0) if x in res]:
    Ta, Tf = res[a]
    N = len(Ta)
    out(f"\n## alpha = {a}, N = {N}")
    for arm, T, T0 in (("adaptive", Ta, T0a), ("fixed", Tf, T0f)):
        b, e, r = stats(T, T0, a)
        out(f"**{arm}**  nonfinite={int((~np.isfinite(T)).sum())}")
        for j in range(6):
            k = np.argmax(np.abs(b[j]))
            out(f"  {names[j]:8s} max|bias| {b[j,k]:+.3f}±{e[j,k]:.3f} at {theta[k]:.2f}' (z={b[j,k]/e[j,k]:+.1f}); "
                f"bias at θmin {b[j,0]:+.3f}±{e[j,0]:.3f}; σemp/σana {r[j].min():.3f}–{r[j].max():.3f} "
                f"(at θmin {r[j,0]:.3f})")
        c, p, ca = chi2(T, T0, a, iB)
        from scipy import stats as _st
        out(f"  χ² of mean B vector vs M ξ: {c:.1f} (empirical cov, Hartlap), {ca:.1f} (M C Mᵀ, p={_st.chi2(p).sf(ca):.2g}) for {p} dof")
        z = np.abs((T - T0) / (a * sig1))[:, iB, :]
        out(f"  draws with any |B - T(ξ)| > 5σ: {np.mean(np.any(z > 5, axis=(1, 2))):.4f}, max {z.max():.1f}σ")
    D = (Ta - Tf) - (T0a - T0f)
    pb = D.mean(0) / (a * sig1); pe = D.std(0, ddof=1) / (a * sig1) / np.sqrt(N)
    v = (D / (a * sig1))[:, iB, :].reshape(N, -1); mv = v.mean(0)
    out(f"  χ² of paired mean B vector (empirical cov): {mv @ np.linalg.solve(np.cov(v, rowvar=False) / N, mv):.1f} / 40")
    for j in iB:
        k = np.argmax(np.abs(pb[j]))
        out(f"  paired {names[j]}: max {pb[j,k]:+.4f}±{pe[j,k]:.4f} at {theta[k]:.2f}'; per-bin rms(adapt-fixed)/σ max "
            f"{np.max(D[:, j].std(0) / (a * sig1[j])):.3f}")

# ---- Figure 1
OI = {"verm": "#D55E00", "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442"}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "axes.linewidth": 0.8})
a = ALPHA_U
Ta, Tf = res[a]
N = len(Ta)
fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), sharex=True, sharey="row")
dx = 1.035
for c, (arm, T, T0) in enumerate((("adaptive", Ta, T0a), ("fixed", Tf, T0f))):
    b, e, r = stats(T, T0, a)
    re_ = r / np.sqrt(2 * (N - 1))
    axes[0, c].axhspan(-2 / np.sqrt(N), 2 / np.sqrt(N), color="0.88", zorder=0, lw=0)
    axes[1, c].axhspan(1 - 2 / np.sqrt(2 * (N - 1)), 1 + 2 / np.sqrt(2 * (N - 1)), color="0.88", zorder=0, lw=0)
    for j, mk, col, fc, sh, lab in ((iB[0], "o", OI["verm"], OI["verm"], 1 / dx, r"$\xi_+^B$"),
                                    (iB[1], "D", OI["blue"], "none", dx, r"$\xi_-^B$")):
        kw = dict(marker=mk, color=col, mfc=fc, ms=5 if mk == "o" else 4.5, ls="none",
                  capsize=2, elinewidth=1, mew=1.1, label=lab)
        axes[0, c].errorbar(theta * sh, b[j], e[j], **kw)
        axes[1, c].errorbar(theta * sh, r[j], re_[j], **kw)
    axes[0, c].axhline(0, ls="--", color="0.3", lw=1)
    axes[1, c].axhline(1, ls="--", color="0.3", lw=1)
    axes[0, c].set_title({"adaptive": "adaptive (QUADPACK dqags)", "fixed": "fixed (Gauss–Legendre per cell)"}[arm],
                         fontsize=11)
    axes[1, c].set_xlabel(r"$\theta$ [arcmin]")
    axes[1, c].set_xscale("log")
axes[0, 0].set_ylabel(r"$[\langle \xi_\pm^B\rangle - M\bar\xi\,]/\sigma$")
axes[1, 0].set_ylabel(r"$\sigma^{\rm emp}/\sigma$")
axes[1, 1].legend(frameon=False, loc="upper right", ncol=2, handletextpad=0.2)
fig.text(0.99, 0.005, f"N = {N} draws ξ̄ + α L z, α = {a} × repo LSST covariance (≈ UNIONS shape noise); "
         "σ = propagated (M C Mᵀ)^½; grey: ±2σ Monte Carlo band\n"
         "setup of test_pure_eb_linearity: 20 nodes of the 1000-bin grid 0.5–250′, tmin/tmax = θ[0], θ[-1], "
         "pad_xim=True, interp_order=5", ha="right", va="bottom", fontsize=7, color="0.35")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{HERE}/fig1_mean_and_scatter.png", dpi=150)
plt.close(fig)

# ---- Figure 2: per-bin bias vs alpha, bins = worst at alpha_U by paired difference
A = np.array(alphas)
D_U = (Ta - Tf) - (T0a - T0f)
pbU = D_U.mean(0) / (a * sig1)
cands = [(j, k) for j in iB for k in range(len(theta))]
worst = [(iB[1], 0), (iB[0], 0)]
out(f"\nFigure 2 bins: {[(names[j], round(theta[k], 2)) for j, k in worst]}")
fig, ax2 = plt.subplots(figsize=(5.2, 3.9))
cols = [OI["verm"], OI["blue"], OI["green"], OI["purple"]]
out("| bin | arm | " + " | ".join(f"α={x:g}" for x in A) + " | log-log slope (paired, >3σ pts) |")
for (j, k), col in zip(worst, cols):
    lab = (r"$\xi_+^B$" if j == iB[0] else r"$\xi_-^B$") + f", {theta[k]:.1f}′"
    B_ = {arm: [] for arm in ("adaptive", "fixed", "paired")}
    for x in A:
        Tax, Tfx = res[x]
        s = x * sig1[j, k]; n = len(Tax)
        for arm, v in (("adaptive", Tax[:, j, k] - T0a[j, k]), ("fixed", Tfx[:, j, k] - T0f[j, k]),
                       ("paired", (Tax[:, j, k] - Tfx[:, j, k]) - (T0a[j, k] - T0f[j, k]))):
            B_[arm].append((v.mean() / s, v.std(ddof=1) / s / np.sqrt(n)))
    for arm in B_:
        B_[arm] = np.array(B_[arm])
    pb, pe = B_["paired"].T
    ok = np.abs(pb) > 3 * pe
    slope = np.polyfit(np.log(A[ok]), np.log(np.abs(pb[ok])), 1)[0] if ok.sum() >= 2 else np.nan
    for arm in B_:
        out(f"| {names[j]} {theta[k]:.2f}' | {arm} | " + " | ".join(f"{m:+.4f}±{s:.4f}" for m, s in B_[arm])
            + (f" | {slope:.2f}" if arm == "paired" else " | |"))
    ax2.errorbar(A, np.abs(pb), pe, marker="o", ms=4, color=col, capsize=2, label=lab)
aa = np.array([0.85, 30])
ref = np.abs(pbU[worst[0]]) / ALPHA_U
ax2.plot(aa, ref * aa, ls=":", color="0.4", lw=1)
ax2.plot(aa, ref * aa**2 / ALPHA_U, ls="--", color="0.4", lw=1)
ax2.text(1.0, ref * 1.0 * 1.3, r"$\propto \alpha$", fontsize=9, color="0.3", va="bottom")
ax2.text(1.6, ref * 1.6**2 / ALPHA_U * 0.7, r"$\propto \alpha^2$", fontsize=9, color="0.3", va="top")
ax2.set_yscale("log")
ax2.set_ylabel(r"quadrature bias, $|\langle T_{\rm dqags} - T_{\rm exact}\rangle|\,/\,\sigma$")
for a_ in (ax2,):
    a_.set_xscale("log")
    a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_xticks([1, 3, 6, 12.6, 25], ["1", "3", "6", "12.6", "25"])
    a_.set_xlabel(r"noise amplitude $\alpha$  ($\xi = \bar\xi + \alpha\,L z$)")
    for x, txt in ((1.0, "LSST\n(repo)"), (ALPHA_U, r"$\alpha_U$")):
        a_.axvline(x, color="0.55", lw=0.8, ls="-.", zorder=0)
    a_.text(1.03, 0.97, "LSST (repo)", transform=a_.get_xaxis_transform(), fontsize=7.5, color="0.4", va="top")
    a_.text(ALPHA_U * 1.03, 0.97, "UNIONS", transform=a_.get_xaxis_transform(), fontsize=7.5, color="0.4", va="top")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(f"{HERE}/fig2_bias_vs_alpha.png", dpi=150)
plt.close(fig)
with open(f"{HERE}/analyse_out.md", "w") as f:
    f.write("\n".join(lines) + "\n")
