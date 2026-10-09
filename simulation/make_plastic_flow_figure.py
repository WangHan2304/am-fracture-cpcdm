"""Taylor iso-strain vs DAMASK spectral-FFT full-field: PLASTIC-flow comparison figure.

Two panels (single-column, paper-ready):
  (a) true (Cauchy) stress vs logarithmic strain, damage-free, 3 alloys;
      DAMASK FFT (solid) vs Taylor CPCDM (dashed). Shows that the iso-strain
      surrogate misrepresents the FLOW-STRESS level through the plastic branch,
      not only the UTS / fracture strain.
  (b) relative flow-stress deviation (Taylor - FFT)/FFT vs logarithmic strain,
      quantifying how the homogenization bias evolves through plastic flow.

Robustness: the raw Taylor stress--strain is jagged (grain-activation noise), so
pointwise differentiation is not used; the hardening-rate bias quoted in the text
is computed from secant slopes sigma_y -> UTS, which are stable.

Reads local CSVs (Taylor: simulation/output/taylor_nodamage_*; the DAMASK FFT
macro curves are staged to simulation/output/damask_*_macro.csv).
Outputs simulation/figures/Fig_plastic_flow_taylor_vs_fft.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(HERE, "output")
fig_dir = os.path.join(HERE, "figures")


def load_curve(path):
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]


def rolling_median(x, w):
    """Centred rolling median (robust to isolated spikes in the raw Taylor curve)."""
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def smooth_cascade(x, w_med=15, w_avg=9):
    """Rolling median then moving average: removes grain-activation noise while
    preserving the smooth hardening trend of the damage-free flow curve."""
    xm = rolling_median(x, w_med)
    k = np.ones(w_avg) / w_avg
    return np.convolve(np.pad(xm, w_avg // 2, mode="edge"), k, mode="valid")


mats = [("316L", "316L SS", "#1f77b4"),
        ("Ti64", "Ti-6Al-4V", "#d62728"),
        ("AlSi10Mg", "AlSi10Mg", "#2ca02c")]

plt.rcParams.update({"font.size": 8.5})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1))

for mat, label, color in mats:
    te, ts_pa = load_curve(os.path.join(out_dir, f"taylor_nodamage_{mat}_0deg.csv"))
    teln = np.log(1.0 + te)
    ts = ts_pa / 1e6
    ts = smooth_cascade(ts)                 # denoise the jagged Taylor flow curve
    de, ds = load_curve(os.path.join(out_dir, f"damask_{mat}_macro.csv"))

    # (a) flow curves
    axA.plot(de, ds, "-", color=color, lw=1.8)
    axA.plot(teln, ts, "--", color=color, lw=1.4, alpha=0.85)

    # common strain grid over the overlap, started past the yield transient so
    # that the near-yield onset mismatch does not dominate the relative bias
    lo = 0.02
    hi = min(teln.max(), de.max())
    grid = np.linspace(lo, hi, 80)
    ty = np.interp(grid, teln, ts)
    dm = np.interp(grid, de, ds)

    # (b) relative flow-stress deviation (lightly smoothed to drop residual spikes)
    rel = (ty - dm) / np.maximum(dm, 1e-9) * 100.0
    rel_s = rolling_median(rel, 7)
    axB.plot(grid, rel_s, "-", color=color, lw=1.6)

    # direct end-of-curve labels (replace a boxed legend): alloy + mean bias
    mean_dev = np.mean(rel)
    if mat == "316L":
        axB.annotate(f"316L SS ({mean_dev:+.0f}%)", (grid[-1], rel_s[-1]),
                     xytext=(-2, 6), textcoords="offset points",
                     ha="right", va="bottom", fontsize=7, color=color, fontweight="bold")
    elif mat == "AlSi10Mg":
        axB.annotate(f"AlSi10Mg ({mean_dev:+.0f}%)", (grid[-1], rel_s[-1]),
                     xytext=(4, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=7, color=color, fontweight="bold")
    elif mat == "Ti64":
        axB.annotate(f"Ti-6Al-4V ({mean_dev:+.0f}%)", (grid[-1], rel_s[-1]),
                     xytext=(4, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=7, color=color, fontweight="bold")

    # robust hardening-rate bias: secant slope of the flow curve over the common
    # plastic window [0.02, hi] (peak-location noise avoided by not using argmax)
    th_t = (ty[-1] - ty[0]) / max(grid[-1] - grid[0], 1e-9)
    th_d = (dm[-1] - dm[0]) / max(grid[-1] - grid[0], 1e-9)
    print(f"{label:11s} UTS {ts.max():6.1f}/{ds.max():6.1f} MPa | secant hardening rate "
          f"Taylor {th_t:7.0f} vs FFT {th_d:7.0f} MPa (bias {100*(th_t/th_d-1):+6.1f}%) | "
          f"mean flow-stress dev {np.mean(rel):+5.1f}%")

axA.set_xlabel(r"Logarithmic strain $\varepsilon_{11}$")
axA.set_ylabel(r"Cauchy stress $\sigma_{11}$ [MPa]")
axA.set_title("(a) Plastic-flow stress–strain", fontsize=9)
axA.grid(alpha=0.25)
handles = [Line2D([0], [0], color=c, lw=1.8, label=l) for _, l, c in mats]
handles += [Line2D([0], [0], color="0.35", lw=1.6, ls="-", label="FFT (full-field)"),
           Line2D([0], [0], color="0.35", lw=1.4, ls="--", label="Taylor (iso-strain)")]
axA.legend(handles=handles, fontsize=6.4, loc="lower right", ncol=2,
           handlelength=1.6, columnspacing=0.9)

axB.axhline(0, color="k", lw=0.7, ls=":")
axB.set_xlabel(r"Logarithmic strain $\varepsilon_{11}$")
axB.set_ylabel(r"Flow-stress bias (Taylor$-$FFT)/FFT [\%]")
axB.set_title("(b) Homogenization bias in plastic flow", fontsize=9)
axB.grid(alpha=0.25)
# group takeaways placed in the empty middle band (no boxed legend)
axB.text(0.5, 0.72, "FCC: Taylor over-predicts", transform=axB.transAxes,
         ha="center", va="center", fontsize=7.5, style="italic", color="0.42")
axB.text(0.5, 0.30, "HCP: Taylor under-predicts", transform=axB.transAxes,
         ha="center", va="center", fontsize=7.5, style="italic", color="0.42")

fig.tight_layout()
out_png = os.path.join(fig_dir, "Fig_plastic_flow_taylor_vs_fft.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("saved", out_png)
