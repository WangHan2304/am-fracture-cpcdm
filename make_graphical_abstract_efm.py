#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_graphical_abstract_efm.py

Build the Elsevier/EFM (Engineering Fracture Mechanics) graphical abstract as a
single wide image, exported to PNG / PDF / TIFF / EPS. Card-style layout with a
header, a rule, and three badge-labelled panels summarising the paper's three
results:

  A  Leakage-free validation protocol (hold one build orientation out, refit a
     single damage strength S0, predict blind -- no data leakage).
  B  Taylor iso-strain homogenization bias flips sign with the kinematic
     constraint, not with the damage law (2D plane-strain over-estimate vs
     3D free under-estimate, Ti-6Al-4V).
  C  Mechanism-keyed applicability gate (xi0 >= 3): the gated hybrid is never
     worse than the isotropic baseline.

All numbers are quoted from the manuscript / HEADLINE_BINDINGS (LOOCV blind
fracture-strain error): gated hybrid 15.9, isotropic baseline 24.3, ungated
directional 31.2 (%); Taylor bias Ti-6Al-4V 2D -26.5/-32.6/-35.8,
3D +92.7/+61.6/+49.8 (%). The script writes nothing outside its output folder.

EFM size requirement (Guide for Authors): at least 531 x 1328 pixels (h x w),
readable at 5 x 13 cm on a 96 dpi screen. The canvas below renders 3240 x 1290
px (w x h) at 300 dpi, i.e. well above the minimum in both directions.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "submission_EFM", "graphical_abstract")
os.makedirs(OUTDIR, exist_ok=True)
PNG = os.path.join(OUTDIR, "graphical_abstract_efm.png")
PDF = os.path.join(OUTDIR, "graphical_abstract_efm.pdf")
TIFF = os.path.join(OUTDIR, "graphical_abstract_efm.tif")
EPS = os.path.join(OUTDIR, "graphical_abstract_efm.eps")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 8.5, "axes.linewidth": 0.8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

# --------------------------------------------------------------------------- palette
INK = "#22303c"
TEAL = "#2a9d8f"      # gated hybrid / 3D (best, positive)
GREY = "#8d99ae"      # isotropic baseline (reference)
ORANGE = "#e76f51"    # ungated directional / 2D (worst, negative)
BLUE = "#1f6f8b"      # brand / panel A
CARD = "#f5f7f9"
CARD_EDGE = "#d7dee6"

fig = plt.figure(figsize=(10.8, 4.3), dpi=200)
fig.patch.set_facecolor("white")

# --------------------------------------------------------------------------- header
fig.text(0.5, 0.992,
         "A leakage-free validation protocol and mechanism-keyed applicability domain for\n"
         "orientation-resolved ductile damage descriptors in additively manufactured metals",
         ha="center", va="top", fontsize=10.5, weight="bold", color=INK)
fig.text(0.5, 0.902,
         "validated out-of-sample \u00b7 homogenization bias quantified \u00b7 applicability gated "
         "\u2014 a screening rule that is never worse than the baseline",
         ha="center", va="top", fontsize=9.0, color="#5a6b7b")
fig.add_artist(Line2D([0.04, 0.96], [0.868, 0.868], color=TEAL, lw=1.6))

# --------------------------------------------------------------------------- cards
CARD_Y0, CARD_H = 0.055, 0.78
CARD_X0 = [0.035, 0.355, 0.675]
CARD_W = 0.29
for x0 in CARD_X0:
    fig.patches.append(FancyBboxPatch(
        (x0, CARD_Y0), CARD_W, CARD_H, transform=fig.transFigure,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        facecolor=CARD, edgecolor=CARD_EDGE, lw=1.0, zorder=-10))

def badge(x, y, letter, color):
    fig.text(x, y, letter, ha="center", va="center", fontsize=9.5,
             weight="bold", color="white", zorder=5,
             bbox=dict(boxstyle="circle,pad=0.34", facecolor=color, edgecolor="none"))

def panel_title(x, y, text):
    fig.text(x, y, text, ha="left", va="center", fontsize=10.2,
             weight="bold", color=INK)

TY = CARD_Y0 + CARD_H - 0.045
badge(CARD_X0[0] + 0.030, TY, "A", BLUE);   panel_title(CARD_X0[0] + 0.055, TY, "Leakage-free validation")
badge(CARD_X0[1] + 0.030, TY, "B", ORANGE); panel_title(CARD_X0[1] + 0.055, TY, "Taylor bias flips sign")
badge(CARD_X0[2] + 0.030, TY, "C", TEAL);   panel_title(CARD_X0[2] + 0.055, TY, "Applicability gate")

# --------------------------------------------------------------------------- Panel A
axA = fig.add_axes([0.052, 0.12, 0.255, 0.62]); axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
axA.set_zorder(2); axA.patch.set_visible(False)

def box(ax, x, y, w, h, text, fc, ec, fs=8.0, tc="white", ls="solid", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.05",
                 facecolor=fc, edgecolor=ec, linestyle=ls, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc)

chips = [("0\u00b0 (BD)", BLUE, "solid", "white"),
         ("45\u00b0", BLUE, "solid", "white"),
         ("90\u00b0 (trans.)", "#cdd6de", "dashed", INK)]
for i, (lab, cc, ls, tc) in enumerate(chips):
    box(axA, 0.02, 0.70 - i * 0.25, 0.34, 0.17, lab, cc, "#4a5b6b",
        fs=8.0, tc=tc, ls=ls, lw=1.1)
axA.text(0.19, 0.94, "train (2 orient.)", ha="center", fontsize=7.2, color=BLUE)
axA.text(0.19, 0.12, "held out (blind)", ha="center", va="center", fontsize=7.2, color="#a23b2e")

box(axA, 0.46, 0.56, 0.52, 0.26, "Refit ONE $S_0$\n(2 training orient.)",
    "white", BLUE, fs=8.0, tc=INK, lw=1.2)
box(axA, 0.46, 0.18, 0.52, 0.26, "Predict held-out\nBLIND \u00b7 no leakage",
    "white", TEAL, fs=8.0, tc=INK, lw=1.2)
axA.add_patch(FancyArrowPatch((0.38, 0.82), (0.50, 0.72), arrowstyle="-|>",
              mutation_scale=10, lw=1.0, color="#4a5b6b"))
axA.add_patch(FancyArrowPatch((0.38, 0.32), (0.44, 0.32), arrowstyle="-|>",
              mutation_scale=10, lw=1.0, color="#4a5b6b"))
axA.add_patch(FancyArrowPatch((0.72, 0.55), (0.72, 0.45), arrowstyle="-|>",
              mutation_scale=10, lw=1.0, color="#4a5b6b"))
axA.text(0.50, 0.02, "orientation multipliers removed \u00b7 literature $g_s$ \u00b7 9 folds",
         ha="center", fontsize=7.0, color="#5a6b7b")

# --------------------------------------------------------------------------- Panel B
axB = fig.add_axes([0.405, 0.185, 0.215, 0.51])
axB.set_zorder(2); axB.patch.set_visible(False)
orient_lab = ["0\u00b0", "45\u00b0", "90\u00b0"]
b2d = [-26.5, -32.6, -35.8]
b3d = [92.7, 61.6, 49.8]
h = 0.36
axB.barh([y + h / 2 for y in range(3)], b2d, height=h, color=ORANGE, label="2D plane strain")
axB.barh([y - h / 2 for y in range(3)], b3d, height=h, color=TEAL, label="3D free")
axB.axvline(0, color=INK, lw=1.3)
for y, v in zip(range(3), b2d):
    axB.text(v - 4, y + h / 2, f"{v:.0f}%", va="center", ha="right", fontsize=7.3, color=ORANGE)
for y, v in zip(range(3), b3d):
    axB.text(v + 4, y - h / 2, f"+{v:.0f}%", va="center", ha="left", fontsize=7.3, color="#1c7368")
axB.set_yticks(range(3)); axB.set_yticklabels(orient_lab)
axB.invert_yaxis()
axB.set_xlim(-62, 155)
axB.set_ylabel("Ti-6Al-4V", fontsize=8)
axB.set_xlabel("fracture-strain bias vs Taylor (%)", fontsize=7.5, labelpad=4)
axB.legend(fontsize=7.3, loc="lower right", framealpha=1.0)
axB.text(0.5, 1.06, "sign set by constraint, not damage law",
         transform=axB.transAxes, fontsize=7.3, ha="center", va="bottom",
         color="#a23b2e", style="italic")
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(left=False)

# --------------------------------------------------------------------------- Panel C
axC = fig.add_axes([0.732, 0.15, 0.205, 0.545])
axC.set_zorder(2); axC.patch.set_visible(False)
names = ["Gated\nhybrid", "Isotropic\nbaseline", "Ungated\ndirectional"]
vals = [15.9, 24.3, 31.2]
colors = [TEAL, GREY, ORANGE]
bars = axC.bar(range(3), vals, color=colors, width=0.64)
for b, v in zip(bars, vals):
    axC.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}", ha="center",
             va="bottom", fontsize=8.6, weight="bold", color=INK)
axC.axhline(24.3, color=GREY, ls="--", lw=1.1)
axC.text(0, 15.9 + 4.6, "never\nworse", ha="center", va="bottom",
         fontsize=7.2, color="#1c7368", weight="bold")
axC.set_xticks(range(3)); axC.set_xticklabels(names, fontsize=7.6)
axC.set_ylim(0, 40)
axC.set_ylabel("blind fracture-strain error (%)", fontsize=7.5)
axC.text(0.5, 1.06, "gate: $\\xi_0 \\geq 3 \\rightarrow$ directional, else isotropic",
         transform=axC.transAxes, fontsize=7.3, ha="center", va="bottom", color=BLUE)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# --------------------------------------------------------------------------- export
fig.savefig(PNG, dpi=300)
fig.savefig(PDF)
fig.savefig(TIFF, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(EPS)
plt.close(fig)
for p in (PNG, PDF, TIFF, EPS):
    print("wrote", p)
