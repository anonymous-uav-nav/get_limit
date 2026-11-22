# -*- coding: utf-8 -*-
"""Phase-diagram figures for the Physica A submission (fig4-fig6).

Three figures, each doing a different job:

  fig4  order parameter S in the (eps, R) plane, one panel per condition and
        swarm size, eight panels.  Shows WHERE the transition lies and whether
        the boundary moves as idealisations are relaxed.
  fig5  the same grid for the susceptibility chi.  The transition is a CURVE,
        and chi is what makes it visible; a diagram of S alone smears the
        boundary over the region where clusters are still merging.
  fig6  the chi ridge as eps*(R) per condition, superimposed.  This is the
        figure that answers the paper's question at the level of the phase
        diagram: if relaxing an idealisation moves the boundary, the ridges
        separate, and by how much is readable off the axis.

Every value is read from phase_raw.csv at run time.  Nothing is typed in.

Output: figs/fig4_phase_S.{png,pdf}, figs/fig5_phase_chi.{png,pdf},
        figs/fig6_ridge_shift.{png,pdf}
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 140, "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
SRC = os.path.join(HERE, "phase_raw.csv")

STAGES = ("C0", "C1", "C2", "C3")
STAGE_LABEL = {"C0": "C0 idealised", "C1": "C1 +graph dynamics",
               "C2": "C2 +measurement noise", "C3": "C3 +link failure"}
CB_LABEL = {"S_max": r"order parameter $S$", "chi": r"susceptibility $\chi$"}


def load() -> list[dict]:
    with open(SRC, newline="", encoding="utf-8") as fh:
        return [{**r,
                 "n_swarm": int(float(r["n_swarm"])),
                 "eps": float(r["eps"]),
                 "r_mult": float(r["r_mult"]),
                 "S_max": float(r["S_max"]),
                 "chi": float(r["chi"])}
                for r in csv.DictReader(fh)]


def grid(rows: list[dict], stage: str, n: int, key: str):
    sub = [r for r in rows if r["stage"] == stage and r["n_swarm"] == n]
    eps = np.array(sorted({r["eps"] for r in sub}))
    rm = np.array(sorted({r["r_mult"] for r in sub}))
    z = np.full((rm.size, eps.size), np.nan)
    for r in sub:
        i = int(np.argmin(np.abs(rm - r["r_mult"])))
        j = int(np.argmin(np.abs(eps - r["eps"])))
        z[i, j] = r[key]
    return eps, rm, z


def panel_figure(key: str, stem: str, title: str, cmap: str,
                 vmin=None, vmax=None, per_row: bool = False) -> None:
    rows = load()
    ns = sorted({r["n_swarm"] for r in rows})
    fig, axes = plt.subplots(len(ns), len(STAGES),
                             figsize=(3.05 * len(STAGES), 2.15 * len(ns)),
                             squeeze=False)
    im = None
    for a, n in enumerate(ns):
        sub_ns = [r for r in rows if r["n_swarm"] == n]
        lo = min(r[key] for r in sub_ns) if vmin is None else vmin
        hi = max(r[key] for r in sub_ns) if vmax is None else vmax
        for b, stage in enumerate(STAGES):
            ax = axes[a][b]
            eps, rm, z = grid(rows, stage, n, key)
            im = ax.pcolormesh(eps, rm, z, cmap=cmap, vmin=lo, vmax=hi,
                               shading="auto")
            ax.set_title(f"{STAGE_LABEL[stage]}\n$N={n}$", fontsize=8)
            if a == len(ns) - 1:
                ax.set_xlabel(r"$\varepsilon$")
            else:
                # the upper row's tick labels collide with the lower row's
                # titles otherwise
                ax.tick_params(labelbottom=False)
            if b == 0:
                ax.set_ylabel(r"trust radius $R/R_{\mathrm{sense}}$")
        cbar = fig.colorbar(im, ax=axes[a], shrink=0.85, pad=0.02,
                            label=CB_LABEL[key])
        if per_row:
            cbar.set_ticks([lo, (lo + hi) / 2, hi])
            cbar.ax.set_ylim(lo, hi)
        else:
            cbar.set_ticks([lo, (lo + hi) / 2, hi])
    fig.suptitle(title, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote figs/{stem}.png and .pdf")


def ridge_figure() -> None:
    """chi ridge eps*(R) per condition, both sizes, on one pair of axes."""
    rows = load()
    ns = sorted({r["n_swarm"] for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 3.4),
                             squeeze=False)
    handles, labels = [], []
    marks = {"C0": ("o", "-"), "C1": ("s", "--"), "C2": ("^", ":"),
             "C3": ("v", "-.")}
    for a, n in enumerate(ns):
        ax = axes[0][a]
        for stage in STAGES:
            eps, rm, z = grid(rows, stage, n, "chi")
            ridge = [eps[int(np.nanargmax(z[i]))] for i in range(rm.size)]
            mk, ls = marks[stage]
            ln, = ax.plot(rm, ridge, marker=mk, ls=ls,
                          color="black" if stage in ("C0", "C2") else "0.45",
                          ms=4, lw=1.1, label=STAGE_LABEL[stage])
            if a == 0:
                handles.append(ln)
                labels.append(STAGE_LABEL[stage])
        ax.set_xlabel(r"trust radius $R/R_{\mathrm{sense}}$")
        ax.set_ylabel(r"ridge position $\varepsilon^\ast$")
        ax.set_title(f"$N={n}$")
        ax.set_ylim(0.205, 0.272)
        ax.set_yticks([0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27])
        ax.margins(y=0.06)
        ax.grid(alpha=0.3, lw=0.5)
    # a legend inside either panel sits on the C0 ridge, which is flat near
    # the top of the axis; move it below the panels instead
    fig.legend(handles, labels, loc="lower center", ncol=len(STAGES),
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"fig6_ridge_shift.{ext}"))
    plt.close(fig)
    print("  wrote figs/fig6_ridge_shift.png and .pdf")


def main() -> int:
    if not os.path.exists(SRC):
        print(f"missing {os.path.basename(SRC)} -- run phase_map.py first")
        return 1
    print("phase-diagram figures from phase_raw.csv")
    panel_figure("S_max", "fig4_phase_S",
                 r"Order parameter $S$ in the $(\varepsilon, R)$ plane",
                 "viridis", vmin=0.0, vmax=1.0)
    panel_figure("chi", "fig5_phase_chi",
                 r"Susceptibility $\chi$ in the $(\varepsilon, R)$ plane",
                 "magma", per_row=True)
    ridge_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())