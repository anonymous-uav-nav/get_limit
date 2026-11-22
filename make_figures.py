# -*- coding: utf-8 -*-
"""Generate the three paper figures FROM THE CSVs.

Every plotted value is read from an artefact at run time.  Nothing is typed in
from a terminal printout, for the same reason make_results.py exists: a printed
number can come from code that later changes, and this project has already
produced one normalised value that turned out to be a division artefact.

Figure conventions
------------------
* Error bars are the bootstrap intervals recorded in the CSVs, not recomputed
  here, so the figure and RESULTS.md cannot disagree.
* The withdrawn claim is drawn as withdrawn: figure 2 shows the round-2 slope
  estimates that replaced the first round's retracted reading, rather than
  showing the retracted reading as if it still stood.
* Grayscale-safe: series are distinguished by marker and line style as well as
  by shade, because MDPI review often prints in black and white.

Output: figs/fig1_phase.{png,pdf}, figs/fig2_leverage.{png,pdf},
        figs/fig3_beta.{png,pdf}
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
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)


def load(name: str) -> list[dict]:
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fl(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError):
            out.append(float("nan"))
    return np.array(out)


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote figs/{stem}.png and .pdf")


def fig1_phase() -> None:
    """eps_c against swarm size, with the published targets as bands."""
    rows = load("repro_check.csv")
    ns = sorted({int(float(r["n"])) for r in rows})
    ec = []
    for n in ns:
        sub = [r for r in rows if int(float(r["n"])) == n]
        e = fl(sub, "eps")
        f = fl(sub, "consensus_frac")
        o = np.argsort(e)
        e, f = e[o], f[o]
        above = f >= 0.5
        if not above.any():
            ec.append(np.nan)
            continue
        i = int(np.argmax(above))
        if i == 0:
            ec.append(float(e[0]))
        else:
            y0, y1, x0, x1 = f[i - 1], f[i], e[i - 1], e[i]
            ec.append(float(x1) if y1 == y0
                      else float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))
    ec = np.array(ec)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhspan(0.25 - 0.03, 0.25 + 0.03, color="0.85", zorder=0)
    ax.axhspan(0.20 - 0.03, 0.20 + 0.03, color="0.70", zorder=0)
    ax.plot(ns, ec, "o-", color="black", lw=1.2, ms=5, zorder=3)
    ax.axhline(0.25, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.axhline(0.20, color="0.35", ls="--", lw=0.9, zorder=1)
    ax.annotate("published $\\epsilon_c(N{=}100)\\approx0.25$", (ns[1], 0.25),
                xytext=(6, 6), textcoords="offset points", fontsize=7,
                color="0.25")
    ax.annotate("published large-$N$ limit $\\approx0.20$",
                (ns[-1], 0.20), xytext=(-6, -12), textcoords="offset points",
                fontsize=7, color="0.25", ha="right")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("population size $N$")
    ax.set_ylabel("critical confidence threshold $\\epsilon_c$")
    ax.set_title("Instrument validation: reproducing the published diagram")
    save(fig, "fig1_phase")


def fig2_leverage() -> None:
    """Leverage across operating stages, and the round-2 R slopes."""
    c = load("audit_control.csv")
    stages = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    styles = {("eps", 100): ("o", "-", "black"),
              ("eps", 400): ("s", "--", "0.45"),
              ("r_mult", 100): ("^", "-", "black"),
              ("r_mult", 400): ("v", "--", "0.45")}
    for (knob, nsw), (mk, ls, col) in styles.items():
        vals, los, his = [], [], []
        for s in stages:
            r = [x for x in c if x["stage"] == s and x["knob"] == knob
                 and int(float(x["n_swarm"])) == nsw]
            vals.append(float(r[0]["control"]) if r else np.nan)
            los.append(float(r[0]["ci_lo"]) if r else np.nan)
            his.append(float(r[0]["ci_hi"]) if r else np.nan)
        vals, los, his = map(np.array, (vals, los, his))
        ax.errorbar(range(len(stages)), vals,
                    yerr=[vals - los, his - vals], marker=mk, ls=ls,
                    color=col, lw=1.1, ms=4.5, capsize=2.5,
                    label=f"{knob}, $N$={nsw}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(["C0\nbase", "C1\n+motion", "C2\n+noise",
                        "C3\n+switching"])
    ax.set_ylabel("control on cluster count")
    ax.set_title("(a) leverage vs operating conditions")
    # the eps series occupies the 0.55-0.65 band and upper right would sit on
    # its C2/C3 points; the 0.12-0.53 band is empty, so the legend goes there
    ax.legend(loc="center", framealpha=1.0)

    ax = axes[1]
    s2 = load("r2_slopes.csv")
    ctrl = sorted([r for r in s2 if r["arm"] == "control"],
                  key=lambda z: float(z["n_swarm"]))
    xs = list(range(len(ctrl)))
    ys = np.array([float(r["slope"]) for r in ctrl])
    err = np.array([[float(r["slope"]) - float(r["ci_lo"]),
                     float(r["ci_hi"]) - float(r["slope"])] for r in ctrl]).T
    ax.errorbar(xs, ys, yerr=err, fmt="o", color="black", ms=6, capsize=4,
                lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$N$={int(float(r['n_swarm']))}" for r in ctrl])
    ax.set_ylabel("slope of clusters vs $R$")
    ax.set_title("(b) round 2: $R$ lever, non-saturating grid")
    ax.annotate("interval spans 0", (xs[-1], ys[-1]), xytext=(-4, 12),
                textcoords="offset points", fontsize=7, ha="right")
    fig.tight_layout()
    save(fig, "fig2_leverage")


def fig3_beta() -> None:
    """Cluster count against the anchoring weight, corner marked."""
    rows = load("beta_probe.csv")
    stages = sorted({r["stage"] for r in rows})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    marks = {"C0": ("o", "-"), "C2": ("s", "--"), "C3": ("^", ":")}
    for st in stages:
        sub = sorted([r for r in rows if r["stage"] == st],
                     key=lambda r: float(r["beta"]))
        mk, ls = marks.get(st, ("o", "-"))
        ax.plot(fl(sub, "beta"), fl(sub, "mean_clusters"), marker=mk, ls=ls,
                color="black" if st == "C0" else "0.45", lw=1.1, ms=4, label=st)
    ax.set_yscale("log")
    ax.set_xlabel("anchoring weight $\\beta$")
    ax.set_ylabel("emergent cluster count")
    ax.set_title("Anchoring weight: flat interior, frozen corner")
    ax.axvline(1.0, color="0.3", ls=":", lw=1.0)
    ax.annotate("degenerate corner\n(frozen swarm)", (1.0, 37.25),
                xytext=(-8, -4), textcoords="offset points", fontsize=7,
                ha="right", va="top")
    ax.legend(title="stage", loc="center left")
    save(fig, "fig3_beta")


def main() -> int:
    print("generating figures from CSV artefacts")
    fig1_phase()
    fig2_leverage()
    fig3_beta()
    print(f"\nall figures in {FIGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
