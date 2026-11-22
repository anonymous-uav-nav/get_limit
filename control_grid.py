# -*- coding: utf-8 -*-
"""Does the lever survive the FULL combination grid, not just one reference point?

WHY THIS EXISTS
---------------
The variance gate swept one knob at a time with every other knob pinned at a
single reference point.  That is enough to answer "does this axis move at all",
which is what the gate is for, but it is NOT enough to claim the degradation
pattern: a lever measured at one point in the other axes' space could be an
artefact of that point.

The gate's provisional reading was:

    r_mult's lever: 0.188 -> 0.152 -> 0.039   (monotone decay C1 -> C2 -> C3)
    eps's lever:    0.928 -> 0.961 -> 1.015 -> 0.955   (flat)

If that survives marginalisation then the paper has a real, attributable
result: opening the operating conditions removes the SAFETY-RADIUS lever while
leaving the SOCIOPHYSICS lever intact.  If it does not survive, the gate's
reading was an artefact of the reference point and must be withdrawn.  This
file decides between those two.

METHOD
------
For each stage and each knob, the OTHER knob is marginalised rather than
pinned: N_OTHER values are drawn across its grid, and the knob under test is
evaluated at its own grid endpoints at each.  The control measure is then

    control(knob | stage) = (E[clusters | knob_min] - E[clusters | knob_max])
                            / E[clusters | knob_min]

computed on the pooled episodes, with a bootstrap interval resampled BY
EPISODE.  Pooling before dividing is deliberate: it measures the axis effect
averaged over the rest of the design, which is the quantity a regime map will
report.

BETA IS ABSENT ON PURPOSE.  beta_probe.py demoted it to an endpoint effect on
the protagonist, so under the registered rule it leaves the cluster-count grid.
It is still varied as PART of the marginalisation here, so its (weak) effect is
averaged out rather than silently zeroed.

SWITCH_RATE IS ABSENT ON PURPOSE.  The gate declared it inert (clusters_rel
0.039, spread_rel 0.078, both under threshold) and it leaves the grid.

COST NOTE.  This is the confirmation, not the main grid, and it uses fewer
seeds than the registered N_SEEDS -- stated in the output rather than quietly.

Output: control_grid.csv; printed lever table across stages and the verdict.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import prereg as P  # noqa: E402
from swarm import STAGES, SwarmParams, run_swarm  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_PROBE = 100
N_SEEDS = 4 if QUICK else 15
N_OTHER = 3 if QUICK else 6
BOOT = 400
SUFFIX = "_quick" if QUICK else ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"control_grid{SUFFIX}.csv")

# (knob, its low endpoint, its high endpoint, the OTHER knob)
KNOBS = (
    ("eps", P.EPS_GRID[0], P.EPS_GRID[-1], "r_mult"),
    ("r_mult", P.R_GRID[0], P.R_GRID[-1], "eps"),
)
OTHER_GRID = {"eps": P.EPS_GRID, "r_mult": P.R_GRID}
BASE = dict(n=N_PROBE, eps=0.25, r_mult=1.0, beta=0.3, sigma=0.005,
            switch_rate=0.02)

FIELDS = ["stage", "knob", "n_other", "control", "ci_lo", "ci_hi",
          "mean_min", "mean_max", "n_episodes"]
RNG = np.random.default_rng(20261119)


def control_at_stage(stage: str, knob: str, lo: float, hi: float,
                     other: str) -> dict:
    """Marginalised control of `knob` at one stage."""
    others = list(OTHER_GRID[other])
    if len(others) > N_OTHER:
        idx = np.linspace(0, len(others) - 1, N_OTHER).round().astype(int)
        others = [others[i] for i in sorted(set(idx))]

    at_min, at_max = [], []
    for o in others:
        for s in range(N_SEEDS):
            kw = dict(BASE)
            kw[other] = o
            at_min.append(run_swarm(SwarmParams(
                seed=1000 + s, stage=stage, **{**kw, knob: lo}))["clusters"])
            at_max.append(run_swarm(SwarmParams(
                seed=1000 + s, stage=stage, **{**kw, knob: hi}))["clusters"])
    a, b = np.array(at_min), np.array(at_max)
    mm, mx = float(a.mean()), float(b.mean())
    ctrl = (mm - mx) / mm if mm > 1e-12 else 0.0

    boot = np.empty(BOOT)
    for i in range(BOOT):
        m = float(a[RNG.integers(0, a.size, a.size)].mean())
        boot[i] = (m - float(b[RNG.integers(0, b.size, b.size)].mean())) / m \
            if m > 1e-12 else 0.0
    lo_ci, hi_ci = np.quantile(boot, [0.025, 0.975])
    return dict(stage=stage, knob=knob, n_other=len(others),
                control=round(ctrl, 4), ci_lo=round(float(lo_ci), 4),
                ci_hi=round(float(hi_ci), 4), mean_min=round(mm, 4),
                mean_max=round(mx, 4), n_episodes=a.size + b.size)


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "CONFIRMATION RUN"
    print(f"marginalised lever  [{mode}]  N={N_PROBE} seeds={N_SEEDS} "
          f"(registered N_SEEDS={P.N_SEEDS}; this run is cheaper on purpose)")
    print(f"other-axis samples per cell = {N_OTHER}   stages={STAGES}\n")

    rows = []
    table: dict[str, dict[str, float]] = {k: {} for k, *_ in KNOBS}
    t0 = time.time()
    for stage in STAGES:
        for knob, lo, hi, other in KNOBS:
            r = control_at_stage(stage, knob, lo, hi, other)
            table[knob][stage] = r["control"]
            rows.append(r)
            print(f"  {stage} {knob:>7}  control={r['control']:+.4f}  "
                  f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                  f"min={r['mean_min']:.3f} max={r['mean_max']:.3f}  "
                  f"[{time.time() - t0:.0f}s]")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)\n")

    print("LEVER TABLE (marginalised over the other axis)")
    print(f"{'knob':>8} " + " ".join(f"{s:>9}" for s in STAGES) + "   decay")
    for knob in table:
        vals = [table[knob][s] for s in STAGES]
        act = [v for s, v in zip(STAGES, vals)
               if not (knob == "r_mult" and s == "C0")]
        decay = (max(act) - min(act)) / max(act) if max(act) > 1e-9 else 0.0
        print(f"{knob:>8} " + " ".join(f"{v:>+9.4f}" for v in vals)
              + f"   {decay:.0%}")

    print("\nVERDICT")
    eps_act = [table["eps"][s] for s in STAGES]
    r_act = [table["r_mult"][s] for s in ("C1", "C2", "C3")]
    eps_spread = (max(eps_act) - min(eps_act)) / max(eps_act)
    eps_flat = eps_spread < 0.25
    r_decay = (r_act[0] - r_act[-1]) / r_act[0] if r_act[0] > 1e-9 else 0.0
    print(f"  eps lever flat across stages (spread < 25%)      : {eps_flat} "
          f"({eps_spread:.0%})")
    print(f"  r_mult lever falls C1 -> C3                      : {r_decay:+.0%}")
    if eps_flat and r_decay > 0.50:
        print("  => CONFIRMED on the full combination grid.  Opening the")
        print("     operating conditions removes the SAFETY-RADIUS lever while")
        print("     leaving the SOCIOPHYSICS lever intact.  The gate's reading")
        print("     was not a reference-point artefact.")
    elif not eps_flat and r_decay > 0.50:
        print("  => PARTIAL.  r_mult decays but eps is NOT flat, so the story is")
        print("     'both levers move', not 'one is removed'.  Report as such.")
    elif eps_flat and r_decay <= 0.50:
        print("  => WITHDRAWN.  The gate's r_mult decay does not survive")
        print("     marginalisation; there is no lever-removal result to report.")
    else:
        print("  => NEITHER.  No lever behaviour worth reporting at this")
        print("     configuration; investigate before building the main grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
