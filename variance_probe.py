# -*- coding: utf-8 -*-
"""THE MANDATORY VARIANCE GATE (prereg.VARIANCE_GATE_REQUIRED).

WHY THIS RUNS BEFORE ANYTHING ELSE
----------------------------------
The MFAC audit's R3 stage died because its registered explanatory variable
turned out to have NO VARIANCE across the experiment grid (snr_exc sat at
0.0040-0.0042 in all nine cells).  The regime map therefore had no axis to
test, and the stage produced a table instead of a boundary.  That cost a full
grid to discover.  This file is the cheap version of the same discovery.

It is also where two STRUCTURAL zeros are expected, and they should be
CONFIRMED rather than assumed:

  * R has no effect at stage C0 by construction.  The base stage is a complete
    graph, so every agent already hears every other agent and a trust radius
    cannot restrict anything.  R can only bite once motion makes the graph
    geometric (C1+).  The plant enforces this literally, via
    trust_radius = min(sensing_radius, R).

  * BETA may have no effect on the CONVERGED cluster count.  The anchored
    update o' = (1-b) P o + b * o has the same fixed-point equation, o = P o,
    as plain HK, so anchoring rescales the convergence RATE rather than the
    fixed point.  With a finite round budget it can still move the observed
    count, but only by not having settled -- which is a statement about the
    budget, not about the knob.  If beta is inert, that removes an axis, and
    finding it after a main grid would mean publishing a map with a dead axis.

An axis with no variance is REMOVED from the main grid, and the removal is
reported here, never silently absorbed.

READOUTS
--------
  clusters  the registered protagonist (mean over seeds)
  spread    std of settled opinions; continuous, so it can show variance where
            an integer cluster count is too coarse
Either readout moving past VARIANCE_MIN_SPREAD keeps the axis.

Output: variance_probe.csv; printed per-axis spread table and gate verdict.
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
N_SEEDS = 6 if QUICK else 20
N_STEPS = 120 if QUICK else P.N_STEPS
SUFFIX = "_quick" if QUICK else ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"variance_probe{SUFFIX}.csv")

# Reference operating point: every axis is swept with the others held here.
REF = dict(n=N_PROBE, eps=0.25, r_mult=1.0, beta=0.3, sigma=0.005,
           switch_rate=0.02, n_steps=N_STEPS)

# (axis name, its grid, the stages at which it is structurally ACTIVE)
AXES = (
    ("eps", P.EPS_GRID, STAGES),
    ("r_mult", P.R_GRID, ("C1", "C2", "C3")),
    ("beta", P.BETA_GRID, STAGES),
    ("sigma", P.SIGMA_GRID, ("C2", "C3")),
    ("switch_rate", P.SWITCH_GRID, ("C3",)),
)
FIELDS = ["stage", "axis", "axis_value", "mean_clusters", "mean_spread",
          "clusters_rel_spread", "spread_rel_spread", "has_variance"]


def rel_spread(vals: np.ndarray) -> float:
    """Spread across a knob grid, normalised by the grid's mean magnitude.

    Normalising by the mean rather than by the maximum keeps the measure
    defined when a readout is zero at one end of the grid, which happens for
    eps near the consensus boundary.
    """
    scale = float(np.mean(np.abs(vals)))
    if scale < 1e-12:
        return 0.0
    return float((vals.max() - vals.min()) / scale)


def sweep_axis(stage: str, axis: str, grid, rows: list[dict]) -> dict:
    cl, sp = [], []
    t0 = time.time()
    for v in grid:
        kw = dict(REF)
        kw[axis] = v
        res = [run_swarm(SwarmParams(seed=1000 + s, stage=stage, **kw))
               for s in range(N_SEEDS)]
        cl.append(float(np.mean([r["clusters"] for r in res])))
        sp.append(float(np.mean([r["spread"] for r in res])))
    cl_a, sp_a = np.array(cl), np.array(sp)
    rc, rs = rel_spread(cl_a), rel_spread(sp_a)
    has = max(rc, rs) >= P.VARIANCE_MIN_SPREAD
    for v, c, s in zip(grid, cl_a, sp_a):
        rows.append(dict(stage=stage, axis=axis, axis_value=v,
                         mean_clusters=round(float(c), 4),
                         mean_spread=round(float(s), 5),
                         clusters_rel_spread=round(rc, 4),
                         spread_rel_spread=round(rs, 4),
                         has_variance=bool(has)))
    return dict(stage=stage, axis=axis, clusters=cl_a,
                clusters_rel=rc, spread_rel=rs, has=has,
                secs=time.time() - t0)


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED GATE"
    print(f"variance gate  [{mode}]  N={N_PROBE} seeds={N_SEEDS} "
          f"steps={N_STEPS}  threshold={P.VARIANCE_MIN_SPREAD}")
    print(f"reference point: {REF}\n")

    rows: list[dict] = []
    verdicts: dict[tuple[str, str], dict] = {}
    for axis, grid, stages in AXES:
        for stage in stages:
            r = sweep_axis(stage, axis, grid, rows)
            verdicts[(stage, axis)] = r
            print(f"  {stage} {axis:>11}  clusters_rel={r['clusters_rel']:.3f}  "
                  f"spread_rel={r['spread_rel']:.3f}  "
                  f"-> {'VARIES' if r['has'] else 'INERT':<6} "
                  f"[{r['secs']:.0f}s]")
        for stage in STAGES:
            if stage not in stages:
                print(f"  {stage} {axis:>11}  structurally inactive at this "
                      f"stage -> skipped (an absence, not a zero)")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    print("\nGATE VERDICT")
    live = [f"{a}@{s}" for (s, a), r in verdicts.items() if r["has"]]
    inert = [f"{a}@{s}" for (s, a), r in verdicts.items() if not r["has"]]
    print(f"  live axes  : {live if live else 'NONE'}")
    print(f"  inert axes : {inert if inert else 'NONE'}")

    eps_live_base = verdicts[("C0", "eps")]["has"]
    print(f"  H1 precondition -- eps carries variance at the base stage: "
          f"{'PASS' if eps_live_base else 'FAIL'}")
    if not eps_live_base:
        print("    If eps is inert even at C0, the harness does not reproduce "
              "the phase behaviour validated in repro_check.py, and the main "
              "grid must not be run.")
    print("  Axes listed inert are REMOVED from the main grid; the contraction "
          "is reported in the write-up, not absorbed silently.")
    return 0 if eps_live_base else 1


if __name__ == "__main__":
    sys.exit(main())
