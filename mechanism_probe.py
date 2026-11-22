# -*- coding: utf-8 -*-
"""LOCATE a candidate mediator for the sign flip of R's lever with swarm size.

WHAT IS BEING EXPLAINED
-----------------------
The main grid found R's lever on emergent cluster structure to be POSITIVE at
N = 100 (+0.058 / +0.044 / +0.035) and NEGATIVE at N = 400 (-0.025 / -0.017 /
-0.020), a clean 6/6 split by size.  That came out of a grid designed to test
whether a lever EXISTS, so the design was not aimed at it and it has no
mechanism attached.  This file is the attempt to attach one.

WHAT THIS FILE CAN AND CANNOT DO
--------------------------------
It LOCATES candidates.  A mediator whose response to R flips sign between
N = 100 and N = 400 in the same direction as the cluster count is a candidate.
Matching the sign pattern is NECESSARY, NOT SUFFICIENT: two quantities can both
flip because both respond to a third cause.  Confirming a candidate requires an
INTERVENTION -- holding the mediator fixed while R varies -- which this file
does not perform and does not claim.  The write-up must not describe a
candidate located here as a mechanism found.

CANDIDATES, AND WHAT EACH WOULD MEAN
------------------------------------
  trust_deg      how many agents a member can reach.  If this alone explained
                 it, the story is purely geometric.
  agree_yield    of the reachable agents, how many also pass the opinion test.
                 This is the CONFLUENCE of the two constraints: geometry picks
                 a set, bounded confidence then sub-selects from it.
  inwin_op_std   how opinion-DIVERSE each member's reachable set is.  The
                 armchair hypothesis was density: at larger N the opinion
                 distribution is denser, so widening the window admits more
                 diverse neighbours and bounded confidence then fragments
                 them.  This is what that hypothesis predicts should flip.
  op_range       the opinion range actually held, a sanity readout.

CONFIGURATION
-------------
Stage C1 (motion on, no noise, no dropout), so only the geometry varies with R.
Noise and switching would add two more sign-changing candidates and worsen the
attribution rather than improve it.

Output: mechanism_probe.csv; printed mediator table and candidate ranking.
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
from swarm import SwarmParams, mediators, run_swarm  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_LIST = (100, 400)
R_LIST = P.R_GRID
SEEDS = tuple(range(4 if QUICK else 20))
N_STEPS = 60 if QUICK else P.N_STEPS
STAGE = "C1"
SUFFIX = "_quick" if QUICK else ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"mechanism_probe{SUFFIX}.csv")

MED_KEYS = ("clusters", "spatial_deg", "trust_deg", "agree_yield",
            "inwin_op_std", "op_range")
FIELDS = ["n_swarm", "r_mult"] + list(MED_KEYS) + ["n_seeds"]


def measure(nsw: int, rm: float) -> dict[str, float]:
    acc = {k: [] for k in MED_KEYS}
    for s in SEEDS:
        sp = SwarmParams(seed=1000 + s, stage=STAGE, n=nsw, eps=0.25,
                         r_mult=rm, beta=0.3, sigma=0.0, switch_rate=0.0,
                         n_steps=N_STEPS)
        out = run_swarm(sp, return_state=True)
        med = mediators(sp, out["_pos"], out["_opinions"])
        for k in MED_KEYS:
            acc[k].append(out[k] if k in out else med[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED PROBE"
    print(f"mechanism probe  [{mode}]  stage={STAGE}  seeds={len(SEEDS)} "
          f"steps={N_STEPS}\n")

    rows, agg = [], {}
    t0 = time.time()
    for nsw in N_LIST:
        agg[nsw] = {}
        for rm in R_LIST:
            m = measure(nsw, rm)
            agg[nsw][rm] = m
            row = dict(n_swarm=nsw, r_mult=rm, n_seeds=len(SEEDS))
            for k in MED_KEYS:
                row[k] = round(m[k], 6)
            rows.append(row)
            print(f"  N={nsw:>4} R={rm:<4} " +
                  "  ".join(f"{k}={m[k]:.4f}" for k in MED_KEYS)
                  + f"   [{time.time() - t0:.0f}s]")
        print()

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} rows)\n")

    print("CHANGE FROM R_min TO R_max, normalised, per swarm size")
    print(f"{'quantity':>14} {'N=100':>12} {'N=400':>12}  same sign as clusters?")
    lo, hi = R_LIST[0], R_LIST[-1]
    ref: dict[str, dict[int, float]] = {}
    for nsw in N_LIST:
        a, b = agg[nsw][lo], agg[nsw][hi]
        for k in MED_KEYS:
            den = max(abs(a[k]), 1e-12)
            ref.setdefault(k, {})[nsw] = (a[k] - b[k]) / den
    c100, c400 = ref["clusters"][100], ref["clusters"][400]
    clusters_flip = (c100 > 0) != (c400 > 0)
    matches = []
    for k in MED_KEYS:
        same = ((ref[k][100] > 0) == (c100 > 0)
                and (ref[k][400] > 0) == (c400 > 0))
        tag = "MATCHES" if (k != "clusters" and same) else ""
        if tag:
            matches.append(k)
        print(f"{k:>14} {ref[k][100]:>+12.4f} {ref[k][400]:>+12.4f}  {tag}")

    print("\nVERDICT")
    print(f"  clusters flip sign with N : {clusters_flip} "
          f"(N=100 {c100:+.4f}, N=400 {c400:+.4f})")
    if not clusters_flip:
        print("  => The flip did NOT reproduce in this configuration.  The main "
              "grid's 6/6 split may depend on noise or dropout, which stage C1 "
              "excludes.  Re-run at C3 before pursuing a mechanism.")
    elif matches:
        print(f"  => CANDIDATE MEDIATORS: {matches}")
        print("     Matching the sign pattern is necessary and NOT sufficient. "
              "These are candidates to be CONFIRMED BY INTERVENTION (hold the "
              "mediator fixed, vary R), which this probe does not do.  The "
              "write-up must not call any of them a mechanism yet.")
    else:
        print("  => NO mediator flips the way clusters do.  The effect is not "
              "carried by reach, agreement yield, or reachable opinion "
              "diversity as measured here; the mechanism is elsewhere and the "
              "flip stays unexplained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
