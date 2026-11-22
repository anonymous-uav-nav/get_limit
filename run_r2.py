# -*- coding: utf-8 -*-
"""SECOND ROUND + INTERVENTION A1: does opinion dispersion carry the flip?

Registered in prereg_r2.py, frozen before this run.  Two changes from the first
round and one new arm:

    R grid      (0.2, 0.4, 0.6, 0.8, 1.0), all NON-SATURATING, because the
                plant clips trust_r at the sensing radius so r_mult = 1.0 and
                2.0 were the same condition and the "three-level grid" was
                really one contrast (prereg_r2 defect D1).
    criterion   the SIGNED SLOPE of cluster count against R, with the claim
                being about the sign PATTERN, not about positivity (defect D3).
                S3 -- the difference between sizes -- is the registered claim;
                S1 and S2 alone cannot establish a flip.
    arm A1      pin the in-window opinion dispersion at the N = 100 level
                while R sweeps at N = 400.

    normaliser  everything scales by the grid MEAN, never by an endpoint that
                can be zero (defect D2).

WHAT A1 DECIDES
---------------
    slope at N = 400 turns POSITIVE under A1  -> dispersion carries the flip
    slope at N = 400 stays NEGATIVE under A1  -> candidate WITHDRAWN, reported
                                                 as withdrawn

THE CAVEAT THIS ARM CANNOT ESCAPE
---------------------------------
Rescaling opinions rescales what the bounded-confidence test compares, so the
intervention is entangled with the agreement mechanism by construction -- the
agreement test IS a dispersion test.  If the arm cannot separate them, the
honest report is "the mediator is not separable from the mechanism", which is
a result, not a failed run.

Output: r2_grid.csv, r2_slopes.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import prereg_r2 as R2  # noqa: E402
from swarm import SwarmParams, mediators, run_swarm  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_LIST = (100, 400)
R_LIST = R2.R_GRID_V2
SEEDS = tuple(range(3 if QUICK else 20))
N_STEPS = 60 if QUICK else 400
STAGE = "C1"
SUFFIX = "_quick" if QUICK else ""

# The A1 target: the in-window opinion dispersion measured at N = 100 in
# mechanism_probe.py at this stage and eps (0.0706 / 0.0570 / 0.0570 across the
# old R grid; mean 0.0615).  Recorded here rather than recomputed at runtime,
# so the target cannot drift with the run.
A1_TARGET = 0.0615

ARMS: dict[str, float | None] = {"control": None, "A1": A1_TARGET}

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, f"r2_grid{SUFFIX}.csv")
SLOPES = os.path.join(HERE, f"r2_slopes{SUFFIX}.csv")
GRID_FIELDS = ["arm", "n_swarm", "r_mult", "seed", "clusters",
               "spread", "inwin_op_std"]
SLOPE_FIELDS = ["arm", "n_swarm", "slope", "ci_lo", "ci_hi", "verdict"]
RNG = np.random.default_rng(20261119)


def collect() -> list[dict]:
    rows: list[dict] = []
    t0 = time.time()
    total = len(ARMS) * len(N_LIST) * len(R_LIST) * len(SEEDS)
    n = 0
    for arm, pin in ARMS.items():
        for nsw in N_LIST:
            for rm in R_LIST:
                for s in SEEDS:
                    sp = SwarmParams(seed=1000 + s, stage=STAGE, n=nsw,
                                     eps=0.25, r_mult=rm, beta=0.3, sigma=0.0,
                                     switch_rate=0.0, n_steps=N_STEPS,
                                     pin_inwin_std=pin)
                    out = run_swarm(sp, return_state=True)
                    med = mediators(sp, out["_pos"], out["_opinions"])
                    rows.append(dict(arm=arm, n_swarm=nsw, r_mult=rm, seed=s,
                                     clusters=out["clusters"],
                                     spread=round(out["spread"], 6),
                                     inwin_op_std=round(med["inwin_op_std"], 6)))
                    n += 1
            print(f"  {arm:>7} N={nsw:>4} done  "
                  f"[{n}/{total}, {time.time() - t0:.0f}s]")
    return rows


def slope_for(rows: list[dict], arm: str, nsw: int) -> tuple[float, float, float]:
    """Signed slope of clusters against R, bootstrapped over seeds.

    The first version of this function drew ONE bootstrap index vector outside
    the resampling loop and reused it for every iteration, so all iterations
    were identical and every interval collapsed to a point -- with the point
    estimate sometimes falling OUTSIDE its own degenerate interval.  A fresh
    draw per iteration is the whole point of a bootstrap; the bug is recorded
    here because the intervals it produced looked plausible at a glance.
    """
    cells: dict[float, list[float]] = {}
    for r in rows:
        if r["arm"] != arm or r["n_swarm"] != nsw:
            continue
        cells.setdefault(float(r["r_mult"]), []).append(float(r["clusters"]))
    xs = np.array(sorted(cells))
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan")
    arr = {x: np.array(cells[x]) for x in xs}
    n_seeds = len(next(iter(arr.values())))
    xs_c = xs - xs.mean()
    denom = float(xs_c @ xs_c)
    slope = float(xs_c @ np.array([arr[x].mean() for x in xs])) / denom

    boots = np.empty(R2.N_BOOT_V2)
    for i in range(R2.N_BOOT_V2):
        yb = np.empty(xs.size)
        for j, x in enumerate(xs):
            idx = RNG.integers(0, n_seeds, size=n_seeds)   # fresh draw
            yb[j] = arr[x][idx].mean()
        boots[i] = float(xs_c @ (yb - yb.mean())) / denom
    lo, hi = np.quantile(boots, [(1 - R2.SLOPE_CI_LEVEL) / 2,
                                 1 - (1 - R2.SLOPE_CI_LEVEL) / 2])
    return slope, float(lo), float(hi)


def main() -> int:
    print(f"{R2.VERSION}  second round + A1  "
          f"{'[QUICK]' if QUICK else '[REGISTERED]'}")
    print(f"R grid={R_LIST} (non-saturating)  A1 target={A1_TARGET}  "
          f"arms={list(ARMS)}\n")

    rows = collect()
    with open(GRID, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GRID_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {GRID}  ({len(rows)} rows)")

    # the A1 arm must actually have pinned the dispersion, or it is not the arm
    print("\nA1 PIN CHECK (target %.4f)" % A1_TARGET)
    for arm, pin in ARMS.items():
        for nsw in N_LIST:
            v = [r["inwin_op_std"] for r in rows
                 if r["arm"] == arm and r["n_swarm"] == nsw]
            print(f"  {arm:>7} N={nsw:>4}  mean inwin_op_std={np.mean(v):.4f}")
    a1_400 = [r["inwin_op_std"] for r in rows
              if r["arm"] == "A1" and r["n_swarm"] == 400]
    pinned = bool(a1_400) and abs(float(np.mean(a1_400)) - A1_TARGET) \
        <= R2.INTERVENTION_TOL
    print(f"  A1 pinned within tol={R2.INTERVENTION_TOL}: {pinned}")

    ctrl400 = slope_for(rows, "control", 400)
    ctrl100 = slope_for(rows, "control", 100)
    flip_present = bool(np.isfinite(ctrl400[0])) and ctrl400[1] > 0

    out = []
    print(f"\n{'arm':>8} {'N':>5} {'slope':>10} {'CI':>22}  verdict")
    for arm in ARMS:
        for nsw in N_LIST:
            s, lo, hi = slope_for(rows, arm, nsw)
            # SIGN CONVENTION, spelled out because the first draft of this
            # logic had it backwards and printed "flip NOT reproduced" on data
            # that reproduced it.  control = (E[clusters|R_min] -
            # E[clusters|R_max]) / E[clusters|R_min], so:
            #   first round N = 100: control +0.0581  ->  slope NEGATIVE
            #   first round N = 400: control -0.0246  ->  slope POSITIVE
            if not np.isfinite(s):
                v = "no data"
            elif arm == "control" and nsw == 100:
                v = ("positive control reproduced (slope < 0)" if hi < 0
                     else "flip NOT reproduced")
            elif arm == "control" and nsw == 400:
                v = ("negative control reproduced (slope > 0)" if lo > 0
                     else "FLIP ABSENT: slope is not positive on the "
                          "non-saturating grid")
            elif arm == "A1" and nsw == 400:
                if not flip_present:
                    v = ("UNINTERPRETABLE: A1 only tests the mediator if the "
                         "control arm shows the flip first, and it does not")
                elif hi < 0:
                    v = "slope turns NEGATIVE -> dispersion carries the flip"
                else:
                    v = "slope stays POSITIVE -> CANDIDATE WITHDRAWN"
            else:
                v = "reference"
            out.append(dict(arm=arm, n_swarm=nsw, slope=round(s, 6),
                            ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                            verdict=v))
            print(f"{arm:>8} {nsw:>5} {s:>+10.5f} "
                  f"[{lo:>+8.5f},{hi:>+8.5f}]  {v}")

    with open(SLOPES, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SLOPE_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {SLOPES}")

    print("\nCAVEAT that cannot be engineered away: rescaling opinions rescales "
          "what the\nagreement test compares, so A1 is entangled with the "
          "agreement mechanism by\nconstruction.  If the two arms differ only "
          "slightly, the registered sentence is\n'the mediator is not separable "
          "from the mechanism', NOT 'the mechanism is dispersion'.")
    if not pinned:
        print("\n  WARNING: the A1 arm did not land on its pin target, so it "
              "is not a\n  valid intervention arm and its slope may not be "
              "interpreted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
