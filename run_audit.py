# -*- coding: utf-8 -*-
"""THE REGISTERED MAIN GRID: does each knob have a lever, and can it be bounded?

What runs, after the three gate contractions recorded in prereg.GATE_RESULTS:

    axes    eps (11 levels) x r_mult (3 levels), full cross product
    stages  C0 -> C1 -> C2 -> C3, opened one condition at a time
    N       both registered swarm sizes
    seeds   prereg.N_SEEDS = 30

    beta          FIXED at 0.3   -- demoted to an endpoint effect on the
                                    protagonist, so it is not an axis here
    switch_rate   FIXED at 0.02  -- inert on the grid, so the C3 condition is
                                    PRESENT without being claimed as an axis

WHAT THE VERDICT IS ALLOWED TO SAY
----------------------------------
Two questions with different logic, because the confirmation run showed the
two knobs behave differently.

  eps   a SUPERIORITY question.  Does the lever exist (control > CONTROL_LIVE)
        and does it hold across stages?

  R     a NON-INFERIORITY question.  A lever either exists and can be BOUNDED
        away from zero, or it cannot.  The confirmation run's interval
        contained zero at every stage, so the honest statement at that power
        was "no demonstrable lever".  This run exists to tighten that bound
        with the registered 30 seeds.  If the upper bound falls below
        R_CONTROL_BOUNDABLE_CI the lever is real; if the interval still spans
        zero the correct sentence is "not demonstrable at this power", NOT
        "R is inert" -- failing to bound a lever is not the same as proving
        there is none.

CONTROL IS MARGINALISED, NEVER TAKEN AT ONE POINT
-------------------------------------------------
For a knob, the other axis is walked across its whole grid and the per-cell
controls are averaged.  The confirmation run showed the single-reference-point
version overstates r_mult's lever by about 2x, so the reference-point number is
not reported anywhere here.

Resumable: one row per (stage, n, eps, r_mult, seed), flushed as produced.

Output: audit_grid.csv (raw), audit_control.csv (per-knob verdicts), printed map.
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
N_LIST = (100,) if QUICK else P.N_SWARM
# Quick mode must SAMPLE ACROSS the eps grid, not take its first few points.
# Taking EPS_GRID[:3] puts all three below the consensus threshold (~0.23), so
# every level lands in the fragmented regime, every cluster count is identical,
# and eps reads as inert at C0 -- an artefact of the rehearsal grid that looks
# exactly like a failed experiment.  Stride sampling keeps the transition
# inside the rehearsal.
EPS_LIST = P.EPS_GRID[::4] if QUICK else P.EPS_GRID
R_LIST = P.R_GRID
SEEDS = tuple(range(3 if QUICK else P.N_SEEDS))
SUFFIX = "_quick" if QUICK else ""

BETA_FIXED = 0.3
SWITCH_FIXED = 0.02
SIGMA_FIXED = 0.005

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, f"audit_grid{SUFFIX}.csv")
CTRL = os.path.join(HERE, f"audit_control{SUFFIX}.csv")
RAW_FIELDS = ["stage", "n_swarm", "eps", "r_mult", "seed",
              "clusters", "spread", "opinion_mean"]
CTRL_FIELDS = ["stage", "n_swarm", "knob", "control", "ci_lo", "ci_hi", "verdict"]
RNG = np.random.default_rng(20261119)
BOOT = 2000


def load_done() -> set[tuple]:
    if not os.path.exists(RAW):
        return set()
    with open(RAW, newline="", encoding="utf-8") as fh:
        return {(r["stage"], int(r["n_swarm"]), float(r["eps"]),
                 float(r["r_mult"]), int(r["seed"])) for r in csv.DictReader(fh)}


def run_grid() -> None:
    done = load_done()
    total = len(STAGES) * len(N_LIST) * len(EPS_LIST) * len(R_LIST) * len(SEEDS)
    print(f"grid: {total} episodes, {len(done)} already on disk")
    first = not os.path.exists(RAW)
    fh = open(RAW, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
    if first:
        w.writeheader()
    t0, n = time.time(), 0
    for stage in STAGES:
        for nsw in N_LIST:
            for eps in EPS_LIST:
                for rm in R_LIST:
                    for s in SEEDS:
                        if (stage, nsw, float(eps), float(rm), s) in done:
                            continue
                        m = run_swarm(SwarmParams(
                            seed=1000 + s, stage=stage, n=nsw, eps=eps,
                            r_mult=rm, beta=BETA_FIXED, sigma=SIGMA_FIXED,
                            switch_rate=SWITCH_FIXED))
                        w.writerow(dict(stage=stage, n_swarm=nsw, eps=eps,
                                        r_mult=rm, seed=s,
                                        clusters=m["clusters"],
                                        spread=round(m["spread"], 6),
                                        opinion_mean=round(m["opinion_mean"], 6)))
                        n += 1
                        if n % 100 == 0:
                            fh.flush()
                            print(f"  {n} new  [{time.time() - t0:.0f}s]")
        fh.flush()
        print(f"  stage {stage} done  [{time.time() - t0:.0f}s]")
    fh.close()
    print(f"grid complete: {n} new rows\n")


def load_raw() -> list[dict]:
    with open(RAW, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cell(rows: list[dict], stage: str, nsw: int, **kw) -> np.ndarray:
    out = []
    for r in rows:
        if r["stage"] != stage or int(r["n_swarm"]) != nsw:
            continue
        if all(abs(float(r[k]) - float(v)) < 1e-9 for k, v in kw.items()):
            out.append(float(r["clusters"]))
    return np.array(out)


def control_marginalised(rows: list[dict], stage: str, nsw: int, knob: str,
                         boot_idx: np.ndarray) -> tuple[float, float, float]:
    """Average the per-other-cell control across the other axis's grid."""
    if knob == "eps":
        lo, hi, other, others = EPS_LIST[0], EPS_LIST[-1], "r_mult", R_LIST
    else:
        lo, hi, other, others = R_LIST[0], R_LIST[-1], "eps", EPS_LIST

    ctrl, boots = [], []
    for o in others:
        a = cell(rows, stage, nsw, **{other: o, knob: lo})
        b = cell(rows, stage, nsw, **{other: o, knob: hi})
        if a.size == 0 or b.size == 0:
            continue
        m = float(a.mean())
        if m < 1e-12:
            continue
        ctrl.append((m - float(b.mean())) / m)
        ma = a[boot_idx].mean(axis=1)
        mb = b[boot_idx].mean(axis=1)
        boots.append((ma - mb) / np.where(ma > 1e-12, ma, 1.0))
    if not ctrl:
        return float("nan"), float("nan"), float("nan")
    c = float(np.mean(ctrl))
    if boots:
        bm = np.mean(np.vstack(boots), axis=0)
        lo_ci, hi_ci = np.quantile(bm, [0.025, 0.975])
        return c, float(lo_ci), float(hi_ci)
    return c, float("nan"), float("nan")


def main() -> int:
    print(f"audit {P.VERSION}  main grid  "
          f"{'[QUICK REHEARSAL]' if QUICK else '[REGISTERED]'}")
    run_grid()
    rows = load_raw()
    boot_idx = RNG.integers(0, len(SEEDS), size=(BOOT, len(SEEDS)))

    out = []
    print(f"{'stage':>6} {'N':>5} {'knob':>7} {'control':>9} "
          f"{'CI':>22}  verdict")
    for stage in STAGES:
        for nsw in N_LIST:
            for knob in ("eps", "r_mult"):
                c, lo, hi = control_marginalised(rows, stage, nsw, knob, boot_idx)
                if knob == "eps":
                    if not np.isfinite(c):
                        v = "no data"
                    elif c >= P.CONTROL_LIVE and lo > 0:
                        v = "LEVER (bounded above 0)"
                    elif c >= P.CONTROL_INERT:
                        v = "weak lever"
                    else:
                        v = "inert"
                else:
                    if not np.isfinite(c):
                        v = "no data"
                    elif lo > 0 and hi < P.R_CONTROL_BOUNDABLE_CI:
                        v = "LEVER present (bounded)"
                    elif lo > 0:
                        v = "lever hinted, not bounded"
                    else:
                        v = "no demonstrable lever at this power"
                out.append(dict(stage=stage, n_swarm=nsw, knob=knob,
                                control=round(c, 4), ci_lo=round(lo, 4),
                                ci_hi=round(hi, 4), verdict=v))
                print(f"{stage:>6} {nsw:>5} {knob:>7} {c:>9.4f} "
                      f"[{lo:>+7.4f},{hi:>+7.4f}]  {v}")

    with open(CTRL, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CTRL_FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {RAW} and {CTRL}")

    eps_e = [r for r in out if r["knob"] == "eps"]
    flat = True
    for nsw in N_LIST:
        v = [r["control"] for r in eps_e if r["n_swarm"] == nsw]
        if v and (max(v) - min(v)) / max(v) > 0.25:
            flat = False
    print(f"\n  eps lever present in every stage : "
          f"{all(r['ci_lo'] > 0 for r in eps_e)}")
    print(f"  eps lever stable across stages   : {flat}")
    print("  If R's interval still spans zero, the registered sentence is "
          "'no demonstrable lever at this power' -- not 'R is inert'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
