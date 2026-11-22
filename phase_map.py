# -*- coding: utf-8 -*-
"""Full phase diagrams: the transition in the (eps, R) plane, per condition.

WHY THIS IS NOT audit_grid.csv
------------------------------
The leverage grid holds R at three levels and is designed for marginalisation:
its job is to say how much a parameter moves the cluster count.  A phase
diagram has a different job, which is to show WHERE the transition lies in the
two-parameter plane and how its boundary moves as conditions are relaxed.  That
needs a real R axis, a real eps axis, and -- because the boundary is a curve,
not a number -- a fine enough grid to draw it.  Re-using the leverage grid would
produce a diagram with two R levels and no visible boundary.

QUANTITIES
----------
    S    fraction of the population in the largest cluster    order parameter
    chi  N ( <S^2> - <S>^2 )                                  susceptibility

The susceptibility is what locates the boundary: within a condition, the ridge
of chi against eps traces the transition, and comparing ridges across conditions
shows whether relaxing an idealisation moves it.

A NOTE ON THE ORDER PARAMETER
-----------------------------
S is computed from the SETTLED OPINIONS via cluster_sizes, which requires the
final state.  run_swarm returns it only when asked, so it is asked for
explicitly here.  An earlier draft of this loop assumed the state was present
and, when it was not, silently substituted 1/cluster_count -- a different
quantity that would have produced a plausible-looking diagram of the wrong
thing.  That fallback is gone: if the state is missing the run stops.

Output: phase_raw.csv; printed summary of the chi ridge per condition.
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
from sociophysics import cluster_sizes  # noqa: E402
from swarm import STAGES, SwarmParams, run_swarm  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_LIST = (100,) if QUICK else (100, 400)
EPS_GRID = np.round(np.arange(0.15, 0.351, 0.05 if QUICK else 0.01), 4)
R_GRID = (0.5, 1.0) if QUICK else (0.2, 0.4, 0.6, 0.8, 1.0)
SEEDS = tuple(range(3 if QUICK else 12))
SIGMA = 0.005
SWITCH = 0.02
SUFFIX = "_quick" if QUICK else ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"phase_raw{SUFFIX}.csv")
FIELDS = ["stage", "n_swarm", "eps", "r_mult", "S_max", "chi", "clusters",
          "spread", "n_seeds"]


def cell(stage: str, n: int, eps: float, rm: float) -> dict[str, float]:
    S, cl, sd = [], [], []
    for s in SEEDS:
        cfg = SwarmParams(seed=2000 + s, stage=stage, n=n, eps=eps,
                          r_mult=rm, beta=0.3, sigma=SIGMA, switch_rate=SWITCH)
        out = run_swarm(cfg, return_state=True)
        opinions = out["_opinions"]
        S.append(cluster_sizes(opinions)[0] / n)
        cl.append(out["clusters"])
        sd.append(out["spread"])
    S = np.array(S, dtype=float)
    if not np.isfinite(S).all():
        raise RuntimeError("non-finite order parameter; refusing to continue")
    m1 = float(S.mean())
    m2 = float((S * S).mean())
    return dict(S_max=m1, chi=n * (m2 - m1 * m1),
                clusters=float(np.mean(cl)), spread=float(np.mean(sd)))


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED RUN"
    total = len(STAGES) * len(N_LIST) * len(EPS_GRID) * len(R_GRID)
    print(f"full phase map  [{mode}]  {total} cells  seeds={len(SEEDS)}")
    print(f"eps {EPS_GRID[0]}..{EPS_GRID[-1]} ({EPS_GRID.size})  "
          f"R {R_GRID}\n")

    rows: list[dict] = []
    t0 = time.time()
    for stage in STAGES:
        for n in N_LIST:
            for eps in EPS_GRID:
                for rm in R_GRID:
                    m = cell(stage, n, float(eps), float(rm))
                    rows.append(dict(stage=stage, n_swarm=n, eps=float(eps),
                                     r_mult=float(rm), n_seeds=len(SEEDS), **m))
            sub = [r for r in rows
                   if r["stage"] == stage and r["n_swarm"] == n]
            ridge = max(sub, key=lambda r: r["chi"])
            print(f"  {stage} N={n:>4}  chi ridge at eps={ridge['eps']:.2f} "
                  f"R={ridge['r_mult']:.1f}  (chi={ridge['chi']:.2f})  "
                  f"[{len(rows)}/{total}, {time.time() - t0:.0f}s]")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 7) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    print(f"elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())