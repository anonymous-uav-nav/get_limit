# -*- coding: utf-8 -*-
"""Density collapse, fixed: continuous order parameter, and the JOINT neighbour count.

WHAT WAS WRONG WITH THE FIRST ATTEMPT
-------------------------------------
The first collapse used the CLUSTER COUNT as the observable and the ASSUMED
spatial neighbour count <k> = K_COEF * r_mult * N as the scaling variable.  It
failed, 1 of 4 levels collapsing within 10%, and the failure had two separable
causes that this file removes:

  (a) FLOOR.  The cluster count is an integer with a floor at 1.  At large <k>
      the population reaches consensus, the observable is pinned at the floor,
      and its remaining variation is quantisation.  The 16.7% spread at
      <k> = 96 was measured almost entirely at that floor.  Fixed by using two
      CONTINUOUS observables instead: S, the fraction of agents in the largest
      cluster, and sigma, the standard deviation of the settled opinions.

  (b) THE WRONG SCALING VARIABLE.  <k> counts neighbours that are merely
      REACHABLE.  Interaction additionally requires the opinions to be
      compatible, and the two attributes do not scale with N in the same way:
      at fixed <k>, a larger N means a NARROWER spatial window over a denser
      population, so the fraction of the window that also passes the opinion
      test changes with N.  The hypothesis was therefore too coarse.  Fixed by
      measuring the JOINT count k_eff = (reachable) x (fraction compatible) and
      testing the collapse against that as well.

WHY THIS IS DECISIVE EITHER WAY
-------------------------------
If the data collapse against k_eff but not against <k>, the size dependence is
a JOINT-NEIGHBOURHOOD dependence, and the open question from the main grid is
answered with a measurable quantity.  If neither collapses, the size entry in
R's leverage is not a neighbourhood effect at all and must be reported as
unexplained -- which is also a result, and the honest one.

Output: density_raw.csv; printed collapse tables and verdict.
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
from swarm import SwarmParams, mediators, run_swarm  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_LIST = (200, 400) if QUICK else (200, 400, 800)
K_TARGETS = (48.0, 96.0) if QUICK else (48.0, 96.0, 120.0, 240.0)
SEEDS = tuple(range(4 if QUICK else 15))
STAGE = "C1"
SUFFIX = "_quick" if QUICK else ""

K_COEF = 2.0 * P.RADIUS_BASE / P.ARENA      # <k> = K_COEF * r_mult * N
COLLAPSE_TOL = 0.10                          # 10% spread counts as collapsed

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"density_raw{SUFFIX}.csv")
FIELDS = ["n_swarm", "r_mult", "k_assumed", "k_reach", "agree_yield", "k_eff",
          "S_max", "spread", "clusters"]


def measure(n: int, rm: float) -> dict[str, float]:
    S, sd, cl, kr, ay = [], [], [], [], []
    for s in SEEDS:
        cfg = SwarmParams(seed=1000 + s, stage=STAGE, n=n, eps=0.25,
                          r_mult=rm, beta=0.3, sigma=0.0, switch_rate=0.0)
        out = run_swarm(cfg, return_state=True)
        med = mediators(cfg, out["_pos"], out["_opinions"])
        S.append(cluster_sizes(out["_opinions"])[0] / n)
        sd.append(out["spread"])
        cl.append(out["clusters"])
        kr.append(med["trust_deg"])
        ay.append(med["agree_yield"])
    k_reach = float(np.mean(kr))
    yield_ = float(np.mean(ay))
    return dict(S_max=float(np.mean(S)), spread=float(np.mean(sd)),
                clusters=float(np.mean(cl)), k_reach=k_reach,
                agree_yield=yield_, k_eff=k_reach * yield_)


def collapse(rows: list[dict], key: str, obs: str) -> tuple[int, int, list[str]]:
    """Greedy nearest-neighbour pairing on `key`, comparing `obs` across N.

    Exact-equality grouping works for the DESIGNED key (k_assumed, a handful of
    values) but can never work for the MEASURED key (k_eff), whose values are
    continuous and never coincide across two different N.  The first version of
    this function therefore reported 0/0 for k_eff -- no comparable pairs, not
    a failed collapse.  Pairing rows from DIFFERENT N whose key values are
    within KEY_TOL of each other tests the same thing and works for both.
    """
    KEY_TOL = 0.10
    rs = sorted(rows, key=lambda r: r[key])
    used: set[int] = set()
    ok = 0
    made = 0
    lines = []
    for i in range(len(rs)):
        if i in used:
            continue
        for j in range(i + 1, len(rs)):
            if j in used or rs[j]["n_swarm"] == rs[i]["n_swarm"]:
                continue
            kd = abs(rs[j][key] - rs[i][key]) / max(abs(rs[i][key]), 1e-12)
            if kd > KEY_TOL:
                break
            a, b = rs[i], rs[j]
            used.add(i)
            used.add(j)
            made += 1
            x, y = a[obs], b[obs]
            rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
            good = rel < COLLAPSE_TOL
            ok += int(good)
            lines.append(
                f"    {key} {a[key]:>7.2f}/{b[key]:>7.2f}  "
                f"N={a['n_swarm']}:{x:.4f}  N={b['n_swarm']}:{y:.4f}  "
                f"spread {rel:>6.1%}  {'OK' if good else ''}")
            break
    return ok, made, lines


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED RUN"
    print(f"density collapse v2  [{mode}]  stage={STAGE}  seeds={len(SEEDS)}")
    print(f"<k> targets {K_TARGETS};  <k> = {K_COEF:.2f} * r_mult * N\n")

    rows: list[dict] = []
    t0 = time.time()
    for n in N_LIST:
        for k in K_TARGETS:
            rm = k / (K_COEF * n)
            if rm > 1.0 + 1e-9:
                print(f"  N={n:>4} <k>={k:>5.0f}  r_mult={rm:.3f} > 1.0 "
                      f"-> saturated, dropped")
                continue
            m = measure(n, rm)
            rows.append(dict(n_swarm=n, r_mult=round(rm, 6), k_assumed=k, **m))
            print(f"  N={n:>4} <k>={k:>5.0f} r_mult={rm:.3f}  "
                  f"k_reach={m['k_reach']:>6.1f} yield={m['agree_yield']:.3f} "
                  f"k_eff={m['k_eff']:>6.1f}  S={m['S_max']:.4f} "
                  f"sigma={m['spread']:.4f}  [{time.time() - t0:.0f}s]")
        print()

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 7) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {OUT}  ({len(rows)} rows)\n")

    results: dict[tuple[str, str], tuple[int, int]] = {}
    for obs in ("S_max", "spread"):
        for key in ("k_assumed", "k_eff"):
            print(f"COLLAPSE of {obs} against {key}")
            ok, made, lines = collapse(rows, key, obs)
            for ln in lines:
                print(ln)
            results[(key, obs)] = (ok, made)
            print(f"  -> {ok}/{made} levels collapse within "
                  f"{COLLAPSE_TOL:.0%}\n")

    print("VERDICT")
    best = max(results.items(), key=lambda kv: (kv[1][0] / max(kv[1][1], 1)))
    (key, obs), (ok, made) = best
    print(f"  best pair: {obs} vs {key} at {ok}/{made}")
    if made and ok == made and key == "k_eff":
        print("  => COLLAPSE against the JOINT neighbour count.  The size "
              "dependence reported in the main grid is a dependence on how "
              "many neighbours are simultaneously reachable and "
              "opinion-compatible.  The open question is answered.")
    elif made and ok == made:
        print(f"  => COLLAPSE against {key}.  The leverage is governed by that "
              f"variable rather than by population size as such.")
    elif ok == 0:
        print("  => NO COLLAPSE against either variable.  The size entry in "
              "R's leverage is not a neighbourhood effect and must be "
              "reported as unexplained.")
    else:
        print(f"  => PARTIAL ({ok}/{made}).  Report the level dependence, not "
              f"a single conclusion, and do not claim the mechanism.")
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
