# -*- coding: utf-8 -*-
"""Does the trust radius's leverage depend on DENSITY rather than on SIZE?

THE OPEN QUESTION
-----------------
The main grid found the trust radius R to have a clear leverage at N = 100
(slope -0.975, interval excluding zero) and none detectable at N = 400
(-0.050, interval spanning zero).  It reported that without explaining it, and
the registered intervention failed because its precondition did not hold.

THE HYPOTHESIS TESTED HERE
--------------------------
In an embedding of length L, an agent's expected number of reachable
neighbours is
    <k> = 2 R rho,      rho = N / L,
so N and R enter only through their product.  A density hypothesis says the
observable depends on <k> alone: change N and compensate with R, and nothing
moves.  A size hypothesis says N matters on its own and no such compensation
exists.

This is decisive rather than descriptive.  If the data collapse onto <k>, the
size dependence IS a density dependence and the open question is answered --
the parameter's leverage is governed by how many neighbours an agent can
reach, and "depends on swarm size" is a restatement of "changes the density".
If they do not collapse, the size entry in R's leverage is genuinely about N
and the density account is refuted.

DESIGN
------
For a fixed set of target neighbour counts <k> and each N in {100, 200, 400},
the radius multiplier is solved from <k> = K_COEF * r_mult * N, where K_COEF =
2 * RADIUS_BASE / ARENA.  Arms whose r_mult exceeds 1.0 are dropped: above that
the reachable set is saturated, the effective <k> stops growing, and the arm
would not test the hypothesis.

The measured neighbour count is recorded alongside the expected one, so the
geometry is checked rather than assumed.

Output: neighbour_raw.csv; printed collapse table and verdict.
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
N_LIST = (100, 200) if QUICK else (100, 200, 400)
K_TARGETS = (12.0, 48.0) if QUICK else (12.0, 24.0, 48.0, 96.0)
SEEDS = tuple(range(5 if QUICK else 20))
STAGE = "C1"                      # graph dynamics on, no noise, no failure
SUFFIX = "_quick" if QUICK else ""

# <k> = 2 * RADIUS_BASE * r_mult * N / ARENA
K_COEF = 2.0 * P.RADIUS_BASE / P.ARENA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"neighbour_raw{SUFFIX}.csv")
FIELDS = ["n_swarm", "r_mult", "k_expected", "k_measured", "clusters",
          "agree_yield", "spread"]


def measure(n: int, rm: float) -> dict[str, float]:
    cl, km, ay, sp = [], [], [], []
    for s in SEEDS:
        cfg = SwarmParams(seed=1000 + s, stage=STAGE, n=n, eps=0.25,
                          r_mult=rm, beta=0.3, sigma=0.0, switch_rate=0.0)
        out = run_swarm(cfg, return_state=True)
        med = mediators(cfg, out["_pos"], out["_opinions"])
        cl.append(out["clusters"])
        km.append(med["trust_deg"])
        ay.append(med["agree_yield"])
        sp.append(out["spread"])
    return dict(clusters=float(np.mean(cl)), k_measured=float(np.mean(km)),
                agree_yield=float(np.mean(ay)), spread=float(np.mean(sp)))


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED RUN"
    print(f"neighbour-count collapse  [{mode}]  stage={STAGE}  "
          f"seeds={len(SEEDS)}")
    print(f"<k> targets {K_TARGETS};  <k> = {K_COEF:.2f} * r_mult * N\n")

    rows: list[dict] = []
    t0 = time.time()
    for n in N_LIST:
        for k in K_TARGETS:
            rm = k / (K_COEF * n)
            if rm > 1.0 + 1e-9:
                print(f"  N={n:>4} <k>={k:>5.0f}  r_mult={rm:.3f} > 1.0 "
                      f"-> saturated, arm dropped")
                continue
            m = measure(n, rm)
            rows.append(dict(n_swarm=n, r_mult=round(rm, 5), k_expected=k, **m))
            print(f"  N={n:>4} <k>={k:>5.0f}  r_mult={rm:.3f}  "
                  f"k_meas={m['k_measured']:.1f}  S={m['clusters']:.3f}  "
                  f"yield={m['agree_yield']:.3f}  [{time.time() - t0:.0f}s]")
        print()

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {OUT}  ({len(rows)} rows)\n")

    print("COLLAPSE TABLE  (rows = neighbours an agent can reach)")
    print(f"{'<k>':>6} " + " ".join(f"{'N=' + str(n):>12}" for n in N_LIST)
          + f" {'spread':>9}")
    ok_rows = 0
    collapsed = 0
    for k in K_TARGETS:
        vals = {r["n_swarm"]: r["clusters"] for r in rows
                if r["k_expected"] == k}
        got = [vals[n] for n in N_LIST if n in vals]
        if len(got) < 2:
            continue
        ok_rows += 1
        rel = (max(got) - min(got)) / max(max(got), 1e-9)
        if rel < 0.10:
            collapsed += 1
        print(f"{k:>6.0f} " + " ".join(
            f"{vals[n]:>12.3f}" if n in vals else f"{'--':>12}" for n in N_LIST)
            + f" {rel:>9.1%}")

    print("\nVERDICT")
    if ok_rows == 0:
        print("  no comparable arms; nothing was tested")
        return 1
    print(f"  comparable <k> levels: {ok_rows};  collapsed within 10%: "
          f"{collapsed}")
    if collapsed == ok_rows:
        print("  => COLLAPSE.  The trust radius's leverage is governed by the "
              "number of neighbours an agent can reach, not by population size "
              "as such.  The size dependence reported in the main grid is a "
              "density dependence, and the open question is answered.")
    elif collapsed == 0:
        print("  => NO COLLAPSE.  R and N do not enter only through <k>; the "
              "size entry is real and the density account is refuted.  Report "
              "the size dependence as unexplained.")
    else:
        print(f"  => PARTIAL ({collapsed}/{ok_rows}).  The density account "
              f"holds at some levels and not others; report the level "
              f"dependence rather than a single conclusion.")
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
