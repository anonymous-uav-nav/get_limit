# -*- coding: utf-8 -*-
"""Finite-size scaling of the fragmentation-consensus transition.

WHY THIS RUNS FIRST
-------------------
The audit so far measures the transition the way the applied literature does:
by locating where the consensus probability crosses one half.  That is a fine
descriptive statistic and a poor critical one.  Statistical mechanics locates a
transition from the ORDER PARAMETER and its FLUCTUATIONS, and asks whether
curves for different sizes collapse onto one scaling function.

This matters before any analytical work is attempted, because the collapse
answers a question the audit cannot: is what we are looking at a genuine
continuous transition with a correlation-length exponent, or a smooth crossover
that merely looks sharp at the sizes we can simulate?  If the curves collapse
with a sensible exponent, an analytical treatment has something to aim at.  If
they do not, that is itself the finding, and it changes what the theory section
should claim.

QUANTITIES
----------
For each population size N and confidence threshold eps, over many independent
realisations:

    S      = (size of the largest opinion cluster) / N     the order parameter
    chi    = N * ( <S^2> - <S>^2 )                        the susceptibility
    U      = 1 - <S^4> / (3 <S^2>^2)                       the Binder cumulant

Three independent handles on the same transition, which is the point: the
susceptibility peak, the crossing of the Binder cumulant, and the P(consensus)
crossing need not agree, and where they disagree the disagreement is
informative.  Sizes 50 to 800 overlap the range of the published results.

COLLAPSE
--------
The order parameter is assumed to obey S = f( (eps - eps_c) N^{1/nu} ).  For a
grid of nu the data are mapped to the scaling variable and the scatter about a
running median is measured; the nu minimising that scatter is reported, with
the scatter curve so that a flat minimum can be told from a sharp one.  A FLAT
minimum means no exponent is resolved and the collapse claim must not be made.

Output: scaling_raw.csv, scaling_collapse.csv; printed summary.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sociophysics import DOMAIN, cluster_sizes, hk_converge  # noqa: E402

QUICK = os.environ.get("AUDIT_QUICK") == "1"
N_LIST = (50, 100, 200) if QUICK else (50, 100, 200, 400, 800)
N_SEEDS = 8 if QUICK else 60
EPS_GRID = np.round(np.arange(0.15, 0.351, 0.02 if QUICK else 0.01), 4)
SUFFIX = "_quick" if QUICK else ""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_RAW = os.path.join(HERE, f"scaling_raw{SUFFIX}.csv")
OUT_COL = os.path.join(HERE, f"scaling_collapse{SUFFIX}.csv")

RAW_FIELDS = ["n", "eps", "n_seeds", "mean_S", "chi", "binder", "consensus_frac"]
NU_GRID = tuple(round(v, 2) for v in np.arange(0.4, 3.01, 0.1))


def measure(n: int, eps: float) -> dict[str, float]:
    """One (N, eps) cell: order parameter, susceptibility, Binder cumulant."""
    S = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        rng = np.random.default_rng(1_000_000 + 7919 * s + n)
        x0 = rng.uniform(DOMAIN[0], DOMAIN[1], size=n)
        x, _, _ = hk_converge(x0, eps)
        S[s] = cluster_sizes(x)[0] / n
    m1 = float(S.mean())
    m2 = float((S * S).mean())
    m4 = float((S ** 4).mean())
    chi = n * (m2 - m1 * m1)
    binder = 1.0 - m4 / (3.0 * m2 * m2) if m2 > 1e-12 else float("nan")
    return dict(mean_S=m1, chi=chi, binder=binder,
                consensus_frac=float(np.mean(S >= 1.0 - 1e-9)))


def collapse_quality(rows: list[dict], nu: float,
                     eps_c: dict[int, float]) -> float:
    """Scatter of S about a running median in the scaling variable.

    Lower is better.  A flat minimum across nu means no exponent is resolved.
    """
    xs, ys = [], []
    for r in rows:
        n = r["n"]
        if n not in eps_c:
            continue
        xs.append((r["eps"] - eps_c[n]) * n ** (1.0 / nu))
        ys.append(r["mean_S"])
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if x.size < 6:
        return float("inf")
    k = max(3, x.size // 6)
    resid = []
    for i in range(x.size):
        lo = max(0, i - k // 2)
        hi = min(x.size, lo + k)
        lo = max(0, hi - k)
        resid.append(abs(y[i] - np.median(y[lo:hi])))
    return float(np.median(resid))


def main() -> int:
    mode = "QUICK REHEARSAL" if QUICK else "REGISTERED RUN"
    print(f"finite-size scaling  [{mode}]  N={N_LIST}  seeds={N_SEEDS}  "
          f"eps={EPS_GRID[0]}..{EPS_GRID[-1]} ({EPS_GRID.size})")

    rows: list[dict] = []
    t0 = time.time()
    for n in N_LIST:
        for eps in EPS_GRID:
            m = measure(n, float(eps))
            rows.append(dict(n=n, eps=float(eps), n_seeds=N_SEEDS, **m))
        sub = [r for r in rows if r["n"] == n]
        peak = max(sub, key=lambda r: r["chi"])
        print(f"  N={n:>4}  chi peak at eps={peak['eps']:.2f} "
              f"(chi={peak['chi']:.3f}, S={peak['mean_S']:.3f})  "
              f"[{time.time() - t0:.0f}s]")

    with open(OUT_RAW, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nwrote {OUT_RAW}  ({len(rows)} rows)")

    eps_c: dict[int, float] = {}
    for n in N_LIST:
        sub = [r for r in rows if r["n"] == n]
        eps_c[n] = float(max(sub, key=lambda r: r["chi"])["eps"])
    print("\nsusceptibility peak, per size: "
          + "  ".join(f"N={n}:{eps_c[n]:.2f}" for n in N_LIST))
    mono = all(eps_c[N_LIST[i]] >= eps_c[N_LIST[i + 1]] - 1e-9
               for i in range(len(N_LIST) - 1))
    print(f"  peak moves to lower eps as N grows: {mono}")

    scan = [(nu, collapse_quality(rows, nu, eps_c)) for nu in NU_GRID]
    with open(OUT_COL, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nu", "quality"])
        for nu, q in scan:
            w.writerow([nu, round(q, 6)])
        w.writerow([])
        w.writerow(["eps_c_per_N"])
        for n in N_LIST:
            w.writerow([n, eps_c[n]])
    best_nu, best_q = min(scan, key=lambda t: t[1])
    qs = np.array([q for _, q in scan])
    sharp = bool(qs.min() < 0.85 * np.median(qs))
    print(f"\ncollapse scan over nu in [{NU_GRID[0]}, {NU_GRID[-1]}]")
    print(f"  best nu = {best_nu:.2f}  (scatter {best_q:.4f})")
    print(f"  minimum is {'SHARP' if sharp else 'FLAT'} "
          f"(min {qs.min():.4f} vs median {np.median(qs):.4f})")
    if not sharp:
        print(f"  => no exponent is resolved at these sizes.  The collapse "
              f"claim must NOT be made; report the transition as unresolved "
              f"within N <= {max(N_LIST)}.")
    print(f"\nwrote {OUT_COL}")
    print(f"elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
