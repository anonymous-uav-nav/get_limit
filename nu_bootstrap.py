# -*- coding: utf-8 -*-
"""Bootstrap the uncertainty of the collapse exponent nu.

WHY THIS EXISTS
---------------
The manuscript reports nu = 2.60 with a sharp collapse minimum but no
uncertainty on nu.  A referee will ask for one.  The honest instrument is a
bootstrap over the independent realisations: replay every (N, eps) cell's
per-seed largest-cluster fractions -- the RNG streams are deterministic
(1_000_000 + 7919*s + n), the same streams scaling_probe.py consumed -- so
the replayed means match scaling_raw.csv bit for bit, and the resampling can
be checked against the published numbers before it is trusted.

WHAT IT DOES
------------
For each of B bootstrap rounds:
  1. resample the 60 seeds within every cell (ONE fresh draw per cell per
     round; the resampled mean AND the resampled chi both come from that
     single draw, so the two observables stay coupled as they are in the
     real data);
  2. re-locate eps_c per N (the chi peak) from the resampled cells;
  3. rescan nu over the registered grid and record the argmin.
The reported interval is the 2.5/97.5 percentile range of nu*, plus the
fraction of rounds whose minimum is sharp by the registered criterion
(min < 0.85 * median).

Output: nu_bootstrap.csv  (per-round nu*, plus a summary block).
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

N_LIST = (50, 100, 200, 400, 800)
N_SEEDS = 60
EPS_GRID = np.round(np.arange(0.15, 0.351, 0.01), 4)
NU_GRID = tuple(round(v, 2) for v in np.arange(0.4, 3.01, 0.1))
N_BOOT = 400
SHARP_FRACTION_THRESHOLD = 0.85

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nu_bootstrap.csv")
RNG = np.random.default_rng(20261121)


def replay_cell(n: int, eps: float) -> np.ndarray:
    """Per-seed largest-cluster fractions for one cell, same streams as
    scaling_probe.py."""
    S = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        rng = np.random.default_rng(1_000_000 + 7919 * s + n)
        x0 = rng.uniform(DOMAIN[0], DOMAIN[1], size=n)
        x, _, _ = hk_converge(x0, float(eps))
        S[s] = cluster_sizes(x)[0] / n
    return S


def collapse_quality(cells: dict[tuple[int, float], float],
                     eps_c: dict[int, float], nu: float) -> float:
    """Same running-median scatter as scaling_probe.collapse_quality, on
    resampled means."""
    xs, ys = [], []
    for (n, eps), s_mean in cells.items():
        if n not in eps_c:
            continue
        xs.append((eps - eps_c[n]) * n ** (1.0 / nu))
        ys.append(s_mean)
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
    t0 = time.time()
    cells_S: dict[tuple[int, float], np.ndarray] = {}
    replay_mean: dict[tuple[int, float], float] = {}
    for n in N_LIST:
        for eps in EPS_GRID:
            key = (n, float(eps))
            cells_S[key] = replay_cell(n, float(eps))
            replay_mean[key] = float(cells_S[key].mean())
        print(f"  replayed N={n:>4}  [{time.time() - t0:.0f}s]")

    # --- fidelity gate: replayed means must match scaling_raw.csv ----------
    raw = {}
    with open(os.path.join(HERE, "scaling_raw.csv"), newline="",
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            raw[(int(float(r["n"])), round(float(r["eps"]), 4))] = \
                float(r["mean_S"])
    worst = 0.0
    for key, m in replay_mean.items():
        worst = max(worst, abs(m - raw[key]))
    print(f"  fidelity vs scaling_raw.csv: max |delta mean_S| = {worst:.2e}")
    if worst > 5e-7:
        print("  FAIL: replay does not reproduce the published means; "
              "refusing to bootstrap.")
        return 1

    keys = list(cells_S.keys())
    ns = np.array([k[0] for k in keys])
    epss = np.array([k[1] for k in keys])

    nus_round = np.empty(N_BOOT)
    sharp_count = 0
    for b in range(N_BOOT):
        # ONE resample per cell per round; mean and chi both from that draw
        means = np.empty(len(keys))
        chis = np.empty(len(keys))
        for i, key in enumerate(keys):
            S = cells_S[key]
            idx = RNG.integers(0, N_SEEDS, size=N_SEEDS)
            s = S[idx]
            means[i] = s.mean()
            n = key[0]
            chis[i] = n * ((s * s).mean() - s.mean() ** 2)
        eps_c: dict[int, float] = {}
        for n in N_LIST:
            mask = ns == n
            eps_c[n] = float(epss[mask][np.argmax(chis[mask])])
        scan = [(nu, collapse_quality(dict(zip(keys, means)), eps_c, nu))
                for nu in NU_GRID]
        best_nu, _ = min(scan, key=lambda t: t[1])
        qs = np.array([q for _, q in scan])
        if qs.min() < SHARP_FRACTION_THRESHOLD * np.median(qs):
            sharp_count += 1
        nus_round[b] = best_nu
        if (b + 1) % 50 == 0:
            print(f"  boot {b + 1}/{N_BOOT}  nu* median so far "
                  f"{np.median(nus_round[:b + 1]):.2f}  "
                  f"[{time.time() - t0:.0f}s]")

    lo, hi = np.quantile(nus_round, [0.025, 0.975])
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "nu_star"])
        for b, v in enumerate(nus_round):
            w.writerow([b, round(float(v), 3)])
        w.writerow([])
        w.writerow(["summary"])
        w.writerow(["n_boot", N_BOOT])
        w.writerow(["nu_star_median", round(float(np.median(nus_round)), 3)])
        w.writerow(["nu_star_ci_lo", round(float(lo), 3)])
        w.writerow(["nu_star_ci_hi", round(float(hi), 3)])
        w.writerow(["sharp_fraction", round(sharp_count / N_BOOT, 4)])
    print(f"\nnu* = {np.median(nus_round):.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"sharp-minimum fraction: {sharp_count / N_BOOT:.2f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
