# -*- coding: utf-8 -*-
"""Idealised bounded-confidence models, for instrument validation.

WHY THIS FILE EXISTS BEFORE ANY UAV CODE
----------------------------------------
The audit's decisive validation is two-step: first REPRODUCE the published
phase diagrams under the exact conditions the sociophysics literature used
(static complete graph, no noise, no motion), and only then open the operating
conditions and measure how the control collapses.  If the reproduction fails,
the instrument is wrong and everything built on it is worthless, so the
reproduction is the first thing built and the first thing that can kill the
project.

Targets it must hit, taken from the literature:
  * [Nonmonotonic consensus transitions in bounded-confidence dynamics,
    arXiv 2507.14276] "The threshold for consensus eps_c depends on the number
    of agents... saturating at eps_c = 0.2 in the large N limit... in the
    fully-connected case with N = 100 consensus appears around eps_c = 0.25."
  * [Critical behavior of the Hegselmann-Krause model with smart agents,
    arXiv 2102.02385] finite-size scaling P(consensus) = f((eps - eps_c) N^a).

Reproducing those two numbers is the kill line for this file.

MODELS
------
HK (Hegselmann-Krause, synchronous, complete graph):
    N_i(t) = { j : |x_i(t) - x_j(t)| <= eps }   (includes i)
    x_i(t+1) = mean of x_j(t) over j in N_i(t)
Converged HK clusters are sets of EXACTLY equal opinions, so cluster counting
is a gap split rather than a distance-based clustering -- no algorithm choice
left to argue about.

Domains and conventions
-----------------------
Opinions start uniform on [0, 1]; the published eps_c = 0.2 is quoted for that
domain.  A cluster split threshold of 0.01 follows the convention of
arXiv 2102.02385, which used 0.01 on a [-1, 1] domain -- the same 1%-of-range
rule.  Both are recorded here so a referee can check the comparison was
apples-to-apples rather than assume it.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DOMAIN = (0.0, 1.0)
CLUSTER_TOL = 0.01          # 1% of range, matches arXiv 2102.02385 convention
CLUSTER_TOL_STRICT = 1e-6   # for converged runs, where gaps are exact
CONV_TOL = 1e-12
MAX_STEPS = 4000


def component_fixed_point(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Exact HK fixed point on a FROZEN interaction graph.

    Averaging is NOT sum-conserving on a component: the transition matrix is
    P = D^-1 m, which is row-stochastic but doubly stochastic only when all
    degrees are equal.  The fixed point is therefore the DEGREE-WEIGHTED mean
    over each connected component -- the stationary distribution of P is
    proportional to the degrees, because m is symmetric and P is reversible
    with respect to them.  Taking the plain mean here would be a second, more
    silent version of the bug this function exists to fix.
    """
    deg = m.sum(axis=1).astype(float)
    n_comp, labels = connected_components(csr_matrix(m), directed=False)
    out = np.empty_like(x)
    for c in range(n_comp):
        sel = labels == c
        w = deg[sel]
        out[sel] = float((w * x[sel]).sum() / w.sum())
    return out


def hk_step(x: np.ndarray, eps: float) -> np.ndarray:
    """One synchronous Hegselmann-Krause update on the complete graph."""
    d = np.abs(x[:, None] - x[None, :])
    m = d <= eps
    return (m @ x) / m.sum(axis=1)


def hk_converge(x0: np.ndarray, eps: float,
                max_steps: int = MAX_STEPS) -> tuple[np.ndarray, bool, int]:
    """Iterate HK to convergence.  Returns (final opinions, converged, steps).

    THE EARLIER VERSION OF THIS FUNCTION WAS WRONG, and the reproduction check
    caught it.  It stopped as soon as the ACTIVE SET stopped changing, on the
    reasoning that a frozen neighbourhood means there is nothing left to do.
    That is false: a frozen set only fixes each agent's neighbour list, and
    opinions inside a component keep moving towards that component's fixed
    point.  The early exit therefore returned MID-CONVERGENCE, which inflates
    the cluster count and pushes eps_c up -- and it bites hardest at large N,
    which is exactly where the published large-N limit lives.  The whole
    reproduction read 0.26 instead of trending towards 0.2 because of this.

    A frozen active set is now used as an ACCELERATOR, never as a stopping
    rule: jump straight to the exact fixed point of the frozen graph, then
    verify that the jump did not itself change the interaction graph.  Only
    when it did not is the result a true fixed point of the HK dynamics.
    """
    x = x0.astype(float).copy()
    prev_m: np.ndarray | None = None
    for k in range(max_steps):
        d = np.abs(x[:, None] - x[None, :])
        m = d <= eps
        if prev_m is not None and np.array_equal(m, prev_m):
            x_fp = component_fixed_point(x, m)
            d_fp = np.abs(x_fp[:, None] - x_fp[None, :])
            if np.array_equal(d_fp <= eps, m):
                return x_fp, True, k + 1
            x = x_fp
            prev_m = None
            continue
        prev_m = m
        x_new = (m @ x) / m.sum(axis=1)
        if np.max(np.abs(x_new - x)) < CONV_TOL:
            return x_new, True, k + 1
        x = x_new
    return x, False, max_steps


def count_clusters(x: np.ndarray, tol: float = CLUSTER_TOL) -> int:
    """Number of opinion clusters, by splitting sorted opinions on gaps > tol."""
    xs = np.sort(np.asarray(x, dtype=float))
    if xs.size == 0:
        return 0
    return 1 + int(np.sum(np.diff(xs) > tol))


def cluster_sizes(x: np.ndarray, tol: float = CLUSTER_TOL) -> np.ndarray:
    """Sizes of the opinion clusters, largest first.

    Same gap-split rule as count_clusters, so the two cannot disagree.  Added
    for the scaling analysis, where the ORDER PARAMETER is the relative size of
    the largest cluster and the SUSCEPTIBILITY is its fluctuation -- neither of
    which the cluster COUNT can supply.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    if xs.size == 0:
        return np.zeros(0, dtype=int)
    breaks = np.nonzero(np.diff(xs) > tol)[0] + 1
    return np.sort(np.diff(np.concatenate(([0], breaks, [xs.size]))))[::-1]


def hk_run(n: int, eps: float, seed: int, **kw) -> dict[str, float]:
    """One episode: uniform initial opinions on DOMAIN, HK to convergence."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(DOMAIN[0], DOMAIN[1], size=n)
    x, conv, steps = hk_converge(x0, eps, **kw)
    return dict(n_clusters=float(count_clusters(x)),
                n_clusters_strict=float(count_clusters(x, CLUSTER_TOL_STRICT)),
                converged=float(conv), steps=float(steps))


def critical_eps(eps_grid: np.ndarray, consensus_frac: np.ndarray) -> float:
    """First eps where P(consensus) crosses 0.5, linearly interpolated.

    Returns nan when the curve never crosses, which is itself a reportable
    outcome: it means the sweep range does not bracket the transition.
    """
    above = consensus_frac >= 0.5
    if not above.any():
        return float("nan")
    i = int(np.argmax(above))
    if i == 0:
        return float(eps_grid[0])
    x0, x1 = eps_grid[i - 1], eps_grid[i]
    y0, y1 = consensus_frac[i - 1], consensus_frac[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))
