# -*- coding: utf-8 -*-
"""Verify the algebra behind the anchoring-weight result.

THE CLAIM IN THE MANUSCRIPT, STATED PROPERLY
--------------------------------------------
The anchored update is
    o_i(k+1) = (1 - beta) * mean_{j in N_i} o_j(k) + beta * o_i(k),
i.e. in matrix form
    o(k+1) = M o(k),     M = (1 - beta) P + beta I,
where P = D^-1 A is the row-stochastic averaging matrix of the frozen
interaction graph.

PROPOSITION 1 (fixed point).  M and P have the same fixed-point set.
Proof.  o = Mo  <=>  o = (1-beta)Po + beta o  <=>  (1-beta)o = (1-beta)Po
        <=>  o = Po  for beta != 1.  The beta = 1 case, o = Io, is the frozen
        system in which no fixed point is reached from a non-stationary start.
Consequence: beta cannot change WHICH collective state is reached, only how
fast.  A cluster count is a property of the fixed point and is therefore inert
in beta; a dispersion measure is a property of the transient and is not.

PROPOSITION 2 (rate).  If P has eigenvalues 1 = L1 > |L2| >= ... then M has
eigenvalues (1-beta)L_i + beta, so the asymptotic contraction rate of the
transient is
    r(beta) = |(1 - beta) L2 + beta| ,
which is non-decreasing in beta and strictly increasing whenever L2 < 1.
Anchoring therefore SLOWS convergence by moving the sub-dominant eigenvalue
toward 1.

This script checks both propositions numerically on a frozen graph: it
predicts the contraction rate from the spectrum of P and compares it with the
rate measured from the iterated dynamics.  Agreement is the evidence; a
mismatch means the propositions as stated are wrong.

Output: printed table only (no CSV).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sociophysics import DOMAIN  # noqa: E402

N = 200
EPS = 0.25
BETAS = (0.0, 0.2, 0.4, 0.6, 0.8)
N_ITER = 150


def frozen_graph(seed: int = 7) -> np.ndarray:
    """Agreement graph of a random opinion draw at threshold EPS, held fixed."""
    rng = np.random.default_rng(seed)
    o0 = rng.uniform(DOMAIN[0], DOMAIN[1], size=N)
    d = np.abs(o0[:, None] - o0[None, :])
    A = (d <= EPS).astype(float)
    np.fill_diagonal(A, 1.0)
    return A


def main() -> int:
    A = frozen_graph()
    P = A / A.sum(axis=1, keepdims=True)
    ev = np.linalg.eigvals(P)
    order = np.argsort(-np.abs(ev))
    lam2 = ev[order[1]]
    print(f"frozen graph: N={N}  eps={EPS}  mean degree={A.sum(axis=1).mean():.1f}")
    print(f"dominant eigenvalue {ev[order[0]].real:.4f}; "
          f"sub-dominant |L2| = {abs(lam2):.6f}  (real part {lam2.real:.6f})")
    if abs(lam2) > 1.0 - 1e-9:
        print("sub-dominant eigenvalue is on the unit circle; the graph is "
              "disconnected and the rate is not defined.  Aborting.")
        return 1

    # The reference fixed point is obtained by ITERATING P, not by a
    # least-squares solve of (I - P)o = 0.  With P row-stochastic that system
    # is singular, so lstsq returns a compromise that is only APPROXIMATELY
    # stationary; a slightly wrong reference makes the measured deviation
    # flatten at ||o_limit - o*|| instead of decaying, which the first version
    # of this script duly reported as "measured rate = 1.00000".  Iterating P
    # converges at |L2| per step, so a few thousand steps land on the fixed
    # point to machine precision.
    rng = np.random.default_rng(11)
    v = rng.uniform(DOMAIN[0], DOMAIN[1], size=N)
    for _ in range(4000):
        v = P @ v
        v = v / v.sum()          # P is not doubly stochastic; fix the scale
    o_star = v.copy()

    ev_sorted = ev[np.argsort(-np.abs(ev))]
    print("top |eigenvalues|: "
          + " ".join(f"{abs(e):.4f}" for e in ev_sorted[:5])
          + "   (a second value at 1.0 would mean several stationary vectors)")
    print(f"reference residual |P o* - o*| = "
          f"{np.linalg.norm(P @ o_star - o_star):.3e}")

    print(f"\n{'beta':>6} {'predicted r':>13} {'measured r':>12} "
          f"{'rel err':>9}  fixed point same?")
    ok = True
    for beta in BETAS:
        if beta == 1.0:
            continue
        M = (1.0 - beta) * P + beta * np.eye(N)
        r_pred = abs((1.0 - beta) * lam2 + beta)

        o = rng.uniform(DOMAIN[0], DOMAIN[1], size=N)
        o = o / o.sum()
        dev = []
        for _ in range(N_ITER):
            o = M @ o
            # M's fixed point is a DIRECTION, not a point: the iteration
            # converges to c * o* with c set by the initial condition and the
            # left eigenvector, and P = D^-1 A is not doubly stochastic, so the
            # scale is not conserved.  Without this normalisation the deviation
            # ||o - o*|| flattens at |c-1| * ||o*|| and the measured rate reads
            # 1.00000 no matter what the true contraction rate is.
            o = o / o.sum()
            dev.append(float(np.linalg.norm(o - o_star)))
        dev = np.array(dev)
        # Fit the asymptotic regime, not the whole transient.  The window must
        # start after the faster modes have died (L3/L2 = 0.71, so a few tens of
        # steps suffice) and must end before the decay underflows: at rate 0.87
        # the deviation reaches 5e-25 by step 400, and an all-zero tail makes
        # the fit raise rather than return a wrong number.
        idx = np.arange(dev.size)
        sel = (idx >= 40) & (dev > 1e-12)
        if sel.sum() < 5:
            print(f"{beta:>6.1f}   no usable fit window "
                  f"({sel.sum()} points above 1e-12 after step 40)")
            ok = False
            continue
        r_meas = float(np.exp(np.polyfit(idx[sel], np.log(dev[sel]), 1)[0]))
        rel = abs(r_meas - r_pred) / r_pred

        # Proposition 1: the reached state must not depend on beta
        same = bool(np.linalg.norm((M @ o_star) - o_star) < 1e-9)
        ok &= (rel < 0.02) and same
        print(f"{beta:>6.1f} {r_pred:>13.5f} {r_meas:>12.5f} {rel:>9.4f} "
              f"  {'yes' if same else 'NO'}")

    print("\nVERDICT")
    if ok:
        print("  Both propositions hold numerically: the fixed point is "
              "beta-independent, and the contraction rate matches "
              "|(1-beta)L2 + beta| to within 2%.")
        print("  This is what the manuscript reports as 'the separation matches "
              "the algebra'.  It is a verified proposition, not an aside.")
    else:
        print("  A proposition FAILED.  Do not describe the anchoring result as "
              "algebraically explained until this is resolved.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
