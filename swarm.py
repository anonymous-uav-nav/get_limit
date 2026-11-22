# -*- coding: utf-8 -*-
"""Multi-agent bounded-confidence swarm, with operating conditions as stages.

The protagonist is the EMERGENT CLUSTER COUNT: how many opinion groups the
swarm settles into.  Everything else here exists to move that number.

THE THREE KNOBS, AND WHY EACH CAN BE INERT
------------------------------------------
eps   confidence threshold.  Fuse with a neighbour only if the measured
      opinions differ by less than eps.  The sociophysics knob.
beta  anchoring weight.  o_i' = (1-beta) * mean_trusted + beta * o_i.
      beta = 1 freezes the swarm, beta = 0 is plain HK averaging.
R     the safety radius the controller BELIEVES it must keep.  A neighbour
      outside R is not trusted at all.

R has a mechanism for inertness that this plant makes explicit:
      trust_radius = min(sensing_radius, R)
so once R rises above the sensing radius it is SATURATED -- the swarm cannot
hear those neighbours anyway, and sweeping R further changes nothing.  That is
a candidate explanation for the D2 finding that four baselines were bitwise
insensitive to the radius they believed in, and it is exactly the kind of
structure the variance gate exists to DETECT rather than assume.

THE STAGES, OPENED ONE AT A TIME (order registered in prereg)
-------------------------------------------------------------
C0 base       static complete graph, noiseless, agents fixed
C1 +motion    agents random-walk, so the interaction graph becomes geometric
              and DYNAMIC
C2 +noise     measurement noise enters both the agreement test and the update
C3 +switching random edge dropout each round (Markov)
Each stage adds exactly one condition, so a collapse is attributable.

WHY THE EDGE TEST USES MEASURED OPINIONS
----------------------------------------
Agreement is judged on what an agent MEASURES, not on ground truth.  That is
what makes stage C2 a real operating condition rather than a cosmetic
perturbation, and it is why the noise is applied before the comparison rather
than after it.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import prereg as P  # noqa: E402
from sociophysics import count_clusters  # noqa: E402

STAGES = ("C0", "C1", "C2", "C3")


@dataclass(frozen=True)
class SwarmParams:
    """One episode.  Every knob and every condition is explicit."""
    seed: int
    stage: str
    n: int = 100
    eps: float = 0.25
    r_mult: float = 1.0        # R as a multiple of RADIUS_BASE; >1 saturates
    beta: float = 0.3
    sigma: float = 0.0         # ignored unless stage >= C2
    switch_rate: float = 0.0   # ignored unless stage >= C3
    n_steps: int = P.N_STEPS
    warm: int = P.WARM
    # A1 INTERVENTION (prereg_r2.INTERVENTION_A1_REQUIRED).  When set to a
    # target value, the opinions are rescaled about their mean after every
    # update so that the in-window opinion dispersion equals that target, no
    # matter what R is doing.  This is the "hold the candidate fixed, vary R"
    # arm: if the cluster-count slope against R changes sign under it, the
    # dispersion carries the flip; if it does not, the candidate is withdrawn.
    #
    # HONEST CAVEAT, and it is not a small one: rescaling opinions also
    # rescales the quantities the bounded-confidence test compares, so the
    # intervention CANNOT be cleanly separated from the agreement mechanism.
    # Any dispersion mediator is entangled with the agreement test by
    # construction -- the test IS a dispersion test.  If A1 cannot separate
    # them, the correct report is "the mediator is not separable from the
    # mechanism", which is a real result and not a failure to run the arm.
    pin_inwin_std: float | None = None


def stage_flags(stage: str) -> tuple[bool, bool, bool]:
    """(motion, noise, switching) implied by the stage name."""
    i = STAGES.index(stage)
    return i >= 1, i >= 2, i >= 3


def run_swarm(sp: SwarmParams, return_state: bool = False) -> dict[str, float]:
    """Run one episode and return the emergent readouts.

    `return_state=True` additionally hands back the final positions and
    opinions so a diagnostic probe can compute mediators from them.  It
    defaults to False, so every existing caller is untouched, and it changes
    nothing about the dynamics either way.
    """
    motion, noise_on, switch_on = stage_flags(sp.stage)
    rng = np.random.default_rng(sp.seed)

    n = sp.n
    opinions = rng.uniform(0.0, 1.0, size=n)
    pos = rng.uniform(0.0, P.ARENA, size=n) if motion else np.zeros(n)
    heading = rng.choice([-1.0, 1.0], size=n)

    sense_r = P.RADIUS_BASE
    trust_r = min(sense_r, P.RADIUS_BASE * sp.r_mult)

    for _ in range(sp.n_steps):
        if motion:
            pos = np.clip(pos + P.SPEED * heading, 0.0, P.ARENA)
            turn = rng.random(n) < 0.10          # occasional reversals
            heading[turn] *= -1.0

        # --- measurement: noise corrupts BOTH the agreement test and the update
        meas = opinions + (rng.normal(0.0, sp.sigma, size=n) if noise_on else 0.0)

        # --- who can be heard, and who is trusted
        if motion:
            dist = np.abs(pos[:, None] - pos[None, :])
            geo = dist <= sense_r
            within_trust = dist <= trust_r
        else:
            geo = np.ones((n, n), dtype=bool)
            within_trust = np.ones((n, n), dtype=bool)
        agree = np.abs(meas[:, None] - meas[None, :]) <= sp.eps
        adj = geo & within_trust & agree
        if switch_on and sp.switch_rate > 0.0:
            keep = rng.random((n, n)) >= sp.switch_rate
            adj = adj & keep & keep.T
        adj = adj | np.eye(n, dtype=bool)        # every agent always keeps itself

        # --- bounded-confidence update with anchoring
        cnt = adj.sum(axis=1)
        mean_trusted = (adj @ opinions) / cnt
        opinions = (1.0 - sp.beta) * mean_trusted + sp.beta * opinions

        # --- A1 intervention: pin the in-window opinion dispersion
        if sp.pin_inwin_std is not None:
            w = (geo & within_trust).astype(float)
            c = np.maximum(w.sum(axis=1), 1.0)
            m = (w @ opinions) / c
            dev = opinions[None, :] - m[:, None]
            var = (w * dev * dev).sum(axis=1) / c
            cur = float(np.mean(np.sqrt(np.maximum(var, 0.0))))
            if cur > 1e-9:
                # scaling every opinion about the global mean scales every
                # pairwise difference, hence the in-window std, by the same
                # factor -- so a single multiplication hits the target exactly
                s = sp.pin_inwin_std / cur
                mu = float(opinions.mean())
                opinions = mu + s * (opinions - mu)

    out = dict(
        clusters=float(count_clusters(opinions, P.CLUSTER_TOL)),
        spread=float(np.std(opinions)),
        opinion_mean=float(np.mean(opinions)),
    )
    if return_state:
        out["_pos"] = pos
        out["_opinions"] = opinions
    return out


def to_kwargs(sp: SwarmParams) -> dict:
    """Explicit field copy, so a sweep can override exactly one knob."""
    return dict(seed=sp.seed, stage=sp.stage, n=sp.n, eps=sp.eps,
                r_mult=sp.r_mult, beta=sp.beta, sigma=sp.sigma,
                switch_rate=sp.switch_rate, n_steps=sp.n_steps, warm=sp.warm,
                pin_inwin_std=sp.pin_inwin_std)


def mediators(sp: SwarmParams, pos: np.ndarray,
              opinions: np.ndarray) -> dict[str, float]:
    """Diagnostic quantities, recomputed from the FINAL state.

    Deliberately computed OUTSIDE the update loop so the dynamics code stays
    single-source and no probe can perturb the registered runs by accident.
    Switch dropout is not reapplied here: these describe the geometry and the
    opinion composition of the trusted set, not the particular random mask of
    the last round.

    They exist to locate a MECHANISM for the sign flip of R's lever with swarm
    size (positive at N=100, negative at N=400).  A mediator that flips sign in
    the same way is the CANDIDATE; the candidate is then tested directly, never
    asserted.

      spatial_deg   agents inside the sensing radius
      trust_deg     agents inside min(sensing radius, R)
      agree_yield   fraction of trusted pairs that also pass the opinion test
      inwin_op_std  mean, over agents, of the opinion std among its trusted
                    neighbours -- how opinion-DIVERSE each agent's reachable
                    set is
      op_range      opinion range actually held inside the trusted set
    """
    motion, _, _ = stage_flags(sp.stage)
    n = sp.n
    if motion:
        dist = np.abs(pos[:, None] - pos[None, :])
        inside = dist <= min(P.RADIUS_BASE, P.RADIUS_BASE * sp.r_mult)
        spatial = dist <= P.RADIUS_BASE
    else:
        inside = np.ones((n, n), dtype=bool)
        spatial = np.ones((n, n), dtype=bool)
    off = ~np.eye(n, dtype=bool)
    agree = np.abs(opinions[:, None] - opinions[None, :]) <= sp.eps

    trusted_pairs = int((inside & off).sum())
    agreed_pairs = int((inside & agree & off).sum())

    w = inside.astype(float)
    cnt = np.maximum(w.sum(axis=1), 1.0)
    mean = (w @ opinions) / cnt
    dev = opinions[None, :] - mean[:, None]
    var = (w * dev * dev).sum(axis=1) / cnt
    inwin_std = float(np.mean(np.sqrt(np.maximum(var, 0.0))))

    return dict(
        spatial_deg=float(spatial.sum(axis=1).mean()),
        trust_deg=float((inside & off).sum(axis=1).mean()),
        agree_yield=(agreed_pairs / trusted_pairs) if trusted_pairs else 0.0,
        inwin_op_std=inwin_std,
        op_range=float(opinions.max() - opinions.min()),
    )
