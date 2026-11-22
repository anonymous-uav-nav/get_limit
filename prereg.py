# -*- coding: utf-8 -*-
"""PREREGISTRATION -- parameter-leverage map for a bounded-confidence swarm.

FROZEN.  Every value the runner uses lives here, so the runner can assert that
it is running the registered protocol, and a later change shows up as a diff to
this file rather than as a silent decision.

WHAT IS BEING ASKED
-------------------
Not "is bounded confidence good".  The question is which of the loop's three
knobs actually controls the swarm's EMERGENT COLLECTIVE STRUCTURE -- how many
opinion clusters the swarm settles into -- and whether that control survives
the conditions a flying swarm actually operates under.

The literature answers this under idealised conditions: static complete graph,
no measurement noise, agents that do not move.  Phase diagrams and finite-size
scaling laws are quoted from exactly that setting ([arXiv 2507.14276],
[arXiv 2102.02385]).  A real swarm has none of it.  The registered claim is
therefore a boundary of validity, not a new mechanism:

    H1  the lever a knob exerts on cluster structure is measured in the
        idealised setting, quoted in the literature, and reproduced here as
        the first stage;
    H2  opening the operating conditions one at a time degrades that lever,
        and the amount of degradation is the deliverable.

WHY THIS CONTRIBUTION TYPE
--------------------------
Every mechanism cell in UAV + bounded confidence is occupied: HK fusion for
multi-UAV was published in IEEE T-ASE 2025; tunable dissensus / emergent
allocation is the Nonlinear Opinion Dynamics programme (arXiv 2610.39018,
2026-10, with a chapter in Annual Reviews in Control); cluster count versus
threshold is mature sociophysics; and the "trust weighting versus a fixed
equal-weight baseline" comparison already exists (arXiv 2205.12880 with 30
seeds, plus Safe Wrapper with hardware).  A method paper would have to beat a
fixed baseline on occupied ground, which is the failure mode that has already
killed four directions in this project.  A validity-boundary paper has no
baseline to beat.

THE THREE KNOBS
---------------
  eps   confidence threshold          -- the sociophysics knob
  R     the safety radius the controller BELIEVES it must keep
                                      -- the knob measured in the D2 probe
  beta  the trust / anchoring weight  -- the knob of the published UAV paper

R is defined here as a PARAMETER OF THIS AUDIT, not inherited from any mother
paper's metric definition.  See the disclosure section below for why that
decoupling is deliberate.

THE OPERATING CONDITIONS, OPENED ONE AT A TIME
----------------------------------------------
  C0 base      static complete graph, noiseless, agents fixed
  C1 +motion   agents move, so the interaction graph becomes dynamic
  C2 +noise    measurement noise enters the agreement test
  C3 +switching random topology dropout (Markov) on top of C2
Each stage adds exactly one condition, so a collapse can be attributed to a
stage rather than to "operating conditions" as a blob.  This ordering is
registered; running the stages in a different order would break attribution.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------
VERSION = "bcg-audit-1.0"
FROZEN_ON = "2026-11"

# --------------------------------------------------------------------------
# swarm and dynamics
# --------------------------------------------------------------------------
N_SWARM = (100, 400)           # swarm size; the literature's quoted range
OPINION_DIM = 1                # scalar opinions; multidimensional is a follow-up
DT = 0.1
N_STEPS = 400                  # interaction rounds per episode
WARM = 50                      # rounds discarded before scoring
CLUSTER_TOL = 0.01             # 1%-of-range split, matches the reproduced sweep

# motion (C1): agents on a line, interaction by geometric proximity
ARENA = 1.0
SPEED = 0.02                   # arena units per round
RADIUS_BASE = 0.30             # sensing radius -- the graph's geometric analogue of degree

# measurement noise (C2), in opinion units
SIGMA_GRID = (0.0, 0.005, 0.02)

# topology switching (C3): per-round probability an edge is dropped
SWITCH_GRID = (0.0, 0.02, 0.10)

# --------------------------------------------------------------------------
# the knob grids.  eps follows the validated reproduction exactly, so stage C0
# is directly comparable to a diagram this harness has already reproduced.
# --------------------------------------------------------------------------
EPS_GRID = tuple(round(0.15 + 0.02 * i, 4) for i in range(11))   # 0.15 .. 0.35
R_GRID = (0.5, 1.0, 2.0)       # multiples of RADIUS_BASE
BETA_GRID = (0.0, 0.3, 0.6, 1.0)

STAGES = ("C0", "C1", "C2", "C3")

# ==========================================================================
# GATE RESULTS, recorded BEFORE the main grid was run.
#
# All three gates ran while no main-grid data existed.  They are recorded here
# as pre-data revisions, in the same spirit as the two-stage search revision in
# the MFAC audit: a contraction made after seeing gate output is legitimate, a
# contraction made after seeing the ANSWERING data is not.
# ==========================================================================
GATE_RESULTS = {
    "variance_gate": {
        "file": "variance_probe.py",
        "finding": "switch_rate has no variance on the grid "
                   "(clusters_rel 0.039, spread_rel 0.078, both under "
                   "VARIANCE_MIN_SPREAD); every other axis moves.",
        "action": "switch_rate leaves the grid.  It is still held at a fixed "
                  "nonzero value in the C3 stage, so the condition is PRESENT "
                  "without being claimed as an axis.",
    },
    "beta_probe": {
        "file": "beta_probe.py",
        "finding": "On the registered protagonist beta is an ENDPOINT EFFECT: "
                   "clusters go 1.20 -> 1.30 across beta <= 0.8 (exactly flat "
                   "for beta <= 0.5) while beta = 1 jumps 30x to 37.25.  The "
                   "interior rel-spread is 0.082 / 0.158 / 0.082 at C0/C2/C3, "
                   "so it does not move consistently in any stage.",
        "action": "beta is DEMOTED to a binary condition on the cluster-count "
                  "map and is not swept as an axis there.",
        "note": "The readouts split exactly as the prereg algebra predicted: "
                "the anchored update shares the fixed-point equation o = P o "
                "with plain HK, so beta rescales the CONVERGENCE RATE, not the "
                "fixed point.  Cluster count (a fixed-point object) is inert; "
                "spread (a finite-time object) is live, interior 0.64 / 0.45.",
    },
    "confirmation_run": {
        "file": "control_grid.py",
        "finding": "The variance gate's single-reference-point reading of an "
                   "r_mult decay (0.188 -> 0.152 -> 0.039 across C1..C3) does "
                   "NOT survive marginalisation over the other axis: it becomes "
                   "0.084 -> 0.033 -> 0.060, non-monotone, and its bootstrap "
                   "interval CONTAINS ZERO AT EVERY STAGE.  eps by contrast is "
                   "+0.583 / +0.631 / +0.631 / +0.628 with tight intervals and "
                   "only 8% spread across stages.",
        "action": "The 'lever removal' claim is WITHDRAWN.  R's question is "
                  "restated as whether a lever exists at all, which the main "
                  "grid answers by BOUNDING it rather than by measuring decay.",
    },
}

# --------------------------------------------------------------------------
# the registered output and the control measure
# --------------------------------------------------------------------------
# Protagonist: the EMERGENT CLUSTER COUNT after convergence.
#
#     control(knob | stage) = (E[clusters | knob_min] - E[clusters | knob_max])
#                             / E[clusters | knob_min]
#
# computed with the other axis MARGINALISED (sampled across its grid), never at
# a single reference point: the confirmation run showed the reference-point
# version overstates the effect by roughly 2x.
CONTROL_INERT = 0.05           # below this the knob is declared inert
CONTROL_LIVE = 0.25            # above this the knob is declared live
# COLLAPSE_FRACTION is DEPRECATED.  It encoded "a stage removes >= 50% of a
# lever", which was the shape of the withdrawn r_mult claim.  The name is kept
# only so nothing silently reuses it; no criterion uses it.
COLLAPSE_FRACTION = 0.50       # DEPRECATED -- see GATE_RESULTS['confirmation_run']

# The R criterion, registered before the main grid.  R is a NON-INFERIORITY
# question: a lever on the emergent structure either exists and can be BOUNDED
# away from zero, or it cannot.  The registered N_SEEDS = 30 exists to make
# that bound tight enough to state out loud.
R_CONTROL_BOUNDABLE_CI = 0.05  # the interval's upper bound must fall below
                               # this for R to be declared to HAVE a lever; if
                               # the interval still contains zero, the honest
                               # statement is "no demonstrable lever at this
                               # power", not "R is inert".

# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
N_SEEDS = 30                   # >= 30, per the lesson from the E4 refutation
N_BOOT = 2000
CI_LEVEL = 0.95
EFFECT_MIN = 0.33              # |Cliff's delta| for a claim of a difference

# --------------------------------------------------------------------------
# MANDATORY GATE, registered before any result exists
# --------------------------------------------------------------------------
# The R3 stage of the MFAC audit died because its registered explanatory
# variable turned out to have NO variance across the experiment grid, leaving
# a regime map with no axis to test.  The same trap is available here in three
# places at once, so the gate is registered as a precondition:
#
#   NO stage of the main grid may be run until variance_probe.py has shown
#   that every axis supposed to carry an effect actually MOVES on the grid it
#   will be swept over.  An axis with no variance is removed from the grid and
#   that removal is reported, never silently absorbed.
VARIANCE_GATE_REQUIRED = True
VARIANCE_MIN_SPREAD = 0.10     # an axis must move its readout by >= 10% relative

# --------------------------------------------------------------------------
# DISCLOSURE: the self-reference question, registered but NOT decided here
# --------------------------------------------------------------------------
# The D2 probe measured the mother paper's safety-radius knob.  A question is
# open about whether that paper's min_dist is a centre-to-centre distance or a
# surface margin, which changes the SEMANTICS of the R result.
#
# This audit is made INDEPENDENT of that decision on purpose: R here is a
# parameter of the audit's own plant (the radius the controller believes it
# must keep), so both readings are handled identically at the experimental
# level.  The pending decision therefore affects only how the R result is
# WORDED, and the two branches are registered here so the wording cannot be
# chosen after the numbers are visible:
SELF_REFERENCE_BRANCHES = (
    "A  centre-to-centre: R is a distance between agent centres",
    "B  surface margin:   R is the free space left after body radii",
)
# The two branches are affine translates of one another on a fixed-radius
# problem, so a claim that holds under one holds under the other; any claim
# that would DIFFER between them is not reportable from this audit.
SELF_REFERENCE_DECISION = "PENDING -- user picks branch A or B for wording"

# --------------------------------------------------------------------------
# abort conditions
# --------------------------------------------------------------------------
# A stage that fails to converge in most episodes is not a result about
# control, it is a result about the solver, and it is reported as such.
CONVERGE_FRAC_MIN = 0.90
