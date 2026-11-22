# -*- coding: utf-8 -*-
"""SECOND, INDEPENDENT pre-registration: the sign of R's lever.

WHY THIS IS A SEPARATE FILE AND NOT AN EDIT
-------------------------------------------
The first registration (prereg.py) stated R's criterion as
`lo > 0 and hi < R_CONTROL_BOUNDABLE_CI`, which only handles a POSITIVE
effect.  The main grid then produced negative effects at N = 400 whose
intervals excluded zero, and the first registration labelled them "no
demonstrable lever" -- a wrong label produced by a criterion that never
contemplated a sign.

Editing prereg.py now would be exactly the thing pre-registration exists to
prevent: changing the rule after seeing the answer.  So the first round's
results stand AS LABELLED BY ITS OWN CRITERION, and the corrected criterion is
registered here, for a NEW run, before that run happens.  Anyone reading the
two files in order can see what was known when.

WHAT THE FIRST ROUND FOUND, AND WHAT REMAINS OPEN
-------------------------------------------------
  eps    lever ~ +0.60, CI excludes 0, flat across 4 stages and both sizes.
         Not in question; the second round re-measures it as a control.
  R      lever is SMALL (2-6%) and its SIGN DEPENDS ON SWARM SIZE: positive at
         N = 100 in all three stages, negative at N = 400 in all three.  6/6
         with no mixing, 5/6 intervals excluding zero.
  beta   endpoint effect on the protagonist (see beta_probe.py).
  mechanism   NOT established.  mechanism_probe.py located candidates and also
         exposed two defects in itself.

DEFECTS IN THE FIRST ROUND'S OWN MACHINERY, CARRIED FORWARD AS FIXES
-------------------------------------------------------------------
D1  R SATURATES.  The plant computes trust_r = min(sense_r, RADIUS_BASE *
    r_mult), so r_mult = 1.0 and r_mult = 2.0 are the SAME condition --
    mechanism_probe.py shows them identical (trust_deg 45.12 and 182.10
    respectively at BOTH 1.0 and 2.0).  The registered three-level grid was
    therefore two levels, and "change from min to max" was a single contrast,
    not a sweep.

D2  NORMALISATION DIVIDES BY ZERO.  control and the mechanism deltas were
    normalised by the value at the grid's LOW end.  At N = 400, R = 0.5 the
    swarm reaches exact consensus (op_range = 0.0000), so that denominator is
    zero and the printed normalised changes (-37844.9) are division artefacts.
    The sign flags happened to survive; the magnitudes did not.

D3  A "SAME SIGN" TEST MISSES CONSISTENT ANTI-CORRELATION.  agree_yield
    responds to R with opposite sign at the two sizes, and is anti-correlated
    with the cluster count within EACH size -- a relation that can carry the
    flip just as well as a positive one, and mechanism_probe.py's test could
    not see it.

WHAT THE SECOND ROUND REGISTERS
------------------------------
An R grid that does not saturate, a normalisation that cannot divide by zero,
and a criterion about the SIGN PATTERN rather than about positivity.
"""
from __future__ import annotations

VERSION = "bcg-audit-r2-1.0"
FROZEN_ON = "2026-11"

# --------------------------------------------------------------------------
# D1 fix: a NON-SATURATING R grid
# --------------------------------------------------------------------------
# All levels at or below r_mult = 1.0, where trust_r = RADIUS_BASE * r_mult is
# not clipped by the sensing radius.  The saturated condition is run
# separately and LABELLED, rather than smuggled in as the grid's top end.
R_GRID_V2 = (0.2, 0.4, 0.6, 0.8, 1.0)   # 1.0 is exactly the saturation point
R_SATURATED = 2.0                        # reported as a distinct condition

# --------------------------------------------------------------------------
# D2 fix: a normaliser that cannot be zero
# --------------------------------------------------------------------------
# Control is normalised by the MEAN cluster count over the whole R grid at that
# (stage, N), not by the grid's low end.  The mean is bounded below by 1 -- a
# swarm always has at least one cluster -- so the denominator cannot vanish.
NORMALISE_BY_GRID_MEAN = True
MIN_CLUSTERS = 1.0

# --------------------------------------------------------------------------
# D3 fix: the criterion is about the SIGN PATTERN
# --------------------------------------------------------------------------
# For each (stage, N) the effect is a SIGNED SLOPE of cluster count against R,
# estimated by least squares over the grid with a bootstrap interval over
# episodes.  The registered claims are then:
#
#   S1  the slope is significantly negative at N = 400 (upper bound < 0)
#   S2  the slope is significantly positive at N = 100 (lower bound > 0)
#   S3  the DIFFERENCE between the two slopes has an interval excluding zero
#
# S3 is the actual claim.  S1 and S2 alone would not establish a size-dependent
# SIGN: two slopes can differ in magnitude while sharing a sign.
SIGN_FLIP_REQUIRES_S3 = True
SLOPE_CI_LEVEL = 0.95
N_BOOT_V2 = 4000

# --------------------------------------------------------------------------
# mechanism: still NOT claimed
# --------------------------------------------------------------------------
# mechanism_probe.py located candidates (in-window opinion dispersion, and
# agreement yield under a corrected sign test) and ruled one out -- reach,
# because trust_deg rises with R identically at both sizes and so cannot carry
# a flip.  A candidate is NOT a mechanism.
#
# The second round therefore registers a DIRECT INTERVENTION, the only thing
# that can convert a candidate into a mechanism: hold the candidate fixed and
# vary R.
#
#   A1  resample neighbour opinions so the in-window opinion dispersion is
#       pinned at the N = 100 level while R sweeps at N = 400.
#       If the slope then turns NEGATIVE, dispersion carries the flip.
#       If it stays POSITIVE, dispersion is NOT the mediator and the candidate
#       is withdrawn.
#
# ERRATUM, recorded rather than silently corrected.  This section first read
# "turns POSITIVE", which is BACKWARDS, and the error is a sign slip in this
# text rather than a judgement about data.  It is derivable by algebra from the
# first registration's own definition: control = (E[clusters|R_min] -
# E[clusters|R_max]) / E[clusters|R_min], so a POSITIVE control at N = 100
# (+0.0581) means clusters FALL as R rises, i.e. a NEGATIVE slope; and a
# NEGATIVE control at N = 400 (-0.0246) means a POSITIVE slope.  Pinning the
# N = 400 dispersion to the N = 100 level would therefore make N = 400 behave
# like N = 100 only if the slope becomes NEGATIVE.  The erratum was NOTICED
# during a rehearsal, which is stated here so a reader can weigh it: the
# correction is algebra, not results, but the noticing was not independent of
# having run something.
#
# An intervention that fails to change the slope is a REAL RESULT and is
# reported as "candidate withdrawn", never quietly dropped.
INTERVENTION_A1_REQUIRED = True
INTERVENTION_TOL = 0.02        # pinned dispersion must land within this of the
                               # N = 100 target for the arm to count as pinned

# --------------------------------------------------------------------------
# what the second round is NOT allowed to claim
# --------------------------------------------------------------------------
FORBIDDEN_CLAIMS = (
    "R is inert -- the first round showed a real effect; small and "
    "sign-unstable is a different and worse problem, not the same one",
    "a mechanism for the flip -- not until intervention A1 reports",
    "that the first round's labels were wrong -- they were correct as applied "
    "by the first round's own criterion, which is the point of registering "
    "the fix separately",
)
