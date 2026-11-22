# get_limit

Code, protocol modules, and result artefacts for the manuscript
*"Robustness of the Fragmentation–Consensus Transition in Bounded-Confidence
Populations to Graph Dynamics, Measurement Noise, and Link Failure:
A Pre-Registered Audit"* (submitted to Physica A).

Every number, table, and figure in the manuscript is regenerated from the
CSV artefacts in this repository; `verify_paper.py` asserts this
(318 assertions over the manuscript files) and exits non-zero on any
disagreement.

## Layout

```
prereg.py  prereg_r2.py      frozen pre-registration modules (grids, seeds,
                             decision rules; prereg_r2 registers the corrected
                             trust-radius criterion as a separate, later file)
swarm.py  sociophysics.py    the plant: bounded-confidence agents on a
                             geometric graph, stages C0-C3, and the HK
                             convergence core used by the reproduction check
run_audit.py                 the registered leverage grid (stage C0-C3 x
                             N in {100,400} x eps x R, 30 episodes/cell)
run_r2.py                    the second-round non-saturating R sweep + A1 arm
control_grid.py              marginalised-leverage reference computation
beta_probe.py                anchoring-weight endpoint-effect probe
variance_probe.py            the pre-grid variance gate
mechanism_probe.py           mediator measurements (spatial degree,
                             agreement yield, in-window dispersion)
repro_check.py               instrument validation against published
                             critical thresholds
scaling_probe.py             finite-size scaling (order parameter,
                             susceptibility, Binder, collapse scan)
neighbour_collapse.py        density-account test, measured <k> version
density_collapse.py          density-account test on S and dispersion
phase_map.py                 the 840-cell (eps, R) phase diagrams
nu_bootstrap.py              bootstrap of the collapse exponent nu
derive_beta.py               numerical check of Propositions 1 and 2
make_results.py              regenerates the results tables from the CSVs
make_figures.py  figures_phase.py   regenerate figures 1-6
build_physa_submission.py    assembles the Elsevier submission file
verify_paper.py              318 assertions locking every manuscript number
                             to the CSVs
paper_meta.py                title/author metadata as supplied
*_raw.csv / *_probe.csv etc. the registered-run result artefacts
figs/                        figure PDFs as used in the manuscript
RESULTS.md  VERSION_LEDGER.md  result summaries and the two-version ledger
```

## Reproduction

Each registered script is standalone; run with Python 3.9+ and NumPy/Matplotlib:

```
python run_audit.py          # the main leverage grid (registered run)
python run_r2.py             # second round + A1 arm
python repro_check.py        # validation against published eps_c
python scaling_probe.py      # finite-size scaling
python phase_map.py          # the 840-cell phase diagrams (hours)
python nu_bootstrap.py       # nu bootstrap (replays scaling streams, fidelity-gated)
python verify_paper.py       # assert the manuscript against the CSVs
```

Every script honours `AUDIT_QUICK=1` for a fast rehearsal pass (marked grids
and fewer seeds); the committed CSVs are from the registered runs.

## Pre-registration status

`prereg.py` froze the population sizes, condition ordering, parameter grids,
seed counts, bootstrap procedure and decision rules before the main grid ran.
`prereg_r2.py` is a separate, later registration for the corrected trust-radius
criterion, so the first round's results stand as labelled by the criterion in
force when they were produced. The (eps, R) phase-diagram grids extend the
registered leverage grids under the same protocol but were designed after the
registration; the manuscript says so at the point of use.

## Withdrawn claim

An intermediate analysis suggested the trust radius's leverage reversed sign
with population size. The three-level grid it used was silently saturated
(levels 1.0 and 2.0 are the same condition), and on a non-saturating grid the
reversal does not reproduce. The claim is withdrawn in the manuscript and the
withdrawal is reported here.
