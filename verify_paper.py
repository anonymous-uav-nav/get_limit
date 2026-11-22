# -*- coding: utf-8 -*-
"""Lock every number in main.tex back to the artefacts.

House rule: every claim in the manuscript gets its own assertion over the CSV
or the protocol module it came from, so a later edit cannot silently drift a
number.  This script reads the manuscript as text, re-derives the values it is
supposed to contain, and FAILS -- not warns -- on any disagreement.

What it checks
--------------
1. VALUES.  Each headline number is re-derived from the CSV artefacts and the
   manuscript is asserted to contain it.
2. PAIRINGS.  A table value must appear with its OWN interval, not merely the
   value somewhere in the file.  Presence alone would pass if a row's number
   drifted into another row.
3. FORBIDDEN CLAIMS.  The phrase "sign flip" and its variants must not appear:
   the intermediate claim they describe was withdrawn, and leaving the wording
   in would re-assert it.
4. REQUIRED RETRACTIONS.  The withdrawal must be present.  A negative result
   that quietly disappears is worse than one that was never reported.
5. IDENTITY.  Author names, ORCIDs and e-mail addresses in the manuscript must
   match paper_meta.py, which records them as the author supplied them.

Exit code is 0 only if every assertion holds.
"""
from __future__ import annotations

import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import paper_meta as PM  # noqa: E402
import prereg as PR  # noqa: E402  -- the frozen protocol module itself

HERE = os.path.dirname(os.path.abspath(__file__))
# Both framings are checked.  They share every number and differ only in the
# identity of the paper, so a drift in one must not go unnoticed in the other.
TEXES = ("main.tex", "main_physa.tex", "manuscript_physa2026.tex")


def rows(name: str) -> list[dict]:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fv(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def eps_c_from_repro() -> dict[int, float]:
    """Same crossing rule as make_results.py: first eps where P>=0.5."""
    data = rows("repro_check.csv")
    out: dict[int, float] = {}
    for n in sorted({int(fv(r, "n")) for r in data}):
        sub = [r for r in data if int(fv(r, "n")) == n]
        e = np.array([fv(r, "eps") for r in sub])
        p = np.array([fv(r, "consensus_frac") for r in sub])
        o = np.argsort(e)
        e, p = e[o], p[o]
        above = p >= 0.5
        if not above.any():
            continue
        i = int(np.argmax(above))
        out[n] = float(e[0]) if i == 0 else (
            float(e[i - 1] + (0.5 - p[i - 1]) * (e[i] - e[i - 1])
                  / (p[i] - p[i - 1]))
            if p[i] != p[i - 1] else float(e[i]))
    return out


def ctrl(stage: str, nsw: int, knob: str) -> dict:
    for r in rows("audit_control.csv"):
        if (r["stage"] == stage and int(fv(r, "n_swarm")) == nsw
                and r["knob"] == knob):
            return r
    return {}


def slope(arm: str, nsw: int) -> dict:
    for r in rows("r2_slopes.csv"):
        if r["arm"] == arm and int(fv(r, "n_swarm")) == nsw:
            return r
    return {}


def check(fname: str) -> tuple[list[tuple[str, bool, str]], str]:
    tex = open(os.path.join(HERE, fname), encoding="utf-8").read()

    # Line wrapping is not a claim.  Collapse every whitespace run (newlines
    # included) to a single space, so a phrase split across a line break is
    # still found.  An earlier version of this script missed "not detectable"
    # purely because the manuscript wrapped between the two words.
    flat = re.sub(r"\s+", " ", tex)
    # Comments are read by nobody.  Strip them for the forbidden-claim check so
    # that the header comment is free to discuss the withdrawn claim without
    # tripping a rule meant for the rendered text.
    body = re.sub(r"(?<!\\)%.*", "", tex)
    body_flat = re.sub(r"\s+", " ", body)

    checks: list[tuple[str, bool, str]] = []

    def need(label: str, token: str) -> None:
        checks.append((label, token in flat, f"'{token}'"))

    def forbid(label: str, token: str) -> None:
        checks.append((label, token.lower() not in body_flat.lower(),
                       f"'{token}'"))

    def require(label: str, token: str) -> None:
        checks.append((label, token.lower() in flat.lower(), f"'{token}'"))

    # ---------------------------------------------------------------- 1 values
    ec = eps_c_from_repro()
    for n, v in ec.items():
        need(f"repro eps_c(N={n})", f"{v:.4f}")
    need("published target N=100", "0.25")
    need("published target large-N", "0.20")
    d100 = ec.get(100, float("nan")) - 0.25
    d800 = ec.get(800, float("nan")) - 0.20
    need("repro deviation at N=100", f"{abs(d100):.4f}")
    need("repro deviation at N=800", f"{abs(d800):.4f}")

    for stage in ("C0", "C1", "C2", "C3"):
        for nsw in (100, 400):
            r = ctrl(stage, nsw, "eps")
            if r:
                need(f"eps control {stage}/N={nsw}", f"{fv(r, 'control'):.4f}")

    for stage in ("C1", "C2", "C3"):
        for nsw in (100, 400):
            r = ctrl(stage, nsw, "r_mult")
            if r:
                need(f"R control {stage}/N={nsw}",
                     f"{abs(fv(r, 'control')):.4f}")

    for nsw in (100, 400):
        s = slope("control", nsw)
        if s:
            need(f"round-2 slope N={nsw}", f"{abs(fv(s, 'slope')):.3f}")
            need(f"round-2 CI bound a N={nsw}", f"{abs(fv(s, 'ci_lo')):.3f}")
            need(f"round-2 CI bound b N={nsw}", f"{abs(fv(s, 'ci_hi')):.3f}")

    bp = rows("beta_probe.csv")
    inner = [fv(r, "mean_clusters") for r in bp
             if str(r.get("in_interior")).lower() == "true"]
    corner = [fv(r, "mean_clusters") for r in bp
              if str(r.get("in_interior")).lower() == "false"]
    if inner and corner:
        need("beta interior min", f"{min(inner):.2f}")
        need("beta interior max", f"{max(inner):.2f}")
        need("beta corner", f"{np.mean(corner):.2f}")
        need("beta corner ratio", f"{np.mean(corner) / np.mean(inner):.1f}")

    vp = rows("variance_probe.csv")
    sw = [r for r in vp if r.get("axis") == "switch_rate"]
    if sw:
        need("inert axis clusters_rel", f"{fv(sw[0], 'clusters_rel_spread'):.3f}")
        need("inert axis spread_rel", f"{fv(sw[0], 'spread_rel_spread'):.3f}")

    refpt = {r["stage"]: fv(r, "clusters_rel_spread") for r in vp
             if r.get("axis") == "r_mult" and r.get("stage") in ("C1", "C2", "C3")}
    marg = {r["stage"]: fv(r, "control") for r in rows("control_grid.csv")
            if r.get("knob") == "r_mult" and r.get("stage") in ("C1", "C2", "C3")}
    for stage in ("C1", "C2", "C3"):
        if stage in refpt:
            need(f"reference-point value {stage}", f"{refpt[stage]:.3f}")
        if stage in marg:
            need(f"marginalised value {stage}", f"{marg[stage]:.3f}")

    # --------------------------------------------------------------- 2 pairings
    for stage in ("C0", "C1", "C2", "C3"):
        for nsw in (100, 400):
            r = ctrl(stage, nsw, "eps")
            if r:
                need(f"pairing eps {stage}/N={nsw}",
                     f"{fv(r, 'control'):.4f}$ & "
                     f"$[{fv(r, 'ci_lo'):.4f}, {fv(r, 'ci_hi'):.4f}]$")
    for stage in ("C1", "C2", "C3"):
        for nsw in (100, 400):
            r = ctrl(stage, nsw, "r_mult")
            if r and fv(r, "control") < 0:
                # Keep the signs.  An earlier version built this token from
                # absolute values and never matched the table it was checking.
                need(f"pairing R {stage}/N={nsw}",
                     f"${fv(r, 'control'):.4f}$ & "
                     f"$[{fv(r, 'ci_lo'):.4f}, {fv(r, 'ci_hi'):.4f}]$")

    # ------------------------------------------------- 3 forbidden assertions
    for phrase in ("sign flip", "flips sign", "sign changes with swarm size",
                   "reverses with swarm size"):
        forbid(f"forbidden claim: {phrase}", phrase)

    # A phrase may appear inside a sentence that RETRACTS it -- that is the
    # honest way to withdraw a claim, so it is allowed, but only there.  Every
    # sentence mentioning the reversal must also carry a retraction marker.
    for sent in re.split(r"(?<=[.])\s+", body_flat):
        if "reversed sign" in sent.lower():
            ok = any(m in sent.lower() for m in
                     ("suggested", "withdraw", "not reproduce", "did not"))
            checks.append(("retraction context for 'reversed sign'", ok,
                           f"sentence: {sent.strip()[:90]}"))

    # ------------------------------------------------ 4 required retractions
    for phrase in ("withdrawn", "does not reproduce", "not detectable"):
        require(f"required honesty: {phrase}", phrase)

    # --------------------------------------------------------------- 5 identity
    for a in PM.AUTHORS:
        need(f"author name {a['name']}", a["name"])
        need(f"ORCID {a['name']}", a["orcid"])
    need("co-corresponding stated", "co-corresponding")
    need("corresponding e-mail 1", "798733991@qq.com")
    need("corresponding e-mail 2", "Jian.Yu@cs.khpi.edu.ua")

    # ------------------------------------------- 6 scaling and anchoring blocks
    # These arrived with the analytical work.  They are locked here for the same
    # reason as everything else: a number that is not asserted will drift.
    if fname == "main_physa.tex":
        sc = rows("scaling_raw.csv")
        if sc:
            ns = sorted({int(fv(r, "n")) for r in sc})
            peaks = []
            for n in ns:
                sub = [r for r in sc if int(fv(r, "n")) == n]
                p = max(sub, key=lambda r: fv(r, "chi"))
                peaks.append(fv(p, "chi"))
                need(f"chi peak N={n}", f"{fv(p, 'chi'):.3f}")
                need(f"chi peak eps N={n}", f"{fv(p, 'eps'):.2f}")
            need("chi growth factor", f"{peaks[-1] / peaks[0]:.1f}")

        nbr = rows("neighbour_raw.csv")
        if nbr:
            # Only the SPREADS are quoted in the prose; the per-cell cluster
            # counts are not, because the prose reports the collapse quality
            # rather than the table.  Asserting cells that are not written would
            # be checking the script against itself, not the manuscript.
            import itertools
            by_k = {}
            for r in nbr:
                by_k.setdefault(fv(r, "k_expected"), {})[int(fv(r, "n_swarm"))] = \
                    fv(r, "clusters")
            for k, cells in sorted(by_k.items()):
                if len(cells) < 2:
                    continue
                got = list(cells.values())
                rel = (max(got) - min(got)) / max(max(got), 1e-9)
                need(f"neighbour spread <k>={k:.0f}", f"{rel:.1%}".replace("%", r"\%"))

        den = rows("density_raw.csv")
        if den:
            # the pair quoted in the prose: <k> = 48, N = 200 and N = 400,
            # expressed as 1-S because S itself is against its ceiling
            for r in den:
                if abs(fv(r, "k_assumed") - 48.0) < 1e-6 \
                        and int(fv(r, "n_swarm")) in (200, 400):
                    one_minus = 1.0 - fv(r, "S_max")
                    need(f"1-S at <k>=48 N={int(fv(r,'n_swarm'))}",
                         f"{one_minus:.4f}")
            # and the ceiling artefact quoted for <k> = 96
            for r in den:
                if abs(fv(r, "k_assumed") - 96.0) < 1e-6 \
                        and int(fv(r, "n_swarm")) == 200:
                    need("S at <k>=96 N=200", f"{fv(r, 'S_max'):.4f}")

        # Propositions 1 and 2 are checked numerically by derive_beta.py; the
        # manuscript quotes its output, so the quoted rate is asserted here.
        for beta, pred in (("0.0", "0.87148"), ("0.2", "0.89719"),
                           ("0.4", "0.92289"), ("0.6", "0.94859"),
                           ("0.8", "0.97430")):
            need(f"anchoring predicted rate beta={beta}", pred)

    # ------------------------------------------------- 6.5 phase-diagram block
    # Every number in the (eps, R) plane section re-derived from phase_raw.csv.
    # The half-maximum band replaces the argmax wherever the ridge is
    # flat-topped; the derivation below mirrors the prose exactly.
    ph = rows("phase_raw.csv")
    if ph and fname == "main_physa.tex":
        n_cells = len(ph)
        need("phase cell count", str(n_cells))

        def block(st: str, n: int) -> list[dict]:
            return [r for r in ph
                    if r["stage"] == st and int(fv(r, "n_swarm")) == n]

        # block ridges (argmax) per condition, both sizes
        ridges = {}
        for st in ("C0", "C1", "C2", "C3"):
            for n in (100, 400):
                b = max(block(st, n), key=lambda r: fv(r, "chi"))
                ridges[(st, n)] = fv(b, "eps")
                need(f"phase block ridge {st} N={n}", f"{fv(b, 'eps'):.2f}")
        for n in (100, 400):
            span = (max(ridges[(s, n)] for s in ("C0", "C1", "C2", "C3"))
                    - min(ridges[(s, n)] for s in ("C0", "C1", "C2", "C3")))
            need(f"phase ridge span N={n}", f"{span:.2f}")

        # half-maximum bands per condition, and their intersection
        def halfmax(st: str, n: int) -> tuple[float, float]:
            s = block(st, n)
            mx = max(fv(r, "chi") for r in s)
            ine = [fv(r, "eps") for r in s if fv(r, "chi") >= 0.5 * mx - 1e-12]
            return min(ine), max(ine)

        for n in (100, 400):
            bands = [halfmax(st, n) for st in ("C0", "C1", "C2", "C3")]
            lo = max(b[0] for b in bands)
            hi = min(b[1] for b in bands)
            need(f"phase halfmax band lo N={n}", f"{lo:.2f}")
            need(f"phase halfmax band hi N={n}", f"{hi:.2f}")

        # worst per-R ridge departure from the C0 ridge
        for n in (100, 400):
            c0e = ridges[("C0", n)]
            worst = 0.0
            for st in ("C1", "C2", "C3"):
                for R in (0.2, 0.4, 0.6, 0.8, 1.0):
                    ss = [r for r in block(st, n)
                          if abs(fv(r, "r_mult") - R) < 1e-9]
                    e = fv(max(ss, key=lambda r: fv(r, "chi")), "eps")
                    worst = max(worst, abs(e - c0e))
            need(f"phase worst per-R departure N={n}", f"{worst:.2f}")

        # ridge heights at N=400 and the worst relative drop
        h0 = max(fv(r, "chi") for r in block("C0", 400))
        need("phase ridge height C0 N=400", f"{h0:.2f}")
        hs = {st: max(fv(r, "chi") for r in block(st, 400))
              for st in ("C1", "C2", "C3")}
        for st in ("C1", "C2", "C3"):
            need(f"phase ridge height {st} N=400", f"{hs[st]:.2f}")
        need("phase worst rel height drop N=400",
             f"{(1 - min(hs.values()) / h0) * 100:.1f}")

        # C0 bitwise R-invariance: every field identical across radii
        viol = 0
        for r in block("C0", 100) + block("C0", 400):
            grp = [q for q in block("C0", int(fv(r, "n_swarm")))
                   if abs(fv(q, "eps") - fv(r, "eps")) < 1e-9]
            keys = {(q["S_max"], q["chi"], q["clusters"], q["spread"])
                    for q in grp}
            if len(keys) != 1:
                viol += 1
        need("phase C0 R-invariance violations", str(viol))

        # ordered tail at N=400: eps>=0.26 exact S=1/chi=0/one cluster
        tail = [r for r in ph if int(fv(r, "n_swarm")) == 400
                and fv(r, "eps") >= 0.26 - 1e-9]
        tail_bad = [r for r in tail
                    if abs(fv(r, "chi")) > 1e-9
                    or abs(fv(r, "S_max") - 1.0) > 1e-9
                    or abs(fv(r, "clusters") - 1.0) > 1e-9]
        need("phase N400 tail cells", str(len(tail)))
        need("phase N400 tail violations", str(len(tail_bad)))

        # N=100 residue in every condition at eps>=0.28
        res_counts = {}
        for st in ("C0", "C1", "C2", "C3"):
            ss = [r for r in block(st, 100) if fv(r, "eps") >= 0.28 - 1e-9]
            res_counts[st] = sum(1 for r in ss if fv(r, "chi") > 1e-9)
        res_total = sum(1 for r in ph if int(fv(r, "n_swarm")) == 100
                        and fv(r, "eps") >= 0.28 - 1e-9)
        need("phase N100 residue min", str(min(res_counts.values())))
        need("phase N100 residue max", str(max(res_counts.values())))

        # the quoted below-boundary R response (C1, N=100, eps=0.20)
        for R in (0.2, 1.0):
            rr = [r for r in block("C1", 100)
                  if abs(fv(r, "r_mult") - R) < 1e-9
                  and abs(fv(r, "eps") - 0.20) < 1e-9][0]
            need(f"phase quoted clusters C1 N=100 eps=0.20 R={R:.1f}",
                 f"{fv(rr, 'clusters'):.2f}")

        # the quoted N=100 residue at eps=0.30 (C0; identical across R)
        r30 = [r for r in block("C0", 100)
               if abs(fv(r, "eps") - 0.30) < 1e-9][0]
        need("phase N100 residue S at 0.30", f"{fv(r30, 'S_max'):.4f}")
        need("phase N100 residue chi at 0.30", f"{fv(r30, 'chi'):.4f}")
        need("phase N100 residue clusters at 0.30",
             f"{fv(r30, 'clusters'):.4f}")

        # figure-caption claim: N=400 ridges ~4.5x the N=100 ones
        h100 = max(fv(r, "chi") for r in ph if int(fv(r, "n_swarm")) == 100)
        need("phase ridge height ratio", f"{h0 / h100:.1f}")

    # ------------------------------------------------- 6.4 fixed-parameters block
    # The fixed-parameter paragraph in the Model section quotes the constants
    # of the frozen protocol module; each is asserted against that module, not
    # against a CSV, because that is where the value lives.
    if fname == "main_physa.tex":
        need("fixed speed", f"{PR.SPEED:.2f}")
        need("fixed R_sense", f"{PR.RADIUS_BASE:.2f}")
        need("fixed n_steps", str(PR.N_STEPS))
        need("fixed warm", str(PR.WARM))
        need("fixed cluster tol percent", f"{PR.CLUSTER_TOL * 100:.0f}")
        # run_audit.py fixes these for every grid; read the source line so a
        # later edit of the constant and of this assertion cannot drift apart
        audit_src = open(os.path.join(HERE, "run_audit.py"),
                         encoding="utf-8").read()
        fixed = dict(re.findall(r"^(BETA_FIXED|SIGMA_FIXED|SWITCH_FIXED)\s*=\s*"
                                r"([\d.]+)", audit_src, re.M))
        need("fixed beta", f"{float(fixed['BETA_FIXED']):.1f}")
        need("fixed sigma", f"{float(fixed['SIGMA_FIXED']):.3f}")
        need("fixed switch", f"{float(fixed['SWITCH_FIXED']):.2f}")

    # ------------------------------------------------------- 6.45 seed census
    # The Statistics paragraph announces an episode count per grid; each is
    # counted from its artefact so the sentence cannot drift from the data.
    if fname == "main_physa.tex":
        from collections import Counter
        ag = rows("audit_grid.csv")
        if ag:
            per_cell = Counter((r["stage"], r["n_swarm"], r["eps"],
                                r["r_mult"]) for r in ag)
            need("seed census main grid", str(sorted(set(per_cell.values()))[0]))
        rg = rows("r2_grid.csv")
        if rg:
            per_cell = Counter((r["arm"], r["n_swarm"], r["r_mult"])
                               for r in rg)
            need("seed census round-2 grid",
                 str(sorted(set(per_cell.values()))[0]))
        sc = rows("scaling_raw.csv")
        if sc:
            need("seed census scaling grid",
                 str(int(float(sc[0]["n_seeds"]))))
        if ph:
            need("seed census phase grid", str(int(float(ph[0]["n_seeds"]))))
        rc = rows("repro_check.csv")
        if rc:
            need("seed census reproduction grid",
                 str(int(float(rc[0]["n_seeds"]))))

    # --------------------------------------------------- 6.46 ridge-agreement
    # The phase-map section no longer claims the ridges "match" the scaling
    # section; it states the argmax offset and the band overlap.  The offset
    # itself is derivable from the two artefacts.
    if fname == "main_physa.tex":
        sc2 = rows("scaling_raw.csv")
        if sc2 and ph:
            for n in (100, 400):
                p = max([r for r in sc2 if int(fv(r, "n")) == n],
                        key=lambda r: fv(r, "chi"))
                off = abs(ridges[("C0", n)] - fv(p, "eps"))
                need(f"phase-vs-scaling argmax offset N={n}", f"{off:.2f}")
        need("ridge agreement wording", "sits two grid steps below")
        need("ridge agreement bands", "the argmax wander which the")
        need("ridge agreement overlap word", "bands absorb, not a disagreement")

    # --------------------------------------------------- 6.47 slope mechanics
    # The round-2 paragraph states the slope construction; the bootstrap count
    # comes from the frozen second-registration module, the resample unit from
    # the runner source.
    if fname == "main_physa.tex":
        import prereg_r2 as R2M  # noqa: E402
        # the prose spells the count out in words; assert both the word form
        # and that the words agree with the frozen module's integer
        words = {4000: "four thousand", 2000: "two thousand",
                 1000: "one thousand", 8000: "eight thousand"}
        need("slope bootstrap count", words[int(R2M.N_BOOT_V2)])
        r2src = open(os.path.join(HERE, "run_r2.py"), encoding="utf-8").read()
        m_resample = re.search(r"integers\(0, n_seeds, size=n_seeds\)", r2src)
        checks.append(("slope resample unit is the episode",
                       m_resample is not None,
                       "run_r2.py resamples seeds within each radius level"))
        need("slope construction wording", "least-squares fit over the five")
        need("slope percentile wording", "2.5 and 97.5 percentiles")

    # --------------------------------------------- 6.48 S-at-peak and table note
    # The non-monotone S-at-peak caveat quotes the bracketing pair at N=800;
    # the Table 3 caption points the R rows at the saturating-grid caveat.
    if fname == "main_physa.tex":
        sc3 = rows("scaling_raw.csv")
        if sc3:
            sub8 = sorted([r for r in sc3 if int(fv(r, "n")) == 800],
                          key=lambda r: fv(r, "eps"))
            peak8 = max(sub8, key=lambda r: fv(r, "chi"))
            s_at_peak = fv(peak8, "mean_S")
            nxt = min(sub8, key=lambda r: abs(fv(r, "eps")
                                              - (fv(peak8, "eps") + 0.01)))
            s_next = fv(nxt, "mean_S")
            need("S at peak N=800", f"{s_at_peak:.3f}")
            need("S one step above peak N=800", f"{s_next:.3f}")
        need("S at peak not-monotone wording",
             "the one column of Table")
        need("leverage caption saturating note",
             "rests on a two-level contrast rather than a sweep")

    # -------------------------------------------------- 6.49 abstract anchors
    # The compressed abstract carries the phase-diagram finding; lock its
    # wording and the word budget separately (the budget is advisory, the
    # claim is not).
    if fname == "main_physa.tex":
        need("abstract phase claim",
             "stays within two grid steps")
        need("abstract conceal claim",
             "conceal the contents of the")
        # The submission's abstract is hand-maintained in the front matter and
        # is NOT spliced by the build script; a drift here is silent.  Both
        # abstracts must carry the same phase-diagram sentence.  Matching is
        # done on whitespace-collapsed text -- the phrase wraps across lines
        # in both files, and a raw substring test false-fails on the wrap.
        if fname == "main_physa.tex":
            sub = open(os.path.join(HERE, "manuscript_physa2026.tex"),
                       encoding="utf-8").read()
            sub_flat = re.sub(r"\s+", " ", sub)
            key = "stays within two grid steps"
            checks.append(("submission abstract in sync with source abstract",
                           key in sub_flat, f"'{key}' present in submission"))

    # ------------------------------------------------ 6.52 innovation narrative
    # The desk-reject pass of 2026-11-21 rewrote the abstract and the
    # introduction around the three-way classification; the classification
    # words themselves are the contribution statement and are locked.
    if fname == "main_physa.tex":
        need("narrative survives", "The confidence threshold \emph{survives}")
        need("narrative degrades", "The trust radius's leverage "
             "\emph{degrades}")
        need("narrative immune", "is \emph{immune} by algebra")
        need("narrative transfer frame",
             "which published effects a modeller may carry out")
        forbid("defensive not-a-mechanism gone (abstract)",
               "answering with a pre-registered audit, not a mechanism")
        forbid("defensive no-baseline gone",
               "it has no baseline to beat")

    # ------------------------------------------------- 6.50 nu uncertainty
    # The bootstrap CI block quotes the summary of nu_bootstrap.csv; read the
    # summary back and assert the prose against it.
    if fname == "main_physa.tex":
        nb = rows("nu_bootstrap.csv")
        if nb:
            summary = {r["round"]: r["nu_star"] for r in nb
                       if r.get("round") in ("n_boot", "nu_star_median",
                                             "nu_star_ci_lo", "nu_star_ci_hi",
                                             "sharp_fraction")}
            need("nu bootstrap median", f"{float(summary['nu_star_median']):.2f}")
            need("nu bootstrap CI lo", f"{float(summary['nu_star_ci_lo']):.2f}")
            need("nu bootstrap CI hi", f"{float(summary['nu_star_ci_hi']):.2f}")
            need("nu bootstrap sharp pct",
                 f"{float(summary['sharp_fraction']) * 100:.1f}")
            need("nu bootstrap rounds", summary["n_boot"])
        need("nu sobering wording", "the answer is sobering")
        need("nu no-second-decimal wording",
             "we do not lean on the second")

    # -------------------------------------------------- 6.51 eight-lens audit
    # The topic-sentence and concept-definition pass of 2026-11-21 locked its
    # rewrites: no convergence-test claim, the leverage disambiguation, the
    # collapsed-paragraph heading, the genre-precedent front-load, and the
    # removed decorative citation.
    if fname == "main_physa.tex":
        need("settled convention wording",
             "There is no convergence")
        need("settled spot-check wording",
             "unchanged at $1600$ rounds")
        need("leverage two-quantities wording",
             "covers two incommensurable quantities")
        need("leverage non-monotone caveat",
             "an endpoint contrast summarises the grid rather than")
        need("leverage denominator bound", "stays between $1.5$")
        need("collapse heading rewrite",
             "and we quantify how well the")
        need("genre precedent front-loaded",
             "The nearest genre precedent for this work is")
        need("gap bridge in related work",
             "is the question the")
        forbid("decorative citation removed", "tokmak2025towards")
    # The two framings target journals with different scopes.  Physica A
    # excludes work framed as mechanics or mechanical engineering, so the
    # engineering vocabulary must NOT survive into the Physica A file; the
    # Drones file is the opposite case and must keep it.
    if fname in ("main_physa.tex", "manuscript_physa2026.tex"):
        for word in ("robot swarm", "unmanned aerial", "uav", "drone"):
            forbid(f"physa scope: no '{word}'", word)
    else:
        require("drones framing present", "robot swarm")

    return checks, body_flat


def main() -> int:
    all_checks: list[tuple[str, str, bool, str]] = []
    flats: dict[str, str] = {}
    for fname in TEXES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            all_checks.append((fname, "file present", False, fname))
            continue
        checks, flat = check(fname)
        flats[fname] = flat
        all_checks += [(fname, lab, ok, det) for lab, ok, det in checks]

    # Cross-file: the two framings must agree on the ORIGINAL numeric content.
    # They are no longer expected to be identical, because the analytical blocks
    # (scaling, anchoring propositions, the three refuted explanations) were
    # added to the Physica A framing only.  What must still hold is that nothing
    # in the shared core drifted: every decimal appearing in the Drones framings
    # must also appear in the Physica A one.
    if len(flats) == len(TEXES):
        nums = {k: set(re.findall(r"\d+\.\d+", v)) for k, v in flats.items()}
        base = nums[TEXES[0]]
        other = nums[TEXES[1]]
        missing = sorted(base - other)
        same = not missing
        all_checks.append((f"{TEXES[1]} superset of {TEXES[0]}",
                           "shared numbers preserved", same,
                           f"missing from second: {missing[:8]}"))

    print(f"verify_paper: {len(all_checks)} assertions over {TEXES}\n")
    bad = [c for c in all_checks if not c[2]]
    for fname, lab, ok, det in all_checks:
        if not ok:
            print(f"  [FAIL] {fname}: {lab}  ->  expected {det}")
    if not bad:
        print("  all assertions hold")
    print(f"\n{len(all_checks) - len(bad)}/{len(all_checks)} passed")
    if bad:
        print("\nVERDICT: FAIL -- a manuscript and its artefacts disagree. "
              "Fix the manuscript, not this script.")
        return 1
    print("VERDICT: PASS -- every checked number matches its artefact, in "
          "both framings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
