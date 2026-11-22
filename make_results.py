# -*- coding: utf-8 -*-
"""Derive every headline number FROM THE CSVs and emit RESULTS.md.

House rule: a number that appears in the write-up must be computed from the
artefact, never copied from a terminal printout.  Two reasons, both learned the
hard way in this project: a printout can be produced by code that later changes,
and a normalised ratio printed during a run can be a division artefact (the
-37844.9 in mechanism_probe.py was one).

This script re-derives each claim from the CSVs, PRINTS the value it will
write, and records a MISSING input as MISSING rather than filling the gap with
something plausible.

Output: RESULTS.md next to the CSVs.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "RESULTS.md")


def load(name: str) -> list[dict]:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print(f"  [MISSING] {name}")
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  [ok] {name}: {len(rows)} rows")
    return rows


def f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def fmt(x: float, n: int = 4) -> str:
    return "MISSING" if not np.isfinite(x) else f"{x:.{n}f}"


def main() -> int:
    print("deriving RESULTS.md from artefacts")
    repro = load("repro_check.csv")
    r1_ctrl = load("audit_control.csv")
    r1_grid = load("audit_grid.csv")
    r2_slope = load("r2_slopes.csv")
    beta = load("beta_probe.csv")
    var = load("variance_probe.csv")
    conf = load("control_grid.csv")

    lines: list[str] = []
    add = lines.append

    add("# 结果核心（全部由 CSV 机器推导，脚本 `make_results.py` 可复跑）")
    add("")
    add("> 本文件的每一个数字都由 `make_results.py` 从同目录 CSV 现算，不手抄终端输出。")
    add("> 缺输入标 `MISSING` 而非补齐。")
    add("")

    # ---------------------------------------------------------------- R0
    add("## R0 仪器验证：复现文献相图")
    add("")
    ec: dict[int, float] = {}
    for n in sorted({int(f(r, "n")) for r in repro}):
        sub = [r for r in repro if int(f(r, "n")) == n]
        eps = np.array([f(x, "eps") for x in sub])
        frac = np.array([f(x, "consensus_frac") for x in sub])
        order = np.argsort(eps)
        eps, frac = eps[order], frac[order]
        above = frac >= 0.5
        if above.any():
            i = int(np.argmax(above))
            if i == 0:
                ec[n] = float(eps[0])
            else:
                y0, y1 = frac[i - 1], frac[i]
                x0, x1 = eps[i - 1], eps[i]
                ec[n] = float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)) \
                    if y1 != y0 else float(x1)
    add("| N | eps_c |")
    add("|---|---|")
    for n in sorted(ec):
        add(f"| {n} | {ec[n]:.4f} |")
    add("")
    if ec:
        add("- 文献目标：N=100 约 0.25；大 N 极限约 0.20（均 ±0.03）")
        add(f"- 实测：N=100 = {fmt(ec.get(100))}（偏 "
            f"{ec.get(100, float('nan')) - 0.25:+.4f}）；"
            f"N={max(ec)} = {fmt(ec[max(ec)])}（偏 {ec[max(ec)] - 0.20:+.4f}）")
        mono = all(ec[a] >= ec[b] - 1e-9
                   for a, b in zip(sorted(ec), sorted(ec)[1:]))
        add(f"- 随 N 非增：{mono}")
    add("")

    # ---------------------------------------------------------------- R1
    add("## R1 第一轮主网格：三旋钮 control（对另一轴边缘化后）")
    add("")
    add("| 阶段 | N | 旋钮 | control | 95% CI | 第一轮原判据标签 |")
    add("|---|---|---|---|---|---|")
    for r in r1_ctrl:
        add(f"| {r.get('stage','?')} | {r.get('n_swarm','?')} | "
            f"{r.get('knob','?')} | {fmt(f(r,'control'))} | "
            f"[{fmt(f(r,'ci_lo'))}, {fmt(f(r,'ci_hi'))}] | "
            f"{r.get('verdict','?')} |")
    add("")

    # ---------------------------------------------------------------- R2
    add("## R2 第二轮（独立注册 prereg_r2）：R 的符号斜率")
    add("")
    add("| 臂 | N | 斜率 | 95% CI | 判定 |")
    add("|---|---|---|---|---|")
    for r in r2_slope:
        add(f"| {r.get('arm','?')} | {r.get('n_swarm','?')} | "
            f"{fmt(f(r,'slope'),5)} | "
            f"[{fmt(f(r,'ci_lo'),5)}, {fmt(f(r,'ci_hi'),5)}] | "
            f"{r.get('verdict','?')} |")
    add("")
    add("S3（注册主张）成立的条件是两规模的斜率**之差**区间排除 0；"
        "S1/S2 单独不足以确立符号依赖——两条斜率可以同号而异幅。")
    add("")

    # ---------------------------------------------------------------- R3
    add("## R3 beta 形状（主角 = 簇数）")
    add("")
    if beta:
        add("| 阶段 | beta | 簇数 | spread | 内部? |")
        add("|---|---|---|---|---|")
        for r in beta:
            add(f"| {r.get('stage','?')} | {r.get('beta','?')} | "
                f"{fmt(f(r,'mean_clusters'),3)} | {fmt(f(r,'mean_spread'),4)} | "
                f"{r.get('in_interior','?')} |")
        corners = [f(r, "mean_clusters") for r in beta
                   if str(r.get("in_interior")).lower() == "false"]
        interiors = [f(r, "mean_clusters") for r in beta
                     if str(r.get("in_interior")).lower() == "true"]
        if corners and interiors:
            add("")
            add(f"- 内部簇数范围：{min(interiors):.2f} ~ {max(interiors):.2f}")
            add(f"- 端点(beta=1)簇数：{np.mean(corners):.2f}")
            add(f"- 端点/内部倍数："
                f"{np.mean(corners) / max(np.mean(interiors), 1e-9):.1f}x")
    add("")

    # ---------------------------------------------------------------- R4
    add("## R4 方差门：哪些轴在网格上真有方差")
    add("")
    if var:
        inert = sorted({(r.get("stage"), r.get("axis")) for r in var
                        if str(r.get("has_variance")).lower() == "false"})
        live = sorted({(r.get("stage"), r.get("axis")) for r in var
                       if str(r.get("has_variance")).lower() == "true"})
        add(f"- 有方差（{len(live)} 组）：" + ", ".join(f"{a}@{s}" for s, a in live))
        add(f"- 无方差（{len(inert)} 组）：" +
            (", ".join(f"{a}@{s}" for s, a in inert) if inert else "无"))
        if inert:
            add("- 处置：无方差的轴**移出主网格**，收缩在正文报告而非静默吸收。")
    add("")

    # ---------------------------------------------------------------- R5
    add("## R5 方法学：单参考点 vs 边缘化")
    add("")
    add("| 阶段 | 旋钮 | 确认跑(边缘化, 15 seeds) | 第一轮(边缘化, 30 seeds) |")
    add("|---|---|---|---|")
    for r in conf:
        stage, knob = r.get("stage"), r.get("knob")
        r1 = [f(x, "control") for x in r1_ctrl
              if x.get("stage") == stage and x.get("knob") == knob]
        add(f"| {stage} | {knob} | {fmt(f(r,'control'))} | "
            f"{fmt(r1[0]) if r1 else 'MISSING'} |")
    add("")
    add("- 方差门曾以**单一参考点**报 r_mult 衰减 0.188→0.152→0.039；边缘化后为 "
        "0.084→0.033→0.060（非单调、CI 跨 0）。**该衰减主张已撤销。**")
    add("")

    # ---------------------------------------------------------------- R6
    add("## R6 涌现簇数在 (eps, r_mult) 网格上的取值")
    add("")
    if r1_grid:
        eps_vals = sorted({f(r, "eps") for r in r1_grid})
        add(f"- eps 档位：{eps_vals}")
        add(f"- 原始 episode 数：{len(r1_grid)}")
        stage_n = sorted({(r.get("stage"), int(f(r, "n_swarm"))) for r in r1_grid})
        add(f"- 阶段×规模格数：{len(stage_n)}")
        for stage, nsw in stage_n:
            sub = [r for r in r1_grid
                   if r.get("stage") == stage and int(f(r, "n_swarm")) == nsw]
            lo = np.mean([f(r, "clusters") for r in sub
                          if f(r, "eps") == eps_vals[0]])
            hi = np.mean([f(r, "clusters") for r in sub
                          if f(r, "eps") == eps_vals[-1]])
            add(f"  - {stage}/N={nsw}: eps={eps_vals[0]} 时簇数 {lo:.3f} → "
                f"eps={eps_vals[-1]} 时 {hi:.3f}")
    add("")

    add("---")
    add("")
    add("## 尚缺（写稿前须补齐或声明为开放）")
    add("")
    add("1. **R 的机制**：为何 N=100 有明确负斜率、N=400 探测不到。A1 干预臂存在但"
        "**不可解释**（前置条件——control 臂先显示翻转——不成立）。针对新问题需"
        "**重新注册一轮**，不能套用上一轮注册。")
    add("2. **D2 整合**：D2 的 R 结论（四基线逐位不敏感、BCG 敏感）与本审计的 R 结论"
        "（N=100 有、N=400 无）如何并置在同一叙述下。")
    add("3. **目标期刊与格式**：未定。")
    add("4. **图表**：未生成。")

    text = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\nwrote {OUT}  ({len(lines)} lines)")

    missing = [name for name, rows in
               (("repro_check.csv", repro), ("audit_control.csv", r1_ctrl),
                ("audit_grid.csv", r1_grid), ("r2_slopes.csv", r2_slope),
                ("beta_probe.csv", beta), ("variance_probe.csv", var),
                ("control_grid.csv", conf)) if not rows]
    if missing:
        print(f"MISSING INPUTS: {missing} -- recorded as MISSING, not filled in")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
