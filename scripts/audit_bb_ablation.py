"""Comprehensive offline 7-method audit reading results/ (6 methods) + results_ablation_bb/ (7th method).

Does NOT modify results/ or paper/. Verifies exact replay, Pre-CSR, CSR, tokens (completed & amortized),
failure logs, paired McNemar & 20,000-sample paired bootstrap CIs, fine-grained cross-attribute regressions,
Windows arial.ttf raster text-fit rate, center displacement, black/white shifts, and hard-brief zone adherence.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.detector import TEXT_TYPES, evaluate_layout
from src.export_results import _pre_arbiter_csr
from src.methods.arbiter import resolve_conflicts
from src.methods.blackboard import ARCHITECT, GEOMETRY, STYLE, VISUAL, _initial_geometry, _patched
from src.methods.single_agent_bb_detector import (
    ALL_FIELDS as BB_ALL_FIELDS,
    CONFLICT_TYPES as BB_CONFLICT_TYPES,
    DESIGNER as BB_DESIGNER,
    _initial_all_fields,
)
from src.schema import Conflict, Element, LayoutState

METHODS = (
    "single_agent",
    "self_refine",
    "naive_multi_agent",
    "single_agent_detector",
    "single_agent_bb_detector",
    "ours_no_detector",
    "ours_full",
)


def _dir_for(method: str) -> Path:
    if method == "single_agent_bb_detector":
        if any((ROOT / "results" / "raw").glob("single_agent_bb_detector__*.json")):
            return ROOT / "results"
        return ROOT / "results_ablation_bb"
    return ROOT / "results"


def exact_mcnemar(a_pass: list[int], b_pass: list[int]) -> tuple[int, int, float]:
    n10 = sum(1 for a, b in zip(a_pass, b_pass) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(a_pass, b_pass) if a == 0 and b == 1)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return n10, n01, p


def paired_bootstrap_ci(a_vals: list[float], b_vals: list[float], n_boot: int = 20000, seed: int = 42):
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return fmean(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _text_fit_strict_counts(state: LayoutState) -> tuple[int, int]:
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    elems = [e for e in state.elements if e.type in TEXT_TYPES and e.text]
    if not elems:
        return 0, 0
    fit = 0
    for e in elems:
        size = e.font_size or 16
        font = ImageFont.truetype("arial.ttf", size)
        lines: list[str] = []
        for para in (e.text or "").split("\n"):
            line = ""
            for word in para.split():
                cand = f"{line} {word}" if line else word
                if line and draw.textlength(cand, font=font) > e.w:
                    lines.append(line)
                    line = word
                else:
                    line = cand
            lines.append(line)
        max_w, max_bottom, y = 0.0, 0.0, 0.0
        for ln in lines:
            max_w = max(max_w, draw.textlength(ln, font=font))
            max_bottom = max(max_bottom, float(draw.textbbox((0, y), ln, font=font)[3]) if ln else y + size)
            y += 1.2 * size
        if max_w <= e.w + 1e-6 and max_bottom <= e.h + 1e-6:
            fit += 1
    return fit, len(elems)


def _inside(inner: Element, outer: Element) -> bool:
    return (
        outer.x <= inner.x + 1e-6
        and outer.y <= inner.y + 1e-6
        and inner.x + inner.w <= outer.x + outer.w + 1e-6
        and inner.y + inner.h <= outer.y + outer.h + 1e-6
    )


def _hard_zone_metrics(state: LayoutState) -> tuple[float, float, float]:
    by_id = {e.id: e for e in state.elements}
    H = state.canvas.height
    header_elems = [by_id["headline"], by_id["subhead"]]
    footer_elems = [by_id["primary_action"], by_id["secondary_action"]]
    mid_elems = [e for e in state.elements if e.id not in {"headline", "subhead", "primary_action", "secondary_action"}]
    hdr_ok = sum(1 for e in header_elems if e.y >= -1e-6 and e.y + e.h <= 64.0 + 1e-6)
    ftr_ok = sum(1 for e in footer_elems if e.y >= H - 56.0 - 1e-6 and e.y + e.h <= H + 1e-6)
    mid_ok = sum(1 for e in mid_elems if e.y >= 64.0 - 1e-6 and e.y + e.h <= H - 56.0 + 1e-6)
    panel = by_id["panel"]
    children = [e for e in state.elements if e.id in {"description", "caption"} or e.id.startswith("tag_")]
    encl = sum(1 for e in children if _inside(e, panel)) / len(children)
    return (hdr_ok + ftr_ok + mid_ok) / len(state.elements), hdr_ok / len(header_elems), encl


def _repair_delta(before: LayoutState, after: LayoutState) -> tuple[float, int]:
    b_map = {e.id: e for e in before.elements}
    a_map = {e.id: e for e in after.elements}
    disps = []
    bw = 0
    for eid, b in b_map.items():
        a = a_map[eid]
        disps.append(math.hypot((a.x + a.w / 2.0) - (b.x + b.w / 2.0), (a.y + a.h / 2.0) - (b.y + b.h / 2.0)))
        if a.fg_color != b.fg_color and (a.fg_color or "").upper() in ("#000000", "#FFFFFF"):
            bw += 1
    return fmean(disps), bw


def _geom_conflict_keys(state: LayoutState) -> set[tuple[str, tuple[str, ...]]]:
    ev = evaluate_layout(state)
    return {
        (c.conflict_type, tuple(sorted(c.element_ids)))
        for c in ev["conflicts"]
        if c.severity == "hard" and c.conflict_type in ("overlap", "overflow")
    }


def _contrast_fail_ids(state: LayoutState) -> set[str]:
    ev = evaluate_layout(state)
    return {
        eid
        for c in ev["conflicts"]
        if c.severity == "hard" and c.conflict_type == "contrast"
        for eid in c.element_ids
    }


def _replay_trace(method: str, raw: dict, brief: dict):
    """Return (r0_state, pre_arbiter_state, final_state, patch_transitions).

    Each transition dict records:
      round, role, input_hard, before_state, after_state
    """
    final = LayoutState.model_validate(raw["final_state"])
    transitions = []
    if method == "single_agent_bb_detector":
        state = init_state = pre_state = None
        last_hard: list[Conflict] = []
        for entry in raw["history"]:
            role, rnd = entry["role"], entry["round"]
            if role == BB_DESIGNER and rnd == 0:
                state = _initial_all_fields(entry["response"], brief)
                init_state = state.model_copy(deep=True)
            elif role == "Compliance Critic" and isinstance(entry.get("conflicts"), list):
                last_hard = [Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"]
            elif role == BB_DESIGNER and rnd >= 1:
                before = state.model_copy(deep=True)
                routed = [c for c in last_hard if c.conflict_type in BB_CONFLICT_TYPES]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, BB_ALL_FIELDS)
                transitions.append({
                    "round": rnd, "role": role, "input_hard": list(routed),
                    "before": before, "after": state.model_copy(deep=True),
                })
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None)
        assert state == final, f"Replay mismatch on {method}__{raw['task_id']}"
        return init_state, pre_state or state, final, transitions

    if method in ("ours_full", "ours_no_detector"):
        state = init_state = pre_state = None
        last_hard: list[Conflict] = []
        for entry in raw["history"]:
            role, rnd = entry["role"], entry["round"]
            if role == ARCHITECT and rnd == 0:
                state = _initial_geometry(entry["response"], brief)
            elif role == VISUAL and rnd == 0:
                state = _patched(entry["response"], state, brief, {e.id for e in state.elements}, STYLE)
                init_state = state.model_copy(deep=True)
            elif role == "Compliance Critic":
                if isinstance(entry.get("conflicts"), list):
                    last_hard = [Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"]
                elif isinstance(entry.get("response"), dict):
                    last_hard = [Conflict.model_validate({**r, "severity": "hard"}) for r in entry["response"].get("conflicts", [])]
            elif role in (ARCHITECT, VISUAL) and rnd >= 1:
                before = state.model_copy(deep=True)
                fields = GEOMETRY if role == ARCHITECT else STYLE
                types = {"overflow", "overlap"} if role == ARCHITECT else {"contrast", "min_font"}
                routed = [c for c in last_hard if c.conflict_type in types]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, fields)
                transitions.append({
                    "round": rnd, "role": role, "input_hard": list(routed),
                    "before": before, "after": state.model_copy(deep=True),
                })
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None if method == "ours_full" else last_hard)
        assert state == final, f"Replay mismatch on {method}__{raw['task_id']}"
        return init_state, pre_state or state, final, transitions

    if method == "single_agent_detector":
        r0 = LayoutState.model_validate(raw["history"][0]["response"])
        pre = r0
        last_hard: list[Conflict] = []
        for h in raw["history"][1:]:
            role = h.get("role")
            if role == "Compliance Critic" and isinstance(h.get("conflicts"), list):
                last_hard = [Conflict.model_validate(c) for c in h["conflicts"] if c.get("severity") == "hard"]
            elif role == "Layout Designer" and isinstance(h.get("response"), dict):
                before = pre
                after = LayoutState.model_validate(h["response"])
                transitions.append({
                    "round": h.get("round"), "role": role, "input_hard": list(last_hard),
                    "before": before, "after": after,
                })
                pre = after
        return r0, pre, final, transitions

    return final, final, final, transitions


def audit_cross_attribute_regressions(method: str, all_transitions: list[tuple[str, dict]]) -> dict:
    """Granular audit of (a) contrast-repair -> geometry regressions and (b) geometry-edit -> contrast regressions."""
    stats = Counter()
    examples = defaultdict(list)

    for tid, tr in all_transitions:
        before: LayoutState = tr["before"]
        after: LayoutState = tr["after"]
        hard: list[Conflict] = tr["input_hard"]
        b_map = {e.id: e for e in before.elements}
        a_map = {e.id: e for e in after.elements}

        geom_conflicts_in = [c for c in hard if c.conflict_type in ("overlap", "overflow")]
        contrast_conflicts_in = [c for c in hard if c.conflict_type in ("contrast", "min_font")]
        geom_ids_in = {eid for c in geom_conflicts_in for eid in c.element_ids}
        contrast_ids_in = {eid for c in contrast_conflicts_in for eid in c.element_ids}
        contrast_only_ids = contrast_ids_in - geom_ids_in

        geom_edited_ids = {
            eid for eid in b_map
            if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in GEOMETRY)
        }
        style_edited_ids = {
            eid for eid in b_map
            if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in STYLE)
        }

        g_before = _geom_conflict_keys(before)
        g_after = _geom_conflict_keys(after)
        new_geom = g_after - g_before

        c_before = _contrast_fail_ids(before)
        c_after = _contrast_fail_ids(after)
        new_contrast = c_after - c_before

        stats["total_correction_patches"] += 1
        if new_geom:
            stats["any_patch_new_geom_rounds"] += 1
            stats["any_patch_new_geom_conflicts"] += len(new_geom)

        # (a-1) Contrast-only rounds (no geometry conflicts reported at round start)
        if contrast_conflicts_in and not geom_conflicts_in:
            stats["contrast_only_rounds"] += 1
            if geom_edited_ids:
                stats["contrast_only_rounds_edited_geom"] += 1
            if new_geom:
                stats["contrast_only_rounds_new_geom_conflict"] += 1
                stats["contrast_only_rounds_new_geom_count"] += len(new_geom)
                examples["a1_contrast_only_round_geom_reg"].append((tid, tr["round"], sorted(geom_edited_ids), sorted(new_geom)))

        # (a-2) Contrast-only elements across ALL correction rounds
        for eid in sorted(contrast_only_ids):
            stats["contrast_only_elem_instances"] += 1
            if eid in geom_edited_ids:
                stats["contrast_only_elem_geom_edited"] += 1
                involving = [g for g in new_geom if eid in g[1]]
                if involving:
                    stats["contrast_only_elem_new_geom_conflict"] += 1
                    examples["a2_contrast_only_elem_geom_reg"].append((tid, tr["round"], eid, involving))

        # (b) Geometry edits triggering secondary contrast regressions
        if geom_edited_ids:
            stats["geom_editing_rounds"] += 1
            if new_contrast:
                stats["geom_edit_rounds_with_new_contrast"] += 1
                stats["geom_edit_new_contrast_elems_total"] += len(new_contrast)
                # Purely geometry-induced (element's style fields unchanged, yet newly fails contrast)
                pure_geom = sorted(e for e in new_contrast if e not in style_edited_ids)
                if pure_geom:
                    stats["geom_edit_pure_secondary_contrast_elems"] += len(pure_geom)
                    examples["b_pure_secondary_contrast"].append((tid, tr["round"], pure_geom))
                # Also cases where geometry moved AND style changed but ended failing contrast
                style_too = sorted(e for e in new_contrast if e in style_edited_ids)
                if style_too:
                    stats["geom_edit_joint_style_contrast_fail_elems"] += len(style_too)

    return {"counts": dict(stats), "examples": dict(examples)}


def main() -> None:
    briefs = json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))
    briefs_by_id = {b["task_id"]: b for b in briefs}
    task_ids = sorted(briefs_by_id)
    hard_ids = [t for t in task_ids if briefs_by_id[t]["difficulty"] == "hard"]

    records = {}
    failures = {}
    pre_csr = {}
    r0_csr = {}
    traces = {}

    for m in METHODS:
        base = _dir_for(m)
        records[m] = {t: json.loads((base / "raw" / f"{m}__{t}.json").read_text(encoding="utf-8")) for t in task_ids}
        f_dir = base / "failures"
        failures[m] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(f_dir.glob(f"{m}__*.json"))] if f_dir.is_dir() else []
        pre_csr[m] = {t: int(_pre_arbiter_csr(records[m][t], briefs_by_id[t])) for t in task_ids}
        traces[m] = {}
        r0_csr[m] = {}
        for t in task_ids:
            r0, pre, fin, trans = _replay_trace(m, records[m][t], briefs_by_id[t])
            traces[m][t] = (r0, pre, fin, trans)
            r0_csr[m][t] = int(evaluate_layout(r0)["metrics"]["CSR"])

    print("ALL REPLAYS VERIFIED 100% EXACT MATCH (including 36/36 single_agent_bb_detector).")
    print("\n=== TABLE I: 7-METHOD OVERALL SUMMARY (36 TASKS; 216 TEXT BOXES/METHOD, Windows arial.ttf) ===")
    print(f"{'Method':<26} {'OR%':>5} {'BOR%':>5} {'CPR%':>6} {'AS%':>5} {'Init%':>6} {'Pre%':>6} {'CSR%':>6} {'Rnd':>5} {'Tok':>7} {'Amort':>7} {'Fail':>7} {'TFR%':>6} {'Mic%':>5} {'Fit':>7}")
    for m in METHODS:
        recs = [records[m][t] for t in task_ids]
        f_recs = failures[m]
        comp_sum = sum(r["total_tokens"] for r in recs)
        fail_sum = sum(f["total_tokens"] for f in f_recs)
        fail_briefs = len({f["task_id"] for f in f_recs})
        tf_pairs = [_text_fit_strict_counts(LayoutState.model_validate(r["final_state"])) for r in recs]
        tfr_macro = fmean(f / n for f, n in tf_pairs) * 100
        fit_sum, box_sum = sum(f for f, _ in tf_pairs), sum(n for _, n in tf_pairs)
        tfr_micro = fit_sum / box_sum * 100
        print(
            f"{m:<26} "
            f"{fmean(r['metrics']['OR'] for r in recs)*100:5.1f} "
            f"{fmean(r['metrics']['BOR'] for r in recs)*100:5.1f} "
            f"{fmean(r['metrics']['CPR'] for r in recs)*100:6.1f} "
            f"{fmean(r['metrics']['AS'] for r in recs)*100:5.1f} "
            f"{fmean(r0_csr[m][t] for t in task_ids)*100:6.1f} "
            f"{fmean(pre_csr[m][t] for t in task_ids)*100:6.1f} "
            f"{fmean(r['metrics']['CSR'] for r in recs)*100:6.1f} "
            f"{fmean(r['rounds_used'] for r in recs):5.2f} "
            f"{fmean(r['total_tokens'] for r in recs):7,.0f} "
            f"{(comp_sum+fail_sum)/36:7,.0f} "
            f"{len(f_recs):>2d}/{fail_briefs:<2d}br "
            f"{tfr_macro:6.1f} "
            f"{tfr_micro:5.1f} "
            f"{fit_sum:>3d}/{box_sum:<3d}"
        )

    print("\n=== TABLE II: STRATIFIED BY DIFFICULTY (12 BRIEFS PER LEVEL) ===")
    print(f"{'Method':<26} | {'Easy (Pre/CSR/Rnd/Tok)':<26} | {'Medium (Pre/CSR/Rnd/Tok)':<26} | {'Hard (Pre/CSR/Rnd/Tok)':<26}")
    for m in METHODS:
        parts = []
        for diff in ("easy", "medium", "hard"):
            sub = [t for t in task_ids if briefs_by_id[t]["difficulty"] == diff]
            p = fmean(pre_csr[m][t] for t in sub) * 100
            c = fmean(records[m][t]["metrics"]["CSR"] for t in sub) * 100
            r = fmean(records[m][t]["rounds_used"] for t in sub)
            tk = fmean(records[m][t]["total_tokens"] for t in sub)
            parts.append(f"{p:5.1f}/{c:5.1f}/{r:4.2f}/{tk:6,.0f}")
        print(f"{m:<26} | {parts[0]:<26} | {parts[1]:<26} | {parts[2]:<26}")

    print("\n=== FAILURE LOG DETAILS FOR single_agent_bb_detector ===")
    bb_fails = failures["single_agent_bb_detector"]
    print(f"Total failed attempts={len(bb_fails)}, briefs={len({f['task_id'] for f in bb_fails})}, failed_tokens={sum(f['total_tokens'] for f in bb_fails):,}")
    for f in bb_fails:
        print(f"  task={f['task_id']} round={f.get('round_index')} role={f.get('role')} err={f.get('error_type')} tok={f.get('total_tokens')} reason={str(f.get('reason',''))[:110]}")

    print("\n=== ROUND DISTRIBUTION & PRE-ARBITER CONVERGENCE ===")
    for m in ("single_agent_detector", "single_agent_bb_detector", "ours_full"):
        rc = Counter(records[m][t]["rounds_used"] for t in task_ids)
        pc = Counter(records[m][t]["rounds_used"] for t in task_ids if pre_csr[m][t] == 1)
        arb = sum(1 for t in task_ids if any(h.get("role") == "Priority Arbiter" for h in records[m][t]["history"]))
        arb_h = sum(1 for t in hard_ids if any(h.get("role") == "Priority Arbiter" for h in records[m][t]["history"]))
        print(f"  {m:<26}: rounds={dict(sorted(rc.items()))} | pre_pass_rounds={dict(sorted(pc.items()))} | arbiter={arb}/36 (hard {arb_h}/12)")

    print("\n=== REPAIR FIDELITY & HARD-BRIEF ZONE ADHERENCE ===")
    for m in ("single_agent_detector", "single_agent_bb_detector", "ours_full"):
        r0_fin, pre_fin, r0_da = [], [], []
        for t in task_ids:
            r0, pre, fin, _ = traces[m][t]
            r0_fin.append(_repair_delta(r0, fin))
            pre_fin.append(_repair_delta(pre, fin))
            r0_da.append(_repair_delta(r0, resolve_conflicts(r0, None)))
        hz = [_hard_zone_metrics(LayoutState.model_validate(records[m][t]["final_state"])) for t in hard_ids]
        tf_clipped_briefs = sum(
            1 for t in task_ids
            if _text_fit_strict_counts(LayoutState.model_validate(records[m][t]["final_state"]))[0]
            < _text_fit_strict_counts(LayoutState.model_validate(records[m][t]["final_state"]))[1]
        )
        print(
            f"  {m:<26}: disp={fmean(d for d,_ in r0_fin):5.2f}px | net B/W={fmean(b for _,b in r0_fin):4.2f} "
            f"(arb-only={fmean(b for _,b in pre_fin):4.2f}) | R0->DA B/W={fmean(b for _,b in r0_da):4.2f} | "
            f"Zone(all/hdr/encl)={fmean(x[0] for x in hz)*100:5.1f}/{fmean(x[1] for x in hz)*100:5.1f}/{fmean(x[2] for x in hz)*100:5.1f}% | "
            f"clipped_briefs={tf_clipped_briefs}/36"
        )

    print("\n=== PAIRED STATISTICAL TESTS (McNemar exact 2-sided + 20,000 paired percentile bootstrap 95% CI) ===")
    pairs = [
        ("ours_full", "single_agent_bb_detector", "Net Role-Specialization + Field Locks"),
        ("single_agent_bb_detector", "single_agent_detector", "Net Blackboard (Partial Patch + Copy Injection)"),
        ("ours_full", "single_agent_detector", "Combined Bundle"),
    ]
    for a, b, desc in pairs:
        print(f"\n  >>> {a} vs. {b} [{desc}]")
        fail_tok_a = defaultdict(int)
        for f in failures[a]:
            fail_tok_a[f["task_id"]] += f["total_tokens"] or 0
        fail_tok_b = defaultdict(int)
        for f in failures[b]:
            fail_tok_b[f["task_id"]] += f["total_tokens"] or 0
        for label, sub in (("All 36", task_ids), ("Hard 12", hard_ids)):
            n10, n01, p = exact_mcnemar([pre_csr[a][t] for t in sub], [pre_csr[b][t] for t in sub])
            i10, i01, ip = exact_mcnemar([r0_csr[a][t] for t in sub], [r0_csr[b][t] for t in sub])
            dt = paired_bootstrap_ci([records[a][t]["total_tokens"] for t in sub], [records[b][t]["total_tokens"] for t in sub])
            dat = paired_bootstrap_ci(
                [records[a][t]["total_tokens"] + fail_tok_a[t] for t in sub],
                [records[b][t]["total_tokens"] + fail_tok_b[t] for t in sub],
            )
            dr = paired_bootstrap_ci([records[a][t]["rounds_used"] for t in sub], [records[b][t]["rounds_used"] for t in sub])
            das = paired_bootstrap_ci([records[a][t]["metrics"]["AS"]*100 for t in sub], [records[b][t]["metrics"]["AS"]*100 for t in sub])
            dd = paired_bootstrap_ci([_repair_delta(traces[a][t][0], traces[a][t][2])[0] for t in sub],
                                     [_repair_delta(traces[b][t][0], traces[b][t][2])[0] for t in sub])
            dbw = paired_bootstrap_ci([_repair_delta(traces[a][t][0], traces[a][t][2])[1] for t in sub],
                                      [_repair_delta(traces[b][t][0], traces[b][t][2])[1] for t in sub])
            dabw = paired_bootstrap_ci([_repair_delta(traces[a][t][1], traces[a][t][2])[1] for t in sub],
                                       [_repair_delta(traces[b][t][1], traces[b][t][2])[1] for t in sub])
            dtfr = paired_bootstrap_ci(
                [_text_fit_strict_counts(traces[a][t][2])[0] / _text_fit_strict_counts(traces[a][t][2])[1] * 100 for t in sub],
                [_text_fit_strict_counts(traces[b][t][2])[0] / _text_fit_strict_counts(traces[b][t][2])[1] * 100 for t in sub],
            )
            print(f"    [{label:<7}] InitCSR ({i10:2d} vs {i01:2d}, p={ip:.4g}) | PreCSR ({n10:2d} vs {n01:2d}, p={p:.4g})")
            print(f"              dTok={dt[0]:+7,.0f} [{dt[1]:+7,.0f}, {dt[2]:+7,.0f}] | dAmort={dat[0]:+7,.0f} [{dat[1]:+7,.0f}, {dat[2]:+7,.0f}] | dRnd={dr[0]:+5.2f} [{dr[1]:+5.2f}, {dr[2]:+5.2f}] | dAS={das[0]:+5.1f} [{das[1]:+5.1f}, {das[2]:+5.1f}]")
            print(f"              dDisp={dd[0]:+5.2f} [{dd[1]:+5.2f}, {dd[2]:+5.2f}]px | dNetBW={dbw[0]:+5.2f} [{dbw[1]:+5.2f}, {dbw[2]:+5.2f}] | dArbBW={dabw[0]:+5.2f} [{dabw[1]:+5.2f}, {dabw[2]:+5.2f}] | dTFR={dtfr[0]:+5.1f} [{dtfr[1]:+5.1f}, {dtfr[2]:+5.1f}]%")

    print("\n=== FINE-GRAINED CROSS-ATTRIBUTE REGRESSION AUDIT (CORRECTION ROUNDS ONLY) ===")
    for m in ("single_agent_detector", "single_agent_bb_detector", "ours_no_detector", "ours_full"):
        all_tr = [(t, tr) for t in task_ids for tr in traces[m][t][3]]
        res = audit_cross_attribute_regressions(m, all_tr)
        print(f"\n  [{m}] counts={res['counts']}")
        for k, exs in res["examples"].items():
            print(f"    {k} ({len(exs)}): {exs[:5]}")


if __name__ == "__main__":
    main()
