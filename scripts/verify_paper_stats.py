"""Reproduce all 7-method paper tables, pre-arbiter metrics, paired tests, and diagnostics offline."""

from __future__ import annotations

import argparse
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
    """Paired percentile bootstrap resampling over briefs (n_boot=20,000)."""
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return fmean(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _load_audit_font(size: int, font_path: str | None, allow_fallback: bool):
    target = font_path or "arial.ttf"
    try:
        return ImageFont.truetype(target, size), target
    except OSError as exc:
        if font_path or not allow_fallback:
            raise SystemExit(
                f"ERROR: Could not load required text-fit audit font '{target}'. "
                "The paper's raster text-fit audit used Windows 'arial.ttf'. "
                "Pass --font-path /path/to/arial.ttf to reproduce exact glyph metrics, "
                "or pass --allow-font-fallback to run with Pillow's built-in default font."
            ) from exc
        return ImageFont.load_default(size=size), "Pillow-default-fallback"


def _text_fit_strict_counts(
    state: LayoutState, *, font_path: str | None = None, allow_fallback: bool = False
) -> tuple[int, int, str]:
    """Return (fitting_boxes, total_text_boxes, font_label) using renderer.py greedy word wrap."""
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    elems = [e for e in state.elements if e.type in TEXT_TYPES and e.text]
    if not elems:
        return 0, 0, font_path or "arial.ttf"
    fit = 0
    used_label = font_path or "arial.ttf"
    for e in elems:
        size = e.font_size or 16
        font, used_label = _load_audit_font(size, font_path, allow_fallback)
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
    return fit, len(elems), used_label


def _inside(inner: Element, outer: Element) -> bool:
    return (
        outer.x <= inner.x + 1e-6
        and outer.y <= inner.y + 1e-6
        and inner.x + inner.w <= outer.x + outer.w + 1e-6
        and inner.y + inner.h <= outer.y + outer.h + 1e-6
    )


def _hard_zone_metrics(state: LayoutState) -> tuple[float, float, float, float, float]:
    """Return (all_elem, header, middle, footer, dark_card_enclosure) on a hard brief."""
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
    return (
        (hdr_ok + ftr_ok + mid_ok) / len(state.elements),
        hdr_ok / len(header_elems),
        mid_ok / len(mid_elems),
        ftr_ok / len(footer_elems),
        encl,
    )


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
    """Return (r0_state, pre_arbiter_state, final_state, patch_transitions)."""
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
                if "conflicts" in entry:
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


def _repair_delta(before: LayoutState, after: LayoutState) -> tuple[float, int]:
    """Return (mean_euclidean_center_disp_px, net_bw_fg_shifts) over all elements."""
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


def _audit_regressions(all_transitions: list[tuple[str, dict]]) -> dict[str, int]:
    stats = Counter()
    for _, tr in all_transitions:
        before, after, hard = tr["before"], tr["after"], tr["input_hard"]
        b_map = {e.id: e for e in before.elements}
        a_map = {e.id: e for e in after.elements}
        geom_in = [c for c in hard if c.conflict_type in ("overlap", "overflow")]
        contrast_in = [c for c in hard if c.conflict_type in ("contrast", "min_font")]
        geom_ids_in = {eid for c in geom_in for eid in c.element_ids}
        contrast_only_ids = {eid for c in contrast_in for eid in c.element_ids} - geom_ids_in
        geom_edited = {eid for eid in b_map if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in GEOMETRY)}
        style_edited = {eid for eid in b_map if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in STYLE)}
        new_geom = _geom_conflict_keys(after) - _geom_conflict_keys(before)
        new_contrast = _contrast_fail_ids(after) - _contrast_fail_ids(before)

        stats["patches"] += 1
        if contrast_in and not geom_in:
            stats["contrast_only_rounds"] += 1
            if geom_edited:
                stats["contrast_only_rounds_edited_geom"] += 1
            if new_geom:
                stats["contrast_only_rounds_new_geom"] += len(new_geom)
        for eid in contrast_only_ids:
            stats["contrast_only_elems"] += 1
            if eid in geom_edited:
                stats["contrast_only_elem_geom_edited"] += 1
                if any(eid in g[1] for g in new_geom):
                    stats["contrast_only_elem_new_geom"] += 1
        if geom_edited:
            stats["geom_edit_rounds"] += 1
            pure_c = [e for e in new_contrast if e not in style_edited]
            if pure_c:
                stats["geom_edit_pure_contrast_rounds"] += 1
                stats["geom_edit_pure_contrast_elems"] += len(pure_c)
    return dict(stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify CoLayout 7-method paper statistics offline.")
    parser.add_argument(
        "--font-path",
        default=None,
        help="Path to TrueType font for raster text-fit audit (paper used Windows arial.ttf).",
    )
    parser.add_argument(
        "--allow-font-fallback",
        action="store_true",
        help="Allow falling back to Pillow's default font if arial.ttf is unavailable (may alter text-fit counts).",
    )
    args = parser.parse_args(argv)

    _, font_label = _load_audit_font(16, args.font_path, args.allow_font_fallback)

    briefs = json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))
    briefs_by_id = {b["task_id"]: b for b in briefs}
    task_ids = sorted(briefs_by_id)
    hard_ids = [t for t in task_ids if briefs_by_id[t]["difficulty"] == "hard"]
    raw_dir = ROOT / "results" / "raw"
    fail_dir = ROOT / "results" / "failures"

    records = {m: {t: json.loads((raw_dir / f"{m}__{t}.json").read_text(encoding="utf-8")) for t in task_ids} for m in METHODS}
    failures = {m: [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fail_dir.glob(f"{m}__*.json"))] for m in METHODS}
    pre_csr = {m: {t: int(_pre_arbiter_csr(records[m][t], briefs_by_id[t])) for t in task_ids} for m in METHODS}
    traces = {m: {t: _replay_trace(m, records[m][t], briefs_by_id[t]) for t in task_ids} for m in METHODS}
    r0_csr = {m: {t: int(evaluate_layout(traces[m][t][0])["metrics"]["CSR"]) for t in task_ids} for m in METHODS}

    print(f"Text-fit audit font: {font_label}")
    print("=== TABLE I SUMMARY (7 METHODS, 36 TASKS = 252 LAYOUTS; 216 TEXT BOXES/METHOD) ===")
    print(f"{'Method':<26} {'OR%':>5} {'BOR%':>5} {'CPR%':>6} {'AS%':>5} {'Init%':>6} {'Pre%':>6} {'CSR%':>6} {'Rnd':>5} {'Tok':>7} {'TFR%':>6} {'Micro%':>7} {'Fit/216':>8} {'AmortTok':>9}")
    tot_comp, tot_fail, tot_fail_n = 0, 0, 0
    for m in METHODS:
        recs = [records[m][t] for t in task_ids]
        f_recs = failures[m]
        comp_sum = sum(r["total_tokens"] for r in recs)
        fail_sum = sum(f["total_tokens"] for f in f_recs)
        tot_comp += comp_sum
        tot_fail += fail_sum
        tot_fail_n += len(f_recs)
        tf_triples = [
            _text_fit_strict_counts(
                LayoutState.model_validate(r["final_state"]),
                font_path=args.font_path,
                allow_fallback=args.allow_font_fallback,
            )
            for r in recs
        ]
        tfr_macro = fmean(f / n for f, n, _ in tf_triples) * 100
        fit_sum, box_sum = sum(f for f, _, _ in tf_triples), sum(n for _, n, _ in tf_triples)
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
            f"{tfr_macro:6.1f} "
            f"{tfr_micro:7.1f} "
            f"{fit_sum:>3d}/{box_sum:<3d} "
            f"{(comp_sum + fail_sum)/36:9,.0f}"
        )
    print(f"Total tokens: completed={tot_comp:,}, failed={tot_fail:,} ({tot_fail_n} attempts), grand={tot_comp+tot_fail:,}")

    print("\n=== PAIRED STATISTICAL TESTS (McNemar exact 2-sided + 20,000 bootstrap 95% CI) ===")
    pairs = (
        ("ours_full", "ours_no_detector"),
        ("ours_full", "single_agent_bb_detector"),
        ("single_agent_bb_detector", "single_agent_detector"),
        ("ours_full", "single_agent_detector"),
    )
    for a, b in pairs:
        ft_a = defaultdict(int)
        for f in failures[a]:
            ft_a[f["task_id"]] += f["total_tokens"] or 0
        ft_b = defaultdict(int)
        for f in failures[b]:
            ft_b[f["task_id"]] += f["total_tokens"] or 0
        for label, subset in (("All 36", task_ids), ("Hard 12", hard_ids)):
            i10, i01, ip = exact_mcnemar([r0_csr[a][t] for t in subset], [r0_csr[b][t] for t in subset])
            n10, n01, p = exact_mcnemar([pre_csr[a][t] for t in subset], [pre_csr[b][t] for t in subset])
            dt = paired_bootstrap_ci([records[a][t]["total_tokens"] for t in subset], [records[b][t]["total_tokens"] for t in subset])
            dat = paired_bootstrap_ci(
                [records[a][t]["total_tokens"] + ft_a[t] for t in subset],
                [records[b][t]["total_tokens"] + ft_b[t] for t in subset],
            )
            dr = paired_bootstrap_ci([records[a][t]["rounds_used"] for t in subset], [records[b][t]["rounds_used"] for t in subset])
            dd = paired_bootstrap_ci(
                [_repair_delta(traces[a][t][0], traces[a][t][2])[0] for t in subset],
                [_repair_delta(traces[b][t][0], traces[b][t][2])[0] for t in subset],
            )
            dbw = paired_bootstrap_ci(
                [_repair_delta(traces[a][t][0], traces[a][t][2])[1] for t in subset],
                [_repair_delta(traces[b][t][0], traces[b][t][2])[1] for t in subset],
            )
            dtfr = paired_bootstrap_ci(
                [_text_fit_strict_counts(traces[a][t][2], font_path=args.font_path, allow_fallback=args.allow_font_fallback)[0]
                 / _text_fit_strict_counts(traces[a][t][2], font_path=args.font_path, allow_fallback=args.allow_font_fallback)[1] * 100 for t in subset],
                [_text_fit_strict_counts(traces[b][t][2], font_path=args.font_path, allow_fallback=args.allow_font_fallback)[0]
                 / _text_fit_strict_counts(traces[b][t][2], font_path=args.font_path, allow_fallback=args.allow_font_fallback)[1] * 100 for t in subset],
            )
            print(
                f"  [{label:<7}] {a} vs {b}: Init({i10}v{i01}, p={ip:.3g}) | Pre({n10}v{n01}, p={p:.3g}) | "
                f"dTok={dt[0]:+,.0f} [{dt[1]:+,.0f}, {dt[2]:+,.0f}] | dAmort={dat[0]:+,.0f} [{dat[1]:+,.0f}, {dat[2]:+,.0f}] | "
                f"dRnd={dr[0]:+.2f} [{dr[1]:+.2f}, {dr[2]:+.2f}] | dDisp={dd[0]:+.2f} [{dd[1]:+.2f}, {dd[2]:+.2f}] | "
                f"dBW={dbw[0]:+.2f} [{dbw[1]:+.2f}, {dbw[2]:+.2f}] | dTFR={dtfr[0]:+.1f} [{dtfr[1]:+.1f}, {dtfr[2]:+.1f}]%"
            )

    print("\n=== REPAIR FIDELITY, HARD-BRIEF ZONES & CROSS-ATTRIBUTE REGRESSIONS ===")
    for m in ("ours_full", "single_agent_bb_detector", "single_agent_detector"):
        r0_fin, pre_fin, r0_da = [], [], []
        arb_briefs = geom_briefs = crop_elems = shift_elems = recolor_elems = 0
        pre_clip_boxes = arb_added_clips = 0
        for t in task_ids:
            r0, pre, fin, _ = traces[m][t]
            r0_fin.append(_repair_delta(r0, fin))
            pre_fin.append(_repair_delta(pre, fin))
            r0_da.append(_repair_delta(r0, resolve_conflicts(r0, None)))
            pf, pt, _ = _text_fit_strict_counts(pre, font_path=args.font_path, allow_fallback=args.allow_font_fallback)
            ff, ft, _ = _text_fit_strict_counts(fin, font_path=args.font_path, allow_fallback=args.allow_font_fallback)
            pre_clip_boxes += (pt - pf)
            arb_added_clips += max(0, (ft - ff) - (pt - pf))
            if any(h.get("role") == "Priority Arbiter" for h in records[m][t]["history"]):
                arb_briefs += 1
                pm = {e.id: e for e in pre.elements}
                fm = {e.id: e for e in fin.elements}
                g_any = False
                for eid in pm:
                    a, b = pm[eid], fm[eid]
                    if a.w != b.w or a.h != b.h:
                        crop_elems += 1; g_any = True
                    elif a.x != b.x or a.y != b.y:
                        shift_elems += 1; g_any = True
                    if a.fg_color != b.fg_color:
                        recolor_elems += 1
                if g_any:
                    geom_briefs += 1
        print(
            f"  {m:<26}: R0->Final disp={fmean(d for d, _ in r0_fin):.2f}px, "
            f"net B/W={fmean(b for _, b in r0_fin):.2f}/brief (arbiter-only={fmean(b for _, b in pre_fin):.2f}), "
            f"R0->DirectArb B/W={fmean(b for _, b in r0_da):.2f}/brief | "
            f"arbiter={arb_briefs}/36 (geom_br={geom_briefs}, crop={crop_elems}, shift={shift_elems}, recolor={recolor_elems}) | "
            f"clipped={pre_clip_boxes}+arb{arb_added_clips}"
        )
    print("  LLM calls/run: " + ", ".join(
        f"{m}={fmean(sum(1 for h in records[m][t]['history'] if 'token_usage' in h) for t in task_ids):.2f}"
        f"({sum(sum(1 for h in records[m][t]['history'] if 'token_usage' in h) for t in task_ids)})"
        for m in METHODS
    ))
    sa_da = [_repair_delta(LayoutState.model_validate(records["single_agent"][t]["final_state"]),
                           resolve_conflicts(LayoutState.model_validate(records["single_agent"][t]["final_state"]), None))
             for t in task_ids]
    print(f"  {'single_agent+DirectArb':<26}: R0->DirectArb B/W={fmean(b for _, b in sa_da):.2f}/brief")
    for m in METHODS:
        hz = [_hard_zone_metrics(LayoutState.model_validate(records[m][t]["final_state"])) for t in hard_ids]
        print(
            f"  Zone[{m:<24}]: all={fmean(x[0] for x in hz)*100:5.1f}%, hdr={fmean(x[1] for x in hz)*100:5.1f}%, "
            f"mid={fmean(x[2] for x in hz)*100:5.1f}%, ftr={fmean(x[3] for x in hz)*100:5.1f}%, encl={fmean(x[4] for x in hz)*100:5.1f}%"
        )
    for m in ("single_agent_detector", "single_agent_bb_detector", "ours_no_detector", "ours_full"):
        all_tr = [(t, tr) for t in task_ids for tr in traces[m][t][3]]
        print(f"  Regressions[{m:<24}]: {_audit_regressions(all_tr)}")


if __name__ == "__main__":
    main()
