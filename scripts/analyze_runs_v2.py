"""Pre-registered Stage-3 analysis script implementing runs_v2/ANALYSIS_PLAN.md (incl. Amendments 1-2).

Supports dual failure accounting:
  1. Primary (Intention-to-Treat / ITT): Residual failed runs (missing from raw/
     after max 3 whole-run attempts) count as R0-CSR=0, Pre-CSR=0, CSR=0, with
     tokens = sum of all failed attempts for that (method, seed, brief).
  2. Sensitivity (Complete-Case): Restricts summaries and paired cluster-bootstrap
     contrasts to completed runs only.
Both regimes are ALWAYS reported side by side.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.detector import TEXT_TYPES, _luminance, effective_background, evaluate_layout
from src.export_results import _pre_arbiter_csr
from src.generate_briefs import PALETTES
from src.methods.arbiter import resolve_conflicts
from src.methods.blackboard import (
    ARCHITECT,
    GEOMETRY,
    STYLE,
    VISUAL,
    _initial_geometry,
    _patched,
)
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
    "sa_twostage_bb_det",
    "ours_no_detector",
    "ours_full",
)
DET_METHODS = (
    "single_agent_detector",
    "single_agent_bb_detector",
    "sa_twostage_bb_det",
    "ours_full",
)
BB_METHODS = (
    "single_agent_bb_detector",
    "sa_twostage_bb_det",
    "ours_no_detector",
    "ours_full",
)
DIFFICULTIES = ("easy", "medium", "hard")

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_DRAW = ImageDraw.Draw(Image.new("RGB", (10, 10)))


def _get_font(size: int, font_path: str | None) -> ImageFont.FreeTypeFont:
    target = font_path or "arial.ttf"
    key = (target, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(target, size)
    return _FONT_CACHE[key]


def text_fit_counts(state: LayoutState, font_path: str | None = None) -> tuple[int, int]:
    elems = [e for e in state.elements if e.type in TEXT_TYPES and e.text]
    if not elems:
        return 0, 0
    fit = 0
    for e in elems:
        size = e.font_size or 16
        font = _get_font(size, font_path)
        lines: list[str] = []
        for para in (e.text or "").split("\n"):
            line = ""
            for word in para.split():
                cand = f"{line} {word}" if line else word
                if line and _DRAW.textlength(cand, font=font) > e.w:
                    lines.append(line)
                    line = word
                else:
                    line = cand
            lines.append(line)
        max_w = max_bottom = y = 0.0
        for ln in lines:
            max_w = max(max_w, _DRAW.textlength(ln, font=font))
            max_bottom = max(
                max_bottom,
                float(_DRAW.textbbox((0, y), ln, font=font)[3]) if ln else y + size,
            )
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


def hard_zone_metrics(state: LayoutState) -> tuple[float, float, float, float, float]:
    by_id = {e.id: e for e in state.elements}
    H = state.canvas.height
    header_elems = [by_id["headline"], by_id["subhead"]]
    footer_elems = [by_id["primary_action"], by_id["secondary_action"]]
    mid_elems = [
        e for e in state.elements
        if e.id not in {"headline", "subhead", "primary_action", "secondary_action"}
    ]
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


def repair_delta(before: LayoutState, after: LayoutState) -> tuple[float, int]:
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


def box_size_delta(before: LayoutState, after: LayoutState) -> float:
    b_map = {e.id: e for e in before.elements}
    a_map = {e.id: e for e in after.elements}
    return fmean(abs(a_map[eid].w - b.w) + abs(a_map[eid].h - b.h) for eid, b in b_map.items())


def has_straddling_contrast_fp(state: LayoutState) -> bool:
    """Return True if any text box passes current contrast but fails worst-case contrast on a partially overlapped card."""
    elems = state.elements
    for idx, e in enumerate(elems):
        if e.type not in TEXT_TYPES or e.bg_color is not None:
            continue
        e_area = max(0.0, e.w) * max(0.0, e.h)
        if e_area == 0.0:
            continue
        fg_l = _luminance(e.fg_color or "#000000")
        thr = 3.0 if (e.font_size or 0) >= 24 else 4.5
        ebg = effective_background(state, idx)
        ebg_l = _luminance(ebg)
        cur_ratio = (max(fg_l, ebg_l) + 0.05) / (min(fg_l, ebg_l) + 0.05)
        if cur_ratio < thr:
            continue
        e_ord = (e.z_index, idx)
        for c_idx, c in enumerate(elems):
            if c.type != "card" or c.bg_color is None or not ((c.z_index, c_idx) < e_ord):
                continue
            ix = max(0.0, min(e.x + e.w, c.x + c.w) - max(e.x, c.x))
            iy = max(0.0, min(e.y + e.h, c.y + c.h) - max(e.y, c.y))
            inter = ix * iy
            if 0.0 < inter < e_area:
                c_l = _luminance(c.bg_color)
                ratio = (max(fg_l, c_l) + 0.05) / (min(fg_l, c_l) + 0.05)
                if ratio < thr:
                    return True
    return False


def geom_conflict_keys(state: LayoutState) -> set[tuple[str, tuple[str, ...]]]:
    ev = evaluate_layout(state)
    return {
        (c.conflict_type, tuple(sorted(c.element_ids)))
        for c in ev["conflicts"]
        if c.severity == "hard" and c.conflict_type in ("overlap", "overflow")
    }


def contrast_fail_ids(state: LayoutState) -> set[str]:
    ev = evaluate_layout(state)
    return {
        eid
        for c in ev["conflicts"]
        if c.severity == "hard" and c.conflict_type == "contrast"
        for eid in c.element_ids
    }


def replay_v2_trace(method: str, raw: dict, brief: dict):
    final = LayoutState.model_validate(raw["final_state"])
    transitions = []
    if method == "single_agent_bb_detector":
        state = init_state = pre_state = None
        last_hard: list[Conflict] = []
        for entry in raw["history"]:
            role, rnd = entry["role"], entry["round"]
            if role == BB_DESIGNER and rnd == 0:
                state = _initial_all_fields(entry["response"], brief, strict_keys=True)
                init_state = state.model_copy(deep=True)
            elif role == "Compliance Critic" and isinstance(entry.get("conflicts"), list):
                last_hard = [Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"]
            elif role == BB_DESIGNER and rnd >= 1:
                before = state.model_copy(deep=True)
                routed = [c for c in last_hard if c.conflict_type in BB_CONFLICT_TYPES]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, BB_ALL_FIELDS, strict_keys=True)
                transitions.append({
                    "round": rnd, "role": role, "input_hard": list(routed),
                    "before": before, "after": state.model_copy(deep=True),
                })
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None)
        assert state == final, f"Strict replay mismatch on {method}__{raw['task_id']}"
        return init_state, pre_state or state, final, transitions

    if method == "sa_twostage_bb_det":
        state = init_state = pre_state = None
        last_hard: list[Conflict] = []
        r0_calls = 0
        for entry in raw["history"]:
            role, rnd = entry["role"], entry["round"]
            if role == BB_DESIGNER and rnd == 0:
                stage = entry.get("init_stage")
                if stage == "geometry" or (stage is None and r0_calls == 0):
                    state = _initial_geometry(entry["response"], brief, strict_keys=True)
                else:
                    state = _patched(
                        entry["response"], state, brief, {e.id for e in state.elements},
                        STYLE, strict_keys=True, require_initial_style=True,
                    )
                    init_state = state.model_copy(deep=True)
                r0_calls += 1
            elif role == "Compliance Critic" and isinstance(entry.get("conflicts"), list):
                last_hard = [Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"]
            elif role == BB_DESIGNER and rnd >= 1:
                before = state.model_copy(deep=True)
                routed = [c for c in last_hard if c.conflict_type in BB_CONFLICT_TYPES]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, BB_ALL_FIELDS, strict_keys=True)
                transitions.append({
                    "round": rnd, "role": role, "input_hard": list(routed),
                    "before": before, "after": state.model_copy(deep=True),
                })
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None)
        assert state == final, f"Strict replay mismatch on {method}__{raw['task_id']}"
        return init_state, pre_state or state, final, transitions

    if method in ("ours_full", "ours_no_detector"):
        state = init_state = pre_state = None
        last_hard: list[Conflict] = []
        for entry in raw["history"]:
            role, rnd = entry["role"], entry["round"]
            if role == ARCHITECT and rnd == 0:
                state = _initial_geometry(entry["response"], brief, strict_keys=True)
            elif role == VISUAL and rnd == 0:
                state = _patched(
                    entry["response"], state, brief, {e.id for e in state.elements},
                    STYLE, strict_keys=True, require_initial_style=True,
                )
                init_state = state.model_copy(deep=True)
            elif role == "Compliance Critic":
                if "conflicts" in entry:
                    last_hard = [Conflict.model_validate(c) for c in entry["conflicts"] if c.get("severity") == "hard"]
                elif isinstance(entry.get("response"), dict):
                    last_hard = [
                        Conflict.model_validate({**r, "severity": "hard"})
                        for r in entry["response"].get("conflicts", [])
                    ]
            elif role in (ARCHITECT, VISUAL) and rnd >= 1:
                before = state.model_copy(deep=True)
                fields = GEOMETRY if role == ARCHITECT else STYLE
                types = {"overflow", "overlap"} if role == ARCHITECT else {"contrast", "min_font"}
                routed = [c for c in last_hard if c.conflict_type in types]
                ids = {i for c in routed for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, fields, strict_keys=True)
                transitions.append({
                    "round": rnd, "role": role, "input_hard": list(routed),
                    "before": before, "after": state.model_copy(deep=True),
                })
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None if method == "ours_full" else last_hard)
        assert state == final, f"Strict replay mismatch on {method}__{raw['task_id']}"
        return init_state, pre_state or state, final, transitions

    if method == "single_agent_detector":
        r0 = LayoutState.model_validate(raw["history"][0]["response"])
        pre = r0
        last_hard = []
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

    if method == "self_refine":
        r0 = LayoutState.model_validate(raw["history"][0]["response"])
        return r0, final, final, transitions

    if method == "naive_multi_agent":
        r0 = None
        for h in raw["history"]:
            if h["round"] == 1 and isinstance(h.get("response"), dict) and "layout" in h["response"]:
                r0 = LayoutState.model_validate(h["response"]["layout"])
        return r0 or final, final, final, transitions

    return final, final, final, transitions


def audit_regressions(all_transitions: list[dict]) -> dict[str, int]:
    stats = Counter()
    for tr in all_transitions:
        before, after, hard = tr["before"], tr["after"], tr["input_hard"]
        b_map = {e.id: e for e in before.elements}
        a_map = {e.id: e for e in after.elements}
        geom_in = [c for c in hard if c.conflict_type in ("overlap", "overflow")]
        contrast_in = [c for c in hard if c.conflict_type in ("contrast", "min_font")]
        geom_ids_in = {eid for c in geom_in for eid in c.element_ids}
        contrast_only_ids = {eid for c in contrast_in for eid in c.element_ids} - geom_ids_in
        geom_edited = {eid for eid in b_map if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in GEOMETRY)}
        style_edited = {eid for eid in b_map if any(getattr(b_map[eid], f) != getattr(a_map[eid], f) for f in STYLE)}
        new_geom = geom_conflict_keys(after) - geom_conflict_keys(before)
        new_contrast = contrast_fail_ids(after) - contrast_fail_ids(before)

        stats["patches"] += 1
        for eid in contrast_only_ids:
            stats["contrast_only_elems"] += 1
            if eid in geom_edited:
                stats["contrast_only_elem_geom_edited"] += 1
                if any(eid in g[1] for g in new_geom):
                    stats["c_to_g"] += 1
        if geom_edited:
            stats["geom_edit_rounds"] += 1
            pure_c = [e for e in new_contrast if e not in style_edited]
            if pure_c:
                stats["g_to_c_rounds"] += 1
                stats["g_to_c"] += len(pure_c)
    return dict(stats)


def brief_clustered_bootstrap(
    diffs: list[float], n_boot: int = 20000, seed: int = 42
) -> tuple[float, float, float]:
    """Brief-clustered paired percentile bootstrap (B=20,000, seed=42)."""
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return fmean(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def analyze(runs_dir: Path, seeds: list[int], font_path: str | None = None) -> dict:
    briefs = json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))
    briefs_by_id = {b["task_id"]: b for b in briefs}
    task_ids = sorted(briefs_by_id)
    by_diff = {d: [t for t in task_ids if briefs_by_id[t]["difficulty"] == d] for d in DIFFICULTIES}

    per_run: dict[tuple[str, int, str], dict] = {}
    fail_logs: dict[tuple[str, int], list[dict]] = defaultdict(list)
    fail_tok_by_pair: dict[tuple[str, int, str], int] = defaultdict(int)
    fail_rej_by_pair: dict[tuple[str, int, str], int] = defaultdict(int)
    fail_cnt_by_pair: dict[tuple[str, int, str], int] = defaultdict(int)

    resolved_models: set[str] = set()
    fingerprints: set[str] = set()
    timestamps: list[str] = []
    metric_mismatches = 0
    strict_replays = 0

    for s in seeds:
        sdir = runs_dir / f"seed_{s}"
        raw_dir = sdir / "raw"
        fail_dir = sdir / "failures"
        for m in METHODS:
            if fail_dir.is_dir():
                for fp in sorted(fail_dir.glob(f"{m}__*.json")):
                    frec = json.loads(fp.read_text(encoding="utf-8"))
                    fail_logs[m, s].append(frec)
                    tid = frec["task_id"]
                    fail_tok_by_pair[m, s, tid] += frec.get("total_tokens") or 0
                    fail_rej_by_pair[m, s, tid] += frec.get("format_rejections") or 0
                    fail_cnt_by_pair[m, s, tid] += 1
                    resolved_models.update(frec.get("resolved_models") or [])
                    fingerprints.update(frec.get("system_fingerprints") or [])
                    if frec.get("timestamp_utc"):
                        timestamps.append(frec["timestamp_utc"])
            for t in task_ids:
                p = raw_dir / f"{m}__{t}.json"
                if not p.is_file():
                    # Residual failed run after max 3 whole-run attempts (ITT record).
                    per_run[m, s, t] = {
                        "completed": False,
                        "r0_csr": 0.0,
                        "pre_csr": 0.0,
                        "CSR": 0.0,
                        "tokens": None,
                        "amort_tokens": fail_tok_by_pair[m, s, t],
                        "format_rejections": fail_rej_by_pair[m, s, t],
                        "failed_attempts": fail_cnt_by_pair[m, s, t],
                        "transitions": [],
                    }
                    continue
                rec = json.loads(p.read_text(encoding="utf-8"))
                resolved_models.update(rec.get("resolved_models") or [])
                fingerprints.update(rec.get("system_fingerprints") or [])
                if rec.get("timestamp_utc"):
                    timestamps.append(rec["timestamp_utc"])
                r0, pre, fin, trans = replay_v2_trace(m, rec, briefs_by_id[t])
                if m in BB_METHODS:
                    strict_replays += 1
                ev_fin = evaluate_layout(fin)
                for k, v in ev_fin["metrics"].items():
                    if not math.isclose(float(rec["metrics"][k]), v, rel_tol=1e-9, abs_tol=1e-12):
                        metric_mismatches += 1
                ev0 = evaluate_layout(r0)
                pre_ok = int(_pre_arbiter_csr(rec, briefs_by_id[t]))
                r0_ok = int(ev0["metrics"]["CSR"])
                fit, n_txt = text_fit_counts(fin, font_path)
                fit_pre, n_txt_pre = text_fit_counts(pre, font_path)
                allfit_fin = int(n_txt == 0 or fit == n_txt)
                allfit_pre = int(n_txt_pre == 0 or fit_pre == n_txt_pre)
                strad_fp_pre = int(has_straddling_contrast_fp(pre))
                strad_fp_fin = int(has_straddling_contrast_fp(fin))
                disp, net_bw = repair_delta(r0, fin)
                dwh = box_size_delta(r0, fin)
                _, arb_bw = repair_delta(pre, fin)
                calls = sum(1 for h in rec["history"] if "token_usage" in h)
                arb_triggered = any(h.get("role") == "Priority Arbiter" for h in rec["history"])
                arb_crops = arb_shifts = arb_recolors = 0
                arb_geom = False
                if arb_triggered:
                    pm = {e.id: e for e in pre.elements}
                    fm = {e.id: e for e in fin.elements}
                    for eid in pm:
                        a, b = pm[eid], fm[eid]
                        if a.w != b.w or a.h != b.h:
                            arb_crops += 1; arb_geom = True
                        elif a.x != b.x or a.y != b.y:
                            arb_shifts += 1; arb_geom = True
                        if a.fg_color != b.fg_color:
                            arb_recolors += 1

                panel = next((e for e in fin.elements if e.type == "card"), None)
                card_info = None
                if panel is not None:
                    exp_hex = PALETTES[briefs_by_id[t]["domain"]][1].upper()
                    card_info = {
                        "colored": int(panel.bg_color is not None),
                        "dark": int(panel.bg_color is not None and _luminance(panel.bg_color) < 0.2),
                        "exact": int(panel.bg_color is not None and panel.bg_color.upper() == exp_hex),
                    }

                bt_tot = bt_set = bt_fail = 0
                c0_ids = {c.element_ids[0] for c in ev0["conflicts"] if c.conflict_type == "contrast"}
                for e in r0.elements:
                    if e.type in ("button", "tag"):
                        bt_tot += 1
                        if e.bg_color is not None:
                            bt_set += 1
                            if e.id in c0_ids:
                                bt_fail += 1

                hz = hard_zone_metrics(fin) if briefs_by_id[t]["difficulty"] == "hard" else None

                per_run[m, s, t] = {
                    "completed": True,
                    "OR": rec["metrics"]["OR"] * 100,
                    "BOR": rec["metrics"]["BOR"] * 100,
                    "CPR": rec["metrics"]["CPR"] * 100,
                    "AS": rec["metrics"]["AS"] * 100,
                    "CSR": rec["metrics"]["CSR"] * 100,
                    "r0_csr": r0_ok * 100,
                    "pre_csr": pre_ok * 100,
                    "rounds": rec["rounds_used"],
                    "calls": calls,
                    "tokens": rec["total_tokens"],
                    "amort_tokens": rec["total_tokens"] + fail_tok_by_pair[m, s, t],
                    "format_rejections": (rec.get("format_rejections") or 0) + fail_rej_by_pair[m, s, t],
                    "failed_attempts": fail_cnt_by_pair[m, s, t],
                    "tfr": fit / n_txt * 100,
                    "fit": fit,
                    "n_txt": n_txt,
                    "allfit_fin": allfit_fin,
                    "allfit_pre": allfit_pre,
                    "strad_fp_pre": strad_fp_pre,
                    "strad_fp_fin": strad_fp_fin,
                    "disp": disp,
                    "dwh": dwh,
                    "net_bw": net_bw,
                    "arb_bw": arb_bw,
                    "arb_triggered": int(arb_triggered),
                    "arb_geom": int(arb_geom),
                    "arb_crops": arb_crops,
                    "arb_shifts": arb_shifts,
                    "arb_recolors": arb_recolors,
                    "card": card_info,
                    "bt_tot": bt_tot,
                    "bt_set": bt_set,
                    "bt_fail": bt_fail,
                    "hz": hz,
                    "transitions": trans,
                    "r0_contrast": sum(1 for c in ev0["conflicts"] if c.conflict_type == "contrast"),
                    "r0_overlap": sum(1 for c in ev0["conflicts"] if c.conflict_type == "overlap"),
                    "r0_overflow": sum(1 for c in ev0["conflicts"] if c.conflict_type == "overflow"),
                }

    method_summary = {}
    per_seed_summary = {}
    for m in METHODS:
        all_rows = [per_run[m, s, t] for s in seeds for t in task_ids]
        comp_rows = [r for r in all_rows if r["completed"]]
        comp_hard = [per_run[m, s, t] for s in seeds for t in by_diff["hard"] if per_run[m, s, t]["completed"]]
        comp_med = [per_run[m, s, t] for s in seeds for t in by_diff["medium"] if per_run[m, s, t]["completed"]]

        # ITT per-seed means (all 36 briefs)
        seed_pre_itt = [fmean(per_run[m, s, t]["pre_csr"] for t in task_ids) for s in seeds]
        seed_r0_itt = [fmean(per_run[m, s, t]["r0_csr"] for t in task_ids) for s in seeds]
        seed_amort_itt = [fmean(per_run[m, s, t]["amort_tokens"] for t in task_ids) for s in seeds]

        # Complete-case per-seed means
        seed_pre_cc = [fmean(per_run[m, s, t]["pre_csr"] for t in task_ids if per_run[m, s, t]["completed"]) for s in seeds]
        seed_tok_cc = [fmean(per_run[m, s, t]["tokens"] for t in task_ids if per_run[m, s, t]["completed"]) for s in seeds]

        r0_fails_cc = [r for r in comp_rows if r["r0_csr"] == 0]
        r0_conv_cc = sum(1 for r in r0_fails_cc if r["pre_csr"] == 100)
        r0_fails_itt = [r for r in all_rows if r["r0_csr"] == 0]
        r0_conv_itt = sum(1 for r in r0_fails_itt if r["pre_csr"] == 100)

        tot_fail_attempts = sum(len(fail_logs[m, s]) for s in seeds)
        n_comp = len(comp_rows)
        n_sched = len(all_rows)
        n_resid = n_sched - n_comp
        attempt_fail_rate = tot_fail_attempts / (n_comp + tot_fail_attempts) * 100 if (n_comp + tot_fail_attempts) else 0.0
        reg = audit_regressions([tr for r in comp_rows for tr in r["transitions"]])

        method_summary[m] = {
            "n_scheduled": n_sched,
            "n_completed": n_comp,
            "n_residual_failed": n_resid,
            "failed_attempts": tot_fail_attempts,
            "attempt_fail_rate_pct": attempt_fail_rate,
            "format_rejections": sum(r["format_rejections"] for r in all_rows),
            # Primary ITT endpoints
            "itt_r0_csr": fmean(r["r0_csr"] for r in all_rows),
            "itt_r0_csr_sd": stdev(seed_r0_itt) if len(seeds) > 1 else 0.0,
            "itt_pre_csr": fmean(r["pre_csr"] for r in all_rows),
            "itt_pre_csr_sd": stdev(seed_pre_itt) if len(seeds) > 1 else 0.0,
            "itt_csr": fmean(r["CSR"] for r in all_rows),
            "itt_amort_tokens": fmean(r["amort_tokens"] for r in all_rows),
            "itt_amort_tokens_sd": stdev(seed_amort_itt) if len(seeds) > 1 else 0.0,
            "itt_rev_conv_num": r0_conv_itt,
            "itt_rev_conv_den": len(r0_fails_itt),
            "itt_rev_conv_pct": (r0_conv_itt / len(r0_fails_itt) * 100) if r0_fails_itt else 100.0,
            # Complete-case sensitivity endpoints
            "cc_OR": fmean(r["OR"] for r in comp_rows),
            "cc_BOR": fmean(r["BOR"] for r in comp_rows),
            "cc_CPR": fmean(r["CPR"] for r in comp_rows),
            "cc_AS": fmean(r["AS"] for r in comp_rows),
            "cc_r0_csr": fmean(r["r0_csr"] for r in comp_rows),
            "cc_pre_csr": fmean(r["pre_csr"] for r in comp_rows),
            "cc_pre_csr_sd": stdev(seed_pre_cc) if len(seeds) > 1 else 0.0,
            "cc_csr": fmean(r["CSR"] for r in comp_rows),
            "cc_rev_conv_num": r0_conv_cc,
            "cc_rev_conv_den": len(r0_fails_cc),
            "cc_rev_conv_pct": (r0_conv_cc / len(r0_fails_cc) * 100) if r0_fails_cc else 100.0,
            "cc_rounds": fmean(r["rounds"] for r in comp_rows),
            "cc_calls": fmean(r["calls"] for r in comp_rows),
            "cc_tokens": fmean(r["tokens"] for r in comp_rows),
            "cc_tokens_sd": stdev(seed_tok_cc) if len(seeds) > 1 else 0.0,
            "tfr_all": fmean(r["tfr"] for r in comp_rows),
            "tfr_hard": fmean(r["tfr"] for r in comp_hard),
            "tfr_micro": sum(r["fit"] for r in comp_rows) / sum(r["n_txt"] for r in comp_rows) * 100,
            "joint_pre_allfit": fmean((100.0 if (r["pre_csr"] == 100 and r["allfit_pre"]) else 0.0) for r in comp_rows),
            "joint_fin_allfit": fmean((100.0 if (r["CSR"] == 100 and r["allfit_fin"]) else 0.0) for r in comp_rows),
            "joint_pre_allfit_hard": fmean((100.0 if (r["pre_csr"] == 100 and r["allfit_pre"]) else 0.0) for r in comp_hard),
            "joint_fin_allfit_hard": fmean((100.0 if (r["CSR"] == 100 and r["allfit_fin"]) else 0.0) for r in comp_hard),
            "straddle_flip_pre": sum(1 for r in comp_rows if r["pre_csr"] == 100 and r["strad_fp_pre"]),
            "straddle_flip_fin": sum(1 for r in comp_rows if r["CSR"] == 100 and r["strad_fp_fin"]),
            "strict_pre_csr": fmean((100.0 if (r["pre_csr"] == 100 and not r["strad_fp_pre"]) else 0.0) for r in comp_rows),
            "disp": fmean(r["disp"] for r in comp_rows),
            "dwh": fmean(r["dwh"] for r in comp_rows),
            "net_bw": fmean(r["net_bw"] for r in comp_rows),
            "arb_bw": fmean(r["arb_bw"] for r in comp_rows),
            "arb_triggers": sum(r["arb_triggered"] for r in comp_rows),
            "arb_geom": sum(r["arb_geom"] for r in comp_rows),
            "arb_crops": sum(r["arb_crops"] for r in comp_rows),
            "arb_shifts": sum(r["arb_shifts"] for r in comp_rows),
            "arb_recolors": sum(r["arb_recolors"] for r in comp_rows),
            "hard_zone_all": fmean(r["hz"][0] for r in comp_hard) * 100,
            "hard_zone_hdr": fmean(r["hz"][1] for r in comp_hard) * 100,
            "hard_zone_ftr": fmean(r["hz"][3] for r in comp_hard) * 100,
            "hard_zone_encl": fmean(r["hz"][4] for r in comp_hard) * 100,
            "hard_card_colored": fmean(r["card"]["colored"] for r in comp_hard) * 100,
            "hard_card_dark": fmean(r["card"]["dark"] for r in comp_hard) * 100,
            "hard_card_exact": fmean(r["card"]["exact"] for r in comp_hard) * 100,
            "med_card_colored": fmean(r["card"]["colored"] for r in comp_med) * 100,
            "bt_set": sum(r["bt_set"] for r in comp_rows),
            "bt_tot": sum(r["bt_tot"] for r in comp_rows),
            "bt_fail": sum(r["bt_fail"] for r in comp_rows),
            "c_to_g": reg.get("c_to_g", 0),
            "g_to_c": reg.get("g_to_c", 0),
            "r0_contrast": sum(r["r0_contrast"] for r in comp_rows),
            "r0_overlap": sum(r["r0_overlap"] for r in comp_rows),
            "r0_overflow": sum(r["r0_overflow"] for r in comp_rows),
            "by_diff": {
                d: {
                    "itt_r0_csr": fmean(per_run[m, s, t]["r0_csr"] for s in seeds for t in by_diff[d]),
                    "itt_pre_csr": fmean(per_run[m, s, t]["pre_csr"] for s in seeds for t in by_diff[d]),
                    "itt_amort": fmean(per_run[m, s, t]["amort_tokens"] for s in seeds for t in by_diff[d]),
                    "cc_pre_csr": fmean(per_run[m, s, t]["pre_csr"] for s in seeds for t in by_diff[d] if per_run[m, s, t]["completed"]),
                    "cc_tokens": fmean(per_run[m, s, t]["tokens"] for s in seeds for t in by_diff[d] if per_run[m, s, t]["completed"]),
                    "cc_n": sum(1 for s in seeds for t in by_diff[d] if per_run[m, s, t]["completed"]),
                }
                for d in DIFFICULTIES
            },
        }
        per_seed_summary[m] = {
            s: {
                "completed": sum(1 for t in task_ids if per_run[m, s, t]["completed"]),
                "itt_r0_csr": fmean(per_run[m, s, t]["r0_csr"] for t in task_ids),
                "itt_pre_csr": fmean(per_run[m, s, t]["pre_csr"] for t in task_ids),
                "itt_hard_pre": fmean(per_run[m, s, t]["pre_csr"] for t in by_diff["hard"]),
                "itt_amort": fmean(per_run[m, s, t]["amort_tokens"] for t in task_ids),
                "fails": len(fail_logs[m, s]),
                "rej": sum(per_run[m, s, t]["format_rejections"] for t in task_ids),
            }
            for s in seeds
        }

    # Pre-registered brief-clustered paired bootstrap contrasts — BOTH ITT and Complete-Case!
    contrasts_spec = [
        ("H1 (Sequential Decomp.)", "sa_twostage_bb_det", "single_agent_bb_detector"),
        ("H2 (Field-Locked Revision)", "ours_full", "sa_twostage_bb_det"),
        ("H3 (Two-Role + Field Locks)", "ours_full", "single_agent_bb_detector"),
        ("Ladder: BB Patching", "single_agent_bb_detector", "single_agent_detector"),
        ("Ladder: Full vs SA+Det", "ours_full", "single_agent_detector"),
        ("Ladder: Det vs Critic", "ours_full", "ours_no_detector"),
    ]
    contrasts = []
    for label, a, b in contrasts_spec:
        entry = {"label": label, "a": a, "b": b, "itt": {}, "complete_case": {}}
        for sub_name, sub_ids in (("All 36", task_ids), ("Hard 12", by_diff["hard"])):
            # 1. Primary ITT (all 36/12 briefs, 3 seeds each)
            itt_res = {}
            for key in ("pre_csr", "r0_csr", "amort_tokens"):
                diffs = [
                    fmean(per_run[a, s, t][key] for s in seeds) - fmean(per_run[b, s, t][key] for s in seeds)
                    for t in sub_ids
                ]
                m_d, lo, hi = brief_clustered_bootstrap(diffs)
                itt_res[key] = {"diff": m_d, "lo": lo, "hi": hi, "excludes_zero": (lo > 1e-9 or hi < -1e-9), "n_clusters": len(diffs)}
            entry["itt"][sub_name] = itt_res

            # 2. Sensitivity Complete-Case (pairs where both a and b completed in that seed)
            cc_res = {}
            for key in ("pre_csr", "r0_csr", "rounds", "calls", "tokens", "tfr", "disp", "net_bw"):
                diffs = []
                for t in sub_ids:
                    paired = [
                        per_run[a, s, t][key] - per_run[b, s, t][key]
                        for s in seeds
                        if per_run[a, s, t]["completed"] and per_run[b, s, t]["completed"]
                    ]
                    if paired:
                        diffs.append(fmean(paired))
                m_d, lo, hi = brief_clustered_bootstrap(diffs)
                cc_res[key] = {"diff": m_d, "lo": lo, "hi": hi, "excludes_zero": (lo > 1e-9 or hi < -1e-9), "n_clusters": len(diffs)}
            entry["complete_case"][sub_name] = cc_res
        contrasts.append(entry)

    comp_tok_sum = sum(r["tokens"] for r in per_run.values() if r["completed"])
    fail_tok_sum = sum(f.get("total_tokens") or 0 for fl in fail_logs.values() for f in fl)
    try:
        runs_dir_label = runs_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        runs_dir_label = runs_dir.name
    return {
        "provenance": {
            "runs_dir": runs_dir_label,
            "seeds": seeds,
            "total_scheduled_runs": len(per_run),
            "total_completed_runs": sum(1 for r in per_run.values() if r["completed"]),
            "total_residual_failed": sum(1 for r in per_run.values() if not r["completed"]),
            "total_failed_attempts": sum(len(v) for v in fail_logs.values()),
            "total_completed_tokens": comp_tok_sum,
            "total_failed_tokens": fail_tok_sum,
            "metric_mismatches": metric_mismatches,
            "strict_bb_replays_verified": strict_replays,
            "resolved_models": sorted(resolved_models),
            "system_fingerprints": sorted(fingerprints),
            "timestamp_earliest": min(timestamps) if timestamps else None,
            "timestamp_latest": max(timestamps) if timestamps else None,
        },
        "methods": method_summary,
        "per_seed": per_seed_summary,
        "contrasts": contrasts,
    }


def render_markdown(res: dict) -> str:
    p = res["provenance"]
    ms = res["methods"]
    seeds = p["seeds"]
    lines: list[str] = []
    lines.append("# CoLayout `runs_v2/` Pre-Registered Multi-Seed Results & Diagnostics")
    lines.append("")
    lines.append(f"- **Model snapshot**: `{p['resolved_models']}` | **Seeds**: `{seeds}` | **Scheduled**: `{p['total_scheduled_runs']}` | **Completed**: `{p['total_completed_runs']}` | **Residual failed**: `{p['total_residual_failed']}` | **Failed whole-run attempts**: `{p['total_failed_attempts']}`")
    lines.append(f"- **Total token spend**: `{p['total_completed_tokens']:,}` completed + `{p['total_failed_tokens']:,}` failed = **`{p['total_completed_tokens'] + p['total_failed_tokens']:,}`** tokens")
    lines.append(f"- **UTC window**: `{p['timestamp_earliest']}` → `{p['timestamp_latest']}`")
    lines.append(f"- **Integrity**: `metric_mismatches = {p['metric_mismatches']}`, `strict_bb_replays_verified = {p['strict_bb_replays_verified']}` (zero silent drops), observed `system_fingerprints` count = `{len(p['system_fingerprints'])}`")
    lines.append("")

    lines.append("## 1. Main 8-Method Results — Primary ITT vs. Complete-Case Sensitivity")
    lines.append("")
    lines.append("| Method | Done / 108 | Whole-Run Fail Attempts (Rate) | Format Rej | **ITT R0-CSR% (±SD)** | **ITT Pre-CSR% (±SD)** | **ITT Amort Tok (±SD)** | CC Pre-CSR% (±SD) | CC Rounds / Calls | CC Tokens (±SD) | CC TFR% (Hard%) | Hard Card (Col/Dark/Exact) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in METHODS:
        d = ms[m]
        lines.append(
            f"| `{m}` | {d['n_completed']}/{d['n_scheduled']} | {d['failed_attempts']} ({d['attempt_fail_rate_pct']:.1f}%) | {d['format_rejections']} | "
            f"{d['itt_r0_csr']:.1f} (±{d['itt_r0_csr_sd']:.1f}) | **{d['itt_pre_csr']:.1f}** (±{d['itt_pre_csr_sd']:.1f}) | {d['itt_amort_tokens']:,.0f} (±{d['itt_amort_tokens_sd']:,.0f}) | "
            f"{d['cc_pre_csr']:.1f} (±{d['cc_pre_csr_sd']:.1f}) | {d['cc_rounds']:.2f} / {d['cc_calls']:.2f} | {d['cc_tokens']:,.0f} (±{d['cc_tokens_sd']:,.0f}) | "
            f"{d['tfr_all']:.1f} ({d['tfr_hard']:.1f}) | {d['hard_card_colored']:.0f}/{d['hard_card_dark']:.0f}/{d['hard_card_exact']:.0f}% |"
        )
    lines.append("")

    def fmt_ci(x: dict, dec: int = 1, comma: bool = False) -> str:
        star = "*" if x["excludes_zero"] else ""
        if comma:
            return f"{x['diff']:+,.0f} [{x['lo']:+,.0f}, {x['hi']:+,.0f}]{star}"
        return f"{x['diff']:+.{dec}f} [{x['lo']:+.{dec}f}, {x['hi']:+.{dec}f}]{star}"

    lines.append("## 2A. Primary Pre-Registered Contrasts — Intention-to-Treat (Brief-Clustered Bootstrap, B=20,000, seed=42)")
    lines.append("")
    lines.append("| Contrast | Scope | Δ ITT Pre-CSR (pp) [95% CI] | Δ ITT R0-CSR (pp) [95% CI] | Δ ITT Amort Tok [95% CI] |")
    lines.append("|---|---|---:|---:|---:|")
    for c in res["contrasts"]:
        for sub_name in ("All 36", "Hard 12"):
            s = c["itt"][sub_name]
            lines.append(
                f"| **{c['label']}** (`{c['a']}` − `{c['b']}`) | {sub_name} (n={s['pre_csr']['n_clusters']}) | "
                f"{fmt_ci(s['pre_csr'], 1)} | {fmt_ci(s['r0_csr'], 1)} | {fmt_ci(s['amort_tokens'], 0, True)} |"
            )
    lines.append("")

    lines.append("## 2B. Sensitivity Contrasts — Complete-Case Only (Brief-Clustered Bootstrap, B=20,000, seed=42)")
    lines.append("")
    lines.append("| Contrast | Scope | Δ CC Pre-CSR (pp) [95% CI] | Δ CC R0-CSR (pp) [95% CI] | Δ Rounds [95% CI] | Δ Calls [95% CI] | Δ CC Tokens [95% CI] | Δ TFR (pp) [95% CI] |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in res["contrasts"]:
        for sub_name in ("All 36", "Hard 12"):
            s = c["complete_case"][sub_name]
            lines.append(
                f"| **{c['label']}** (`{c['a']}` − `{c['b']}`) | {sub_name} (n={s['pre_csr']['n_clusters']}) | "
                f"{fmt_ci(s['pre_csr'], 1)} | {fmt_ci(s['r0_csr'], 1)} | {fmt_ci(s['rounds'], 2)} | {fmt_ci(s['calls'], 2)} | "
                f"{fmt_ci(s['tokens'], 0, True)} | {fmt_ci(s['tfr'], 1)} |"
            )
    lines.append("")
    lines.append("*Asterisk `*` marks 95% cluster-bootstrap CIs excluding zero.*")
    lines.append("")

    lines.append("## 3. Stratified by Difficulty (36 Scheduled Runs per Method × Difficulty)")
    lines.append("")
    lines.append("| Method | Easy ITT R0 / ITT Pre / CC Pre / Amort | Medium ITT R0 / ITT Pre / CC Pre / Amort | Hard ITT R0 / ITT Pre / CC Pre / Amort |")
    lines.append("|---|---:|---:|---:|")
    for m in METHODS:
        bd = ms[m]["by_diff"]
        parts = [f"{bd[d]['itt_r0_csr']:.1f}% / {bd[d]['itt_pre_csr']:.1f}% / {bd[d]['cc_pre_csr']:.1f}% / {bd[d]['itt_amort']:,.0f}" for d in DIFFICULTIES]
        lines.append(f"| `{m}` | {parts[0]} | {parts[1]} | {parts[2]} |")
    lines.append("")

    lines.append("## 4. Diagnostics Across Detector-Guided Methods (Complete-Case)")
    lines.append("")
    lines.append("| Metric | `single_agent_detector` | `single_agent_bb_detector` | `sa_twostage_bb_det` | `ours_full` |")
    lines.append("|---|---:|---:|---:|---:|")
    def row(name, fn):
        lines.append("| " + name + " | " + " | ".join(fn(ms[m]) for m in DET_METHODS) + " |")
    row("Completed / Scheduled", lambda d: f"{d['n_completed']}/{d['n_scheduled']}")
    row("OR / BOR / CPR / AS (%)", lambda d: f"{d['cc_OR']:.1f} / {d['cc_BOR']:.1f} / {d['cc_CPR']:.1f} / {d['cc_AS']:.1f}")
    row("Revision conversion (CC / ITT)", lambda d: f"{d['cc_rev_conv_num']}/{d['cc_rev_conv_den']} ({d['cc_rev_conv_pct']:.1f}%) / {d['itt_rev_conv_num']}/{d['itt_rev_conv_den']} ({d['itt_rev_conv_pct']:.1f}%)")
    row("Mean center shift (px)", lambda d: f"{d['disp']:.2f}")
    row("Net B/W shifts (arbiter-only)", lambda d: f"{d['net_bw']:.2f} ({d['arb_bw']:.2f})")
    row("Arbiter triggers (geom/crop/shift/recolor)", lambda d: f"{d['arb_triggers']}/{d['n_completed']} ({d['arb_geom']}g/{d['arb_crops']}c/{d['arb_shifts']}s/{d['arb_recolors']}r)")
    row("Hard zone adherence (all / hdr / ftr / encl %)", lambda d: f"{d['hard_zone_all']:.1f} / {d['hard_zone_hdr']:.1f} / {d['hard_zone_ftr']:.1f} / {d['hard_zone_encl']:.1f}")
    row("Cross-attribute regressions (C→G / G→C)", lambda d: f"{d['c_to_g']} / {d['g_to_c']}")
    row("Raster text-fit: all / hard / micro (%)", lambda d: f"{d['tfr_all']:.1f} / {d['tfr_hard']:.1f} / {d['tfr_micro']:.1f}")
    row("R0 btn/tag bg_color filled (contrast fails)", lambda d: f"{d['bt_set']}/{d['bt_tot']} ({d['bt_fail']})")
    row("R0 conflicts (contrast / overlap / overflow)", lambda d: f"{d['r0_contrast']} / {d['r0_overlap']} / {d['r0_overflow']}")
    row("Hard card colored / dark / exact brief hex (%)", lambda d: f"{d['hard_card_colored']:.1f} / {d['hard_card_dark']:.1f} / {d['hard_card_exact']:.1f}")
    row("Medium card colored (%)", lambda d: f"{d['med_card_colored']:.1f}")
    row("Format rejections / failed whole-run attempts", lambda d: f"{d['format_rejections']} / {d['failed_attempts']} ({d['attempt_fail_rate_pct']:.1f}%)")
    lines.append("")

    lines.append("## 5. Seed-by-Seed Breakdown (Zero Seed Selection)")
    lines.append("")
    lines.append("| Method | Seed 1 (Done / ITT-R0 / ITT-Pre / HardPre / Amort / Fails / Rej) | Seed 2 | Seed 3 |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        cells = []
        for s in seeds:
            sd = res["per_seed"][m][s]
            cells.append(f"{sd['completed']}/36 · {sd['itt_r0_csr']:.1f}% / {sd['itt_pre_csr']:.1f}% / {sd['itt_hard_pre']:.1f}% · {sd['itt_amort']:,.0f} · {sd['fails']}f / {sd['rej']}r")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze runs_v2/ per pre-registered ANALYSIS_PLAN.md.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_v2")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--md-out", type=Path, default=ROOT / "runs_v2" / "results_summary.md")
    parser.add_argument("--json-out", type=Path, default=ROOT / "runs_v2" / "results_summary.json")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    res = analyze(args.runs_dir, seeds, args.font_path)
    md = render_markdown(res)
    args.md_out.write_text(md, encoding="utf-8")
    args.json_out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print(f"\nSaved {args.md_out} and {args.json_out}")


if __name__ == "__main__":
    main()
