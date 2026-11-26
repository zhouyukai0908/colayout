"""Reproduce all paper tables, pre-arbiter metrics, paired tests, and diagnostics offline."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.detector import TEXT_TYPES
from src.export_results import _pre_arbiter_csr
from src.methods.arbiter import resolve_conflicts
from src.methods.blackboard import ARCHITECT, GEOMETRY, STYLE, VISUAL, _initial_geometry, _patched
from src.schema import Conflict, Element, LayoutState

METHODS = (
    "single_agent",
    "self_refine",
    "naive_multi_agent",
    "single_agent_detector",
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


def _text_fit_strict_counts(state: LayoutState) -> tuple[int, int]:
    """Return (fitting_boxes, total_text_boxes) using renderer.py greedy word wrap & Pillow arial.ttf."""
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    elems = [e for e in state.elements if e.type in TEXT_TYPES and e.text]
    if not elems:
        return 0, 0
    fit = 0
    for e in elems:
        size = e.font_size or 16
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except OSError:
            font = ImageFont.load_default(size=size)
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
    """Return (all_elem_zone_rate, header_rate, dark_card_enclosure_rate) on a hard brief."""
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


def _replay_r0_pre_final(method: str, raw: dict, brief: dict) -> tuple[LayoutState, LayoutState, LayoutState]:
    final = LayoutState.model_validate(raw["final_state"])
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
                fields = GEOMETRY if role == ARCHITECT else STYLE
                types = {"overflow", "overlap"} if role == ARCHITECT else {"contrast", "min_font"}
                ids = {i for c in last_hard if c.conflict_type in types for i in c.element_ids}
                state = _patched(entry["response"], state, brief, ids, fields)
            elif role == "Priority Arbiter":
                pre_state = state.model_copy(deep=True)
                state = resolve_conflicts(state, None if method == "ours_full" else last_hard)
        return init_state, pre_state or state, final
    if method == "single_agent_detector":
        r0 = LayoutState.model_validate(raw["history"][0]["response"])
        pre = r0
        for h in raw["history"]:
            if h.get("role") == "Layout Designer":
                pre = LayoutState.model_validate(h["response"])
        return r0, pre, final
    return final, final, final


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


def main() -> None:
    briefs = json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))
    briefs_by_id = {b["task_id"]: b for b in briefs}
    task_ids = sorted(briefs_by_id)
    hard_ids = [t for t in task_ids if briefs_by_id[t]["difficulty"] == "hard"]
    raw_dir = ROOT / "results" / "raw"
    fail_dir = ROOT / "results" / "failures"

    records = {m: {t: json.loads((raw_dir / f"{m}__{t}.json").read_text(encoding="utf-8")) for t in task_ids} for m in METHODS}
    pre_csr = {m: {t: int(_pre_arbiter_csr(records[m][t], briefs_by_id[t])) for t in task_ids} for m in METHODS}

    print("=== TABLE I SUMMARY (36 TASKS) ===")
    print(f"{'Method':<22} {'OR%':>5} {'BOR%':>5} {'CPR%':>6} {'AS%':>5} {'PreCSR%':>8} {'CSR%':>6} {'Rnd':>5} {'Tok':>7} {'TFR%':>6} {'Fit/216':>8} {'AmortTok':>9}")
    tot_comp, tot_fail, tot_fail_n = 0, 0, 0
    for m in METHODS:
        recs = [records[m][t] for t in task_ids]
        f_recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fail_dir.glob(f"{m}__*.json"))]
        comp_sum = sum(r["total_tokens"] for r in recs)
        fail_sum = sum(f["total_tokens"] for f in f_recs)
        tot_comp += comp_sum
        tot_fail += fail_sum
        tot_fail_n += len(f_recs)
        tf_pairs = [_text_fit_strict_counts(LayoutState.model_validate(r["final_state"])) for r in recs]
        tfr_macro = fmean(f / n for f, n in tf_pairs) * 100
        fit_sum, box_sum = sum(f for f, _ in tf_pairs), sum(n for _, n in tf_pairs)
        print(
            f"{m:<22} "
            f"{fmean(r['metrics']['OR'] for r in recs)*100:5.1f} "
            f"{fmean(r['metrics']['BOR'] for r in recs)*100:5.1f} "
            f"{fmean(r['metrics']['CPR'] for r in recs)*100:6.1f} "
            f"{fmean(r['metrics']['AS'] for r in recs)*100:5.1f} "
            f"{fmean(pre_csr[m][t] for t in task_ids)*100:8.1f} "
            f"{fmean(r['metrics']['CSR'] for r in recs)*100:6.1f} "
            f"{fmean(r['rounds_used'] for r in recs):5.2f} "
            f"{fmean(r['total_tokens'] for r in recs):7,.0f} "
            f"{tfr_macro:6.1f} "
            f"{fit_sum:>3d}/{box_sum:<3d} "
            f"{(comp_sum + fail_sum)/36:9,.0f}"
        )
    print(f"Total tokens: completed={tot_comp:,}, failed={tot_fail:,} ({tot_fail_n} attempts), grand={tot_comp+tot_fail:,}")

    print("\n=== PAIRED STATISTICAL TESTS (CoLayout Full vs Comparators) ===")
    for comp in ("ours_no_detector", "single_agent_detector"):
        for label, subset in (("All 36", task_ids), ("Hard 12", hard_ids)):
            n10, n01, p = exact_mcnemar([pre_csr["ours_full"][t] for t in subset], [pre_csr[comp][t] for t in subset])
            dt = paired_bootstrap_ci([records["ours_full"][t]["total_tokens"] for t in subset],
                                     [records[comp][t]["total_tokens"] for t in subset])
            dr = paired_bootstrap_ci([records["ours_full"][t]["rounds_used"] for t in subset],
                                     [records[comp][t]["rounds_used"] for t in subset])
            print(f"  [{label}] Full vs {comp:<22}: Pre-CSR ({n10} vs {n01}, p={p:.3g}) | dTok={dt[0]:+,.0f} [{dt[1]:+,.0f}, {dt[2]:+,.0f}] | dRnd={dr[0]:+.2f} [{dr[1]:+.2f}, {dr[2]:+.2f}]")

    print("\n=== REPAIR FIDELITY & HARD-BRIEF ZONE ADHERENCE ===")
    for m in ("ours_full", "single_agent_detector"):
        r0_fin, pre_fin, r0_da = [], [], []
        for t in task_ids:
            r0, pre, fin = _replay_r0_pre_final(m, records[m][t], briefs_by_id[t])
            r0_fin.append(_repair_delta(r0, fin))
            pre_fin.append(_repair_delta(pre, fin))
            r0_da.append(_repair_delta(r0, resolve_conflicts(r0, None)))
        print(
            f"  {m:<22}: R0->Final disp={fmean(d for d, _ in r0_fin):.2f}px, "
            f"net B/W={fmean(b for _, b in r0_fin):.2f}/brief (arbiter-only={fmean(b for _, b in pre_fin):.2f}), "
            f"R0->DirectArb B/W={fmean(b for _, b in r0_da):.2f}/brief"
        )
    sa_da = [_repair_delta(LayoutState.model_validate(records["single_agent"][t]["final_state"]),
                           resolve_conflicts(LayoutState.model_validate(records["single_agent"][t]["final_state"]), None))
             for t in task_ids]
    print(f"  {'single_agent+DirectArb':<22}: R0->DirectArb B/W={fmean(b for _, b in sa_da):.2f}/brief")
    for m in METHODS:
        hz = [_hard_zone_metrics(LayoutState.model_validate(records[m][t]["final_state"])) for t in hard_ids]
        print(f"  Zone[{m:<21}]: all_elem={fmean(x[0] for x in hz)*100:5.1f}%, header={fmean(x[1] for x in hz)*100:5.1f}%, dark_card_encl={fmean(x[2] for x in hz)*100:5.1f}%")


if __name__ == "__main__":
    main()
