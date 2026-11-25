"""Reproduce all paper tables, pre-arbiter metrics, paired tests, and diagnostics offline."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from src.detector import TEXT_TYPES, evaluate_layout
from src.export_results import _pre_arbiter_csr
from src.generate_briefs import PALETTES
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
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return fmean(diffs), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _text_fit_strict(state: LayoutState) -> float:
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    elems = [e for e in state.elements if e.type in TEXT_TYPES and e.text]
    if not elems:
        return 1.0
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
    return fit / len(elems)


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
    print(f"{'Method':<22} {'OR%':>5} {'BOR%':>5} {'CPR%':>6} {'AS%':>5} {'PreCSR%':>8} {'CSR%':>6} {'Rnd':>5} {'Tok':>7} {'TFR%':>6} {'AmortTok':>9}")
    tot_comp, tot_fail, tot_fail_n = 0, 0, 0
    for m in METHODS:
        recs = [records[m][t] for t in task_ids]
        f_recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(fail_dir.glob(f"{m}__*.json"))]
        comp_sum = sum(r["total_tokens"] for r in recs)
        fail_sum = sum(f["total_tokens"] for f in f_recs)
        tot_comp += comp_sum
        tot_fail += fail_sum
        tot_fail_n += len(f_recs)
        tfr = fmean(_text_fit_strict(LayoutState.model_validate(r["final_state"])) for r in recs) * 100
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
            f"{tfr:6.1f} "
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


if __name__ == "__main__":
    main()
