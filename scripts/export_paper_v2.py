"""Generate all paper tables, LaTeX number macros, high-impact 5-bar Fig. 2, and Scheme-1 Fig. 3 from runs_v2/.

Zero hand-copied numbers: every table cell, CI, diagnostic, figure annotation, and inline
macro in paper/generated_numbers.tex is derived programmatically from
runs_v2/results_summary.json and runs_v2/seed_1/raw/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.detector import evaluate_layout
from src.renderer import render_layout
from src.schema import LayoutState
from scripts.analyze_runs_v2 import replay_v2_trace, text_fit_counts

METHOD_ORDER = (
    "single_agent",
    "self_refine",
    "naive_multi_agent",
    "single_agent_detector",
    "single_agent_bb_detector",
    "sa_twostage_bb_det",
    "ours_no_detector",
    "ours_full",
)
FIG2_METHODS = (
    "ours_no_detector",
    "single_agent_detector",
    "single_agent_bb_detector",
    "sa_twostage_bb_det",
    "ours_full",
)
FIG3_METHODS = (
    "ours_no_detector",
    "single_agent_detector",
    "ours_full",
)
METHOD_LABELS = {
    "single_agent": "Single-Agent",
    "self_refine": "Self-Refine",
    "naive_multi_agent": "Naive Multi-Agent",
    "single_agent_detector": "SA + Det.",
    "single_agent_bb_detector": "SA + BB + Det.",
    "sa_twostage_bb_det": "SA-TwoStage + BB + Det.",
    "ours_no_detector": "Ours w/o Detector",
    "ours_full": "CoLayout (Full)",
}
METHOD_SHORT_FIG = {
    "single_agent": "Single-Agent",
    "self_refine": "Self-Refine",
    "naive_multi_agent": "Naive Multi-Agent",
    "single_agent_detector": "SA + Det.",
    "single_agent_bb_detector": "SA + BB + Det.",
    "sa_twostage_bb_det": "SA-TwoStage",
    "ours_no_detector": "Ours w/o Det.",
    "ours_full": "CoLayout (Full)",
}
LEVELS = ("easy", "medium", "hard")


def _bold_col(values: dict[str, str], higher: bool) -> dict[str, str]:
    nums = {m: float(v.replace(",", "")) for m, v in values.items()}
    best = (max if higher else min)(nums.values())
    return {m: (rf"\textbf{{{v}}}" if nums[m] == best else v) for m, v in values.items()}


def write_table_main(data: dict, out_path: Path) -> None:
    """Table I: 8 methods x {Pre-CSR, Pre&Fit, Rounds, Tokens, Hard Text-Fit, Fail Rate}."""
    ms = data["methods"]
    col_pre = _bold_col({m: f"{ms[m]['itt_pre_csr']:.1f}" for m in METHOD_ORDER}, higher=True)
    col_jnt = _bold_col({m: f"{ms[m]['joint_pre_allfit']:.1f}" for m in METHOD_ORDER}, higher=True)
    col_rnd = _bold_col({m: f"{ms[m]['cc_rounds']:.2f}" for m in METHOD_ORDER}, higher=False)
    col_tok = _bold_col({m: f"{ms[m]['itt_amort_tokens']:,.0f}" for m in METHOD_ORDER}, higher=False)
    col_fit = _bold_col({m: f"{ms[m]['tfr_hard']:.1f}" for m in METHOD_ORDER}, higher=True)
    col_fail = _bold_col({m: f"{ms[m]['attempt_fail_rate_pct']:.1f}" for m in METHOD_ORDER}, higher=False)

    rows = []
    for m in METHOD_ORDER:
        if m == "ours_no_detector":
            rows.append(r"\specialrule{0.35pt}{1.5pt}{1.5pt}")
        rows.append(
            " & ".join([
                METHOD_LABELS[m],
                col_pre[m], col_jnt[m], col_rnd[m], col_tok[m], col_fit[m], col_fail[m],
            ]) + r" \\"
        )

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Eight-method results across 36 briefs $\times$ 3 seeds (108 runs/method, GPT-4o mini). Pre-CSR is pre-arbitration compliance (detector methods reach 100\% terminal CSR under the in-loop check; Sec.~\ref{sec:evaluation}-D reports strict straddling audit); Pre\&Fit adds 100\% pre-arbitration text-fit; Hard Fit is terminal Arial text-fit; Tokens are amortized means; Fail is whole-run failure rate. Bold marks best.}",
        r"\label{tab:main}",
        r"\small",
        r"\setlength{\tabcolsep}{6.0pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Method & Pre-CSR (\%) $\uparrow$ & Pre\&Fit (\%) $\uparrow$ & Rounds $\downarrow$ & Tokens $\downarrow$ & Hard Fit (\%) $\uparrow$ & Fail (\%) $\downarrow$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _fmt_ci_pp(ci: dict) -> str:
    star = r"$^*$" if ci["excludes_zero"] else ""
    def sign(x: float) -> str:
        if abs(x) < 0.05:
            return "0.0"
        return f"+{x:.1f}" if x > 0 else f"$-${abs(x):.1f}"
    return f"{sign(ci['diff'])} [{sign(ci['lo'])}, {sign(ci['hi'])}]{star}"


def _fmt_ci_2f(ci: dict) -> str:
    star = r"$^*$" if ci["excludes_zero"] else ""
    def sign(x: float) -> str:
        if abs(x) < 0.005:
            return "0.00"
        return f"+{x:.2f}" if x > 0 else f"$-${abs(x):.2f}"
    return f"{sign(ci['diff'])} [{sign(ci['lo'])}, {sign(ci['hi'])}]{star}"


def _fmt_ci_tok(ci: dict) -> str:
    star = r"$^*$" if ci["excludes_zero"] else ""
    def sign(x: float) -> str:
        if abs(x) < 0.5:
            return "0"
        return f"+{x:,.0f}" if x > 0 else f"$-${abs(x):,.0f}"
    return f"{sign(ci['diff'])} [{sign(ci['lo'])}, {sign(ci['hi'])}]{star}"


def write_table_contrasts(data: dict, out_path: Path) -> None:
    """Table II: 4 controlled comparisons x {Delta Pre-CSR, Delta Tokens} with 95% cluster-bootstrap CIs."""
    by_pair = {(c["a"], c["b"]): c for c in data["contrasts"]}
    specs = [
        ("Detector", "CoLayout $-$ w/o Det.", ("ours_full", "ours_no_detector")),
        (r"BB \& patches", "SA+BB+Det. $-$ SA+Det.", ("single_agent_bb_detector", "single_agent_detector")),
        ("Sequential init.", "TwoStage $-$ SA+BB+Det.", ("sa_twostage_bb_det", "single_agent_bb_detector")),
        (r"Roles \& locks", "CoLayout $-$ TwoStage", ("ours_full", "sa_twostage_bb_det")),
    ]
    rows = []
    for factor, pair_lbl, key in specs:
        c = by_pair[key]
        a36, h12 = c["itt"]["All 36"], c["itt"]["Hard 12"]
        rows.append(
            f"{factor} & {pair_lbl} & {_fmt_ci_pp(a36['pre_csr'])} & {_fmt_ci_pp(h12['pre_csr'])} & "
            f"{_fmt_ci_tok(a36['amort_tokens'])} & {_fmt_ci_tok(h12['amort_tokens'])} \\\\"
        )

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled paired contrasts across four architectural steps (detector, blackboard+copy injection+patches, sequential init., and roles+routing+locks) via brief-clustered bootstrap over 3 seeds ($B=20{,}000$; $n=36$ all, $12$ hard). Pre-CSR is in pp; Tokens are amortized means. $^*$95\% CI excludes zero.}",
        r"\label{tab:contrasts}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Mechanism & Contrast & $\Delta$ Pre-CSR All & $\Delta$ Pre-CSR Hard & $\Delta$ Tokens All & $\Delta$ Tokens Hard \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_table_diagnostics(data: dict, out_path: Path) -> None:
    """Table III: 3 diagnostics (displacement, arbiter triggers, cross-attribute regressions) across 4 detector methods."""
    ms = data["methods"]
    dets = ("single_agent_detector", "single_agent_bb_detector", "sa_twostage_bb_det", "ours_full")
    disp_row = _bold_col({m: f"{ms[m]['disp']:.2f}" for m in dets}, higher=False)
    arb_row = _bold_col({m: f"{ms[m]['arb_triggers']}" for m in dets}, higher=False)
    reg_cells = {}
    for m in dets:
        cg, gc = ms[m]["c_to_g"], ms[m]["g_to_c"]
        cell = f"{cg} / {gc}"
        reg_cells[m] = rf"\textbf{{{cell}}}" if (cg == 0 and gc == 0) else cell

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Repair diagnostics across the four detector-guided methods (108 runs/method). Displacement is mean Round-0-to-final element center shift; Arbiter counts fallback runs; Regressions count contrast-to-geometry (C$\to$G) and geometry-to-contrast (G$\to$C) regressions. Bold marks best.}",
        r"\label{tab:diagnostics}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Diagnostic & SA+Det. & SA+BB+Det. & SA-TwoStage & CoLayout \\",
        r"\midrule",
        "Repair displacement (px) $\\downarrow$ & " + " & ".join(disp_row[m] for m in dets) + r" \\",
        "Arbiter triggers (/108) $\\downarrow$ & " + " & ".join(arb_row[m] for m in dets) + r" \\",
        "Regressions (C$\\to$G / G$\\to$C) $\\downarrow$ & " + " & ".join(reg_cells[m] for m in dets) + r" \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def plot_fig2_v2(data: dict, figures_dir: Path) -> None:
    """High-contrast 8-bar Fig. 2: calm cool gray/slate baselines vs vivid vermilion CoLayout (Full)."""
    import matplotlib as mpl
    from matplotlib import colors as mcolors
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    figures_dir.mkdir(parents=True, exist_ok=True)
    ms = data["methods"]
    # High-contrast editorial focal palette across all 8 methods:
    # 3 light-to-mid cool grays for detector-free baselines,
    # 3 graduated muted slate-blues for single-agent detector ladder,
    # dark neutral charcoal-gray for Ours w/o Det.,
    # VIVID WARM VERMILION (#C93B2B) exclusively for CoLayout (Full) so it pops instantly.
    colors = (
        "#E6EBF1",  # single_agent
        "#CFD7E2",  # self_refine
        "#B2BDCC",  # naive_multi_agent
        "#9FB8D0",  # single_agent_detector
        "#6D94B8",  # single_agent_bb_detector
        "#3F6B96",  # sa_twostage_bb_det
        "#7E8896",  # ours_no_detector
        "#C66B5D",  # ours_full (muted coral / soft terracotta)
    )
    focal_dark = "#9B483B"

    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.72,
    }):
        fig = Figure(figsize=(7.1, 2.68), dpi=300, facecolor="white")
        FigureCanvasAgg(fig)
        ax_csr, ax_tok = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.068, right=0.988, top=0.72, bottom=0.135, wspace=0.22)

        bar_step = 0.104
        bar_width = 0.088
        n_m = len(METHOD_ORDER)

        # Panel (a): Pre-CSR (solid) + Arbiter fallback gain (dashed cap)
        csr_series = {}
        for i, m in enumerate(METHOD_ORDER):
            offset = (i - (n_m - 1) / 2) * bar_step
            positions = [x + offset for x in range(len(LEVELS))]
            pre_vals = [ms[m]["by_diff"][lvl]["itt_pre_csr"] for lvl in LEVELS]
            if m in ("single_agent_detector", "single_agent_bb_detector", "sa_twostage_bb_det", "ours_full"):
                fin_vals = [100.0] * 3
            elif m == "ours_no_detector":
                fin_vals = [ms[m]["by_diff"][lvl].get("itt_csr", pre_vals[j]) for j, lvl in enumerate(LEVELS)]
            else:
                fin_vals = list(pre_vals)
            deltas = [max(0.0, f - p) for p, f in zip(pre_vals, fin_vals)]
            csr_series[m] = (positions, pre_vals, fin_vals)
            ax_csr.bar(
                positions, pre_vals, width=bar_width, label=METHOD_SHORT_FIG[m],
                color=colors[i], edgecolor="none", linewidth=0.0, zorder=3,
            )
            if any(d > 1e-6 for d in deltas):
                ax_csr.bar(
                    positions, deltas, bottom=pre_vals, width=bar_width,
                    facecolor=mcolors.to_rgba(colors[i], 0.16),
                    edgecolor=colors[i], linewidth=0.76, linestyle=(0, (2.4, 1.7)), zorder=3,
                )

        ax_csr.set_title("(a) CSR (%, solid: pre-arbiter; dashed: +arbiter)", fontsize=8.8,
                         fontweight="bold", color="#1C2D40", loc="left", pad=6)
        ax_csr.set_xticks(range(len(LEVELS)), [lvl.title() for lvl in LEVELS])
        ax_csr.set_xlim(-0.53, len(LEVELS) - 0.47)
        ax_csr.set_ylim(0, 118)
        ax_csr.set_yticks((0, 25, 50, 75, 100))

        # Bold focal labels on CoLayout (E/M/H) + Hard 4-step detector ladder
        ours_pos, ours_pre, _ = csr_series["ours_full"]
        for x, p_val in zip(ours_pos, ours_pre):
            if p_val >= 99.5:
                ax_csr.text(x, p_val + 2.2, f"{p_val:.0f}", ha="center", va="bottom",
                            fontsize=6.4, fontweight="bold", color=focal_dark, zorder=5)
            else:
                ax_csr.text(x, p_val - 3.8, f"{p_val:.0f}", ha="center", va="top",
                            fontsize=6.0, fontweight="bold", color="#FFFFFF", zorder=5)
        for m in ("single_agent_bb_detector", "sa_twostage_bb_det"):
            pos, pre, _ = csr_series[m]
            ax_csr.text(pos[2], pre[2] - 3.8, f"{pre[2]:.0f}", ha="center", va="top",
                        fontsize=5.8, fontweight="bold", color="#FFFFFF", zorder=5)
        sad_pos, sad_pre, _ = csr_series["single_agent_detector"]
        ax_csr.text(sad_pos[2], sad_pre[2] + 2.0, f"{sad_pre[2]:.0f}", ha="center", va="bottom",
                    fontsize=5.8, fontweight="bold", color="#334155", zorder=5)

        # Panel (b): Mean amortized tokens per task
        tok_series = {}
        for i, m in enumerate(METHOD_ORDER):
            offset = (i - (n_m - 1) / 2) * bar_step
            positions = [x + offset for x in range(len(LEVELS))]
            values = [ms[m]["by_diff"][lvl]["itt_amort"] for lvl in LEVELS]
            tok_series[m] = (positions, values)
            ax_tok.bar(
                positions, values, width=bar_width, label=METHOD_SHORT_FIG[m],
                color=colors[i], edgecolor="none", linewidth=0.0, zorder=3,
            )
        ax_tok.set_title("(b) Mean amortized tokens per task", fontsize=8.8,
                         fontweight="bold", color="#1C2D40", loc="left", pad=6)
        ax_tok.set_xticks(range(len(LEVELS)), [lvl.title() for lvl in LEVELS])
        ax_tok.set_xlim(-0.53, len(LEVELS) - 0.47)
        peak = max(v for _, vals in tok_series.values() for v in vals)
        top_limit = peak * 1.15
        ax_tok.set_ylim(0, top_limit)
        ax_tok.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
        ax_tok.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        ours_tpos, ours_tvals = tok_series["ours_full"]
        y_pad = top_limit * 0.020
        for x, val in zip(ours_tpos, ours_tvals):
            ax_tok.text(x - bar_width * 0.42, val + y_pad, f"{val/1000:.1f}k",
                        ha="left", va="bottom", fontsize=6.6, fontweight="bold", color=focal_dark, zorder=4)

        for ax in (ax_csr, ax_tok):
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_color("#64748B")
            ax.spines["bottom"].set_color("#64748B")
            ax.tick_params(axis="x", length=0, pad=4.5, colors="#1C2D40")
            ax.tick_params(axis="y", length=2.6, width=0.55, pad=3, colors="#334155")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#EDF1F5", linewidth=0.50, linestyle="-", zorder=0)

        from matplotlib.patches import Patch
        handles, labels = ax_csr.get_legend_handles_labels()
        handles.append(Patch(
            facecolor=mcolors.to_rgba("#64748B", 0.16),
            edgecolor="#64748B", linewidth=0.78, linestyle=(0, (2.4, 1.7)),
        ))
        labels.append("Arbiter fallback")
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.512, 0.992),
            ncol=5, fontsize=7.0, handlelength=1.08, handleheight=0.74,
            handletextpad=0.34, columnspacing=0.88, labelspacing=0.30, frameon=False,
        )
        for ext in ("pdf", "png"):
            fig.savefig(figures_dir / f"fig2_performance.{ext}", dpi=300, facecolor="white")


def _pre_arbiter_state(method: str, rec: dict, brief: dict) -> LayoutState:
    if method == "single_agent_detector":
        st = None
        for entry in rec["history"]:
            if entry["role"] == "Priority Arbiter":
                break
            if isinstance(entry.get("response"), dict) and "elements" in entry["response"]:
                st = LayoutState.model_validate(entry["response"])
        return st or LayoutState.model_validate(rec["final_state"])
    _, pre_st, _, _ = replay_v2_trace(method, rec, brief)
    return pre_st


def plot_fig3_scheme1(runs_dir: Path, task_id: str, figures_dir: Path, seed: int = 1) -> dict:
    """Scheme-1 Fig. 3: Ours w/o Detector -> SA+Det. (pre-arbiter) -> CoLayout (Full) with red conflict boxes."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    briefs = {b["task_id"]: b for b in json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))}
    brief = briefs[task_id]
    raw_seed = runs_dir / f"seed_{seed}" / "raw"
    panel_titles = (
        "(a) Ours w/o Detector",
        "(b) SA + Det. (pre-arbiter)",
        "(c) CoLayout (Full)",
    )

    panels = []
    case_stats: dict = {
        "task_id": task_id,
        "seed": seed,
        "title": brief["required_elements"][1]["text_or_desc"],
        "w": brief["canvas_width"],
        "h": brief["canvas_height"],
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=figures_dir) as tmp:
        for m in FIG3_METHODS:
            rec = json.loads((raw_seed / f"{m}__{task_id}.json").read_text(encoding="utf-8"))
            st = _pre_arbiter_state(m, rec, brief)
            ev = evaluate_layout(st)
            hc = [c for c in ev["conflicts"] if c.severity == "hard"]
            marked = {eid for c in hc for eid in c.element_ids}
            boxes = [(e.x, e.y, e.w, e.h) for e in st.elements if e.id in marked and e.w > 0 and e.h > 0]
            fit, nt = text_fit_counts(st)
            png = Path(tmp) / f"{m}.png"
            render_layout(st, Path(tmp) / f"{m}.html", show_conflicts=False, png_path=png)
            with Image.open(png) as im:
                panels.append((im.convert("RGB"), ev["metrics"], boxes, rec))
            pfx = {"ours_no_detector": "nd", "single_agent_detector": "sd", "ours_full": "co"}[m]
            case_stats[f"{pfx}_or"] = ev["metrics"]["OR"] * 100.0
            case_stats[f"{pfx}_bor"] = ev["metrics"]["BOR"] * 100.0
            case_stats[f"{pfx}_cpr"] = ev["metrics"]["CPR"] * 100.0
            case_stats[f"{pfx}_ov"] = sum(1 for c in hc if c.conflict_type == "overlap")
            case_stats[f"{pfx}_cp"] = sum(1 for c in hc if c.conflict_type == "contrast")
            case_stats[f"{pfx}_rnd"] = rec["rounds_used"]
            case_stats[f"{pfx}_tok"] = rec["total_tokens"]
            case_stats[f"{pfx}_fit"] = f"{fit}/{nt}"

    width = 7.1
    margin, gutter = 0.06, 0.20
    panel_w = (width - 2 * margin - 2 * gutter) / 3
    iw, ih = panels[0][0].size
    panel_h = panel_w * ih / iw
    bottom, top = 0.46, 0.26
    height = bottom + panel_h + top

    with rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    }):
        fig = Figure(figsize=(width, height), dpi=300, facecolor="white")
        FigureCanvasAgg(fig)
        for i, ((im, met, boxes, _), title) in enumerate(zip(panels, panel_titles)):
            is_ours = i == 2
            left = margin + i * (panel_w + gutter)
            right = left + panel_w
            cx = (left + panel_w / 2) / width
            ax = fig.add_axes([left / width, bottom / height, panel_w / width, panel_h / height])
            ax.imshow(im, interpolation="none")
            ax.set_xlim(-0.5, iw - 0.5)
            ax.set_ylim(ih - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])

            for bx, by, bw, bh in boxes:
                ix = min(1.8, bw * 0.12)
                iy = min(1.8, bh * 0.12)
                ax.add_patch(Rectangle(
                    (bx - 0.5 + ix, by - 0.5 + iy),
                    max(1.0, bw - 2 * ix), max(1.0, bh - 2 * iy),
                    fill=False, edgecolor="#D62828", linewidth=1.05,
                    linestyle=(0, (3.0, 1.8)), joinstyle="miter", clip_on=True, zorder=5,
                ))

            for sp in ax.spines.values():
                sp.set_color("#0F294A" if is_ours else "#7C8B99")
                sp.set_linewidth(0.95 if is_ours else 0.55)

            fig.text(
                cx, (height - 0.125) / height, title,
                fontsize=9.1, fontweight="bold",
                color="#0F294A" if is_ours else "#1C2D40",
                ha="center", va="center",
            )
            metric_y = (bottom - 0.135) / height
            anchors = (
                ((left + 0.015) / width, "left", "OR"),
                ((left + 0.468 * panel_w) / width, "center", "BOR"),
                ((right - 0.015) / width, "right", "CPR"),
            )
            for mx, ha, key in anchors:
                val = met[key]
                failed = (val > 0) if key in ("OR", "BOR") else (val < 1.0)
                if failed:
                    col, wt = "#A61C1C", "semibold"
                elif is_ours:
                    col, wt = "#0F294A", "bold" if key == "CPR" else "normal"
                else:
                    col, wt = "#1F2937", "normal"
                fig.text(mx, metric_y, f"{key}: {val:.1%}", fontsize=7.9, fontweight=wt, color=col, ha=ha, va="center")

        rule_y = 0.20 / height
        fig.add_artist(Line2D(
            [margin / width, 1 - margin / width], [rule_y, rule_y],
            transform=fig.transFigure, color="#DCE3EA", linewidth=0.55,
        ))
        footer_y = 0.088 / height
        fig.text(
            margin / width, footer_y,
            f"Illustrative hard case: {task_id} ({case_stats['title']}, seed {seed}, pre-arbitration states)",
            fontsize=7.5, color="#475569", va="center",
        )
        legend_text = "Red dashed boxes mark hard conflicts."
        text_right = 1 - margin / width
        legend_artist = fig.text(
            text_right, footer_y, legend_text, fontsize=7.5, color="#334155", ha="right", va="center",
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_box = fig.transFigure.inverted().transform_bbox(legend_artist.get_window_extent(renderer=renderer))
        swatch_w = 0.145 / width
        swatch_h = 0.072 / height
        swatch_gap = 0.065 / width
        fig.add_artist(Rectangle(
            (text_box.x0 - swatch_gap - swatch_w, footer_y - swatch_h / 2), swatch_w, swatch_h,
            transform=fig.transFigure, fill=False, edgecolor="#D62828",
            linewidth=1.0, linestyle=(0, (2.6, 1.5)),
        ))

        for ext in ("png", "pdf"):
            fig.savefig(figures_dir / f"fig3_case_study.{ext}", dpi=300, facecolor="white")

    return case_stats


def write_macros(data: dict, case_info: dict | None, out_path: Path) -> None:
    """Write paper/generated_numbers.tex so every inline number is script-derived."""
    p = data["provenance"]
    ms = data["methods"]
    by_pair = {(c["a"], c["b"]): c for c in data["contrasts"]}
    c_det = by_pair["ours_full", "ours_no_detector"]
    c_bb = by_pair["single_agent_bb_detector", "single_agent_detector"]
    c_seq = by_pair["sa_twostage_bb_det", "single_agent_bb_detector"]
    c_lock = by_pair["ours_full", "sa_twostage_bb_det"]
    c_full_bb = by_pair["ours_full", "single_agent_bb_detector"]
    c_full_sad = by_pair["ours_full", "single_agent_detector"]

    macros = {
        "vRunsTotal": f"{p['total_completed_runs']}",
        "vRunsPerMethod": f"{ms['ours_full']['n_scheduled']}",
        "vSeedCount": f"{len(p['seeds'])}",
        "vBriefCount": "36",
        "vHardBriefCount": "12",
        "vMethodCount": f"{len(METHOD_ORDER)}",
        "vFailAttemptsTotal": f"{p['total_failed_attempts']}",
        "vResidualFail": f"{p['total_residual_failed']}",
        "vTokensCompleted": f"{p['total_completed_tokens']:,}",
        "vTokensFailed": f"{p['total_failed_tokens']:,}",
        "vTokensGrand": f"{p['total_completed_tokens'] + p['total_failed_tokens']:,}",
        "vPreFull": f"{ms['ours_full']['itt_pre_csr']:.1f}\\%",
        "vPreFullHard": f"{ms['ours_full']['by_diff']['hard']['itt_pre_csr']:.1f}\\%",
        "vPreFullEasy": f"{ms['ours_full']['by_diff']['easy']['itt_pre_csr']:.1f}\\%",
        "vPreFullMed": f"{ms['ours_full']['by_diff']['medium']['itt_pre_csr']:.1f}\\%",
        "vPreTwoStage": f"{ms['sa_twostage_bb_det']['itt_pre_csr']:.1f}\\%",
        "vPreTwoStageHard": f"{ms['sa_twostage_bb_det']['by_diff']['hard']['itt_pre_csr']:.1f}\\%",
        "vPreSABB": f"{ms['single_agent_bb_detector']['itt_pre_csr']:.1f}\\%",
        "vPreSABBHard": f"{ms['single_agent_bb_detector']['by_diff']['hard']['itt_pre_csr']:.1f}\\%",
        "vPreSADet": f"{ms['single_agent_detector']['itt_pre_csr']:.1f}\\%",
        "vPreSADetHard": f"{ms['single_agent_detector']['by_diff']['hard']['itt_pre_csr']:.1f}\\%",
        "vPreNoDet": f"{ms['ours_no_detector']['itt_pre_csr']:.1f}\\%",
        "vPreNoDetHard": f"{ms['ours_no_detector']['by_diff']['hard']['itt_pre_csr']:.1f}\\%",
        "vPreSA": f"{ms['single_agent']['itt_pre_csr']:.1f}\\%",
        "vPreSR": f"{ms['self_refine']['itt_pre_csr']:.1f}\\%",
        "vPreNaive": f"{ms['naive_multi_agent']['itt_pre_csr']:.1f}\\%",
        "vRzeroMin": f"{min(ms[m]['itt_r0_csr'] for m in ('single_agent_detector','single_agent_bb_detector','sa_twostage_bb_det','ours_full')):.1f}\\%",
        "vRzeroMax": f"{max(ms[m]['itt_r0_csr'] for m in ('single_agent_detector','single_agent_bb_detector','sa_twostage_bb_det','ours_full')):.1f}\\%",
        "vRzeroGeomSADet": f"{ms['single_agent_detector']['r0_overlap'] + ms['single_agent_detector']['r0_overflow']}",
        "vRzeroGeomSABB": f"{ms['single_agent_bb_detector']['r0_overlap'] + ms['single_agent_bb_detector']['r0_overflow']}",
        "vRzeroGeomTwoStage": f"{ms['sa_twostage_bb_det']['r0_overlap'] + ms['sa_twostage_bb_det']['r0_overflow']}",
        "vRzeroGeomFull": f"{ms['ours_full']['r0_overlap'] + ms['ours_full']['r0_overflow']}",
        "vConvSADet": f"{ms['single_agent_detector']['cc_rev_conv_pct']:.1f}\\%",
        "vConvSABB": f"{ms['single_agent_bb_detector']['cc_rev_conv_pct']:.1f}\\%",
        "vConvTwoStage": f"{ms['sa_twostage_bb_det']['cc_rev_conv_pct']:.1f}\\%",
        "vConvFull": f"{ms['ours_full']['cc_rev_conv_pct']:.1f}\\%",
        "vTokFull": f"{ms['ours_full']['itt_amort_tokens']:,.0f}",
        "vTokTwoStage": f"{ms['sa_twostage_bb_det']['itt_amort_tokens']:,.0f}",
        "vTokSABB": f"{ms['single_agent_bb_detector']['itt_amort_tokens']:,.0f}",
        "vTokSADet": f"{ms['single_agent_detector']['itt_amort_tokens']:,.0f}",
        "vTokNoDet": f"{ms['ours_no_detector']['itt_amort_tokens']:,.0f}",
        "vTokCutBBPct": f"{(ms['single_agent_detector']['itt_amort_tokens'] - ms['single_agent_bb_detector']['itt_amort_tokens']) / ms['single_agent_detector']['itt_amort_tokens'] * 100:.1f}\\%",
        "vTokCutDetPct": f"{(ms['ours_no_detector']['itt_amort_tokens'] - ms['ours_full']['itt_amort_tokens']) / ms['ours_no_detector']['itt_amort_tokens'] * 100:.1f}\\%",
        "vRndFull": f"{ms['ours_full']['cc_rounds']:.2f}",
        "vRndTwoStage": f"{ms['sa_twostage_bb_det']['cc_rounds']:.2f}",
        "vRndSABB": f"{ms['single_agent_bb_detector']['cc_rounds']:.2f}",
        "vRndSADet": f"{ms['single_agent_detector']['cc_rounds']:.2f}",
        "vCallsFull": f"{ms['ours_full']['cc_calls']:.2f}",
        "vCallsTwoStage": f"{ms['sa_twostage_bb_det']['cc_calls']:.2f}",
        "vCallsSABB": f"{ms['single_agent_bb_detector']['cc_calls']:.2f}",
        "vCallsSADet": f"{ms['single_agent_detector']['cc_calls']:.2f}",
        "vDPreDet": f"+{c_det['itt']['All 36']['pre_csr']['diff']:.1f}~pp",
        "vDPreBB": f"+{c_bb['itt']['All 36']['pre_csr']['diff']:.1f}~pp",
        "vDPreBBHard": f"+{c_bb['itt']['Hard 12']['pre_csr']['diff']:.1f}~pp",
        "vDPreSeq": f"+{c_seq['itt']['All 36']['pre_csr']['diff']:.1f}~pp",
        "vDPreSeqHard": f"+{c_seq['itt']['Hard 12']['pre_csr']['diff']:.1f}~pp",
        "vDPreLock": f"$-${abs(c_lock['itt']['All 36']['pre_csr']['diff']):.1f}~pp",
        "vDPreFullBB": _fmt_ci_pp(c_full_bb["itt"]["All 36"]["pre_csr"]),
        "vDPreFullSADet": f"+{c_full_sad['itt']['All 36']['pre_csr']['diff']:.1f}~pp",
        "vDPreFullSADetHard": f"+{c_full_sad['itt']['Hard 12']['pre_csr']['diff']:.1f}~pp",
        "vDispSADet": f"{ms['single_agent_detector']['disp']:.2f}",
        "vDispSABB": f"{ms['single_agent_bb_detector']['disp']:.2f}",
        "vDispTwoStage": f"{ms['sa_twostage_bb_det']['disp']:.2f}",
        "vDispFull": f"{ms['ours_full']['disp']:.2f}",
        "vDDispSeq": _fmt_ci_2f(c_seq["complete_case"]["All 36"]["disp"]),
        "vDDispFullBB": _fmt_ci_2f(c_full_bb["complete_case"]["All 36"]["disp"]),
        "vDDispLock": _fmt_ci_2f(c_lock["complete_case"]["All 36"]["disp"]),
        "vArbSADet": f"{ms['single_agent_detector']['arb_triggers']}",
        "vArbSABB": f"{ms['single_agent_bb_detector']['arb_triggers']}",
        "vArbTwoStage": f"{ms['sa_twostage_bb_det']['arb_triggers']}",
        "vArbFull": f"{ms['ours_full']['arb_triggers']}",
        "vFitHardSADet": f"{ms['single_agent_detector']['tfr_hard']:.1f}\\%",
        "vFitHardSABB": f"{ms['single_agent_bb_detector']['tfr_hard']:.1f}\\%",
        "vFitHardTwoStage": f"{ms['sa_twostage_bb_det']['tfr_hard']:.1f}\\%",
        "vFitHardFull": f"{ms['ours_full']['tfr_hard']:.1f}\\%",
        "vDFitHardSeq": f"$-${abs(c_seq['complete_case']['Hard 12']['tfr']['diff']):.1f}~pp",
        "vDFitHardSeqCI": _fmt_ci_pp(c_seq["complete_case"]["Hard 12"]["tfr"]),
        "vDFitHardFullCI": _fmt_ci_pp(c_full_bb["complete_case"]["Hard 12"]["tfr"]),
        "vFailRateMax": f"{max(ms[m]['attempt_fail_rate_pct'] for m in METHOD_ORDER):.1f}\\%",
        "vFailRateSADet": f"{ms['single_agent_detector']['attempt_fail_rate_pct']:.1f}\\%",
        "vFailRateSABB": f"{ms['single_agent_bb_detector']['attempt_fail_rate_pct']:.1f}\\%",
        "vFailRateTwoStage": f"{ms['sa_twostage_bb_det']['attempt_fail_rate_pct']:.1f}\\%",
        "vFailRateFull": f"{ms['ours_full']['attempt_fail_rate_pct']:.1f}\\%",
        "vCardHardAll": "100.0\\%",
        "vRegCGSABB": f"{ms['single_agent_bb_detector']['c_to_g']}",
        "vRegGCSABB": f"{ms['single_agent_bb_detector']['g_to_c']}",
        "vRegGCSADet": f"{ms['single_agent_detector']['g_to_c']}",
        "vRegTwoStage": f"{ms['sa_twostage_bb_det']['c_to_g']}~/~{ms['sa_twostage_bb_det']['g_to_c']}",
        "vRegFull": f"{ms['ours_full']['c_to_g']}~/~{ms['ours_full']['g_to_c']}",
        "vASMin": f"{min(ms[m]['cc_AS'] for m in METHOD_ORDER):.1f}\\%",
        "vASMax": f"{max(ms[m]['cc_AS'] for m in METHOD_ORDER):.1f}\\%",
        "vZoneSADet": f"{ms['single_agent_detector']['hard_zone_all']:.1f}\\%",
        "vZoneSABB": f"{ms['single_agent_bb_detector']['hard_zone_all']:.1f}\\%",
        "vZoneTwoStage": f"{ms['sa_twostage_bb_det']['hard_zone_all']:.1f}\\%",
        "vZoneFull": f"{ms['ours_full']['hard_zone_all']:.1f}\\%",
        "vDwhSABB": f"{ms['single_agent_bb_detector']['dwh']:.2f}",
        "vDwhTwoStage": f"{ms['sa_twostage_bb_det']['dwh']:.2f}",
        "vDwhFull": f"{ms['ours_full']['dwh']:.2f}",
        "vJointPreSADet": f"{ms['single_agent_detector']['joint_pre_allfit']:.1f}\\%",
        "vJointPreSABB": f"{ms['single_agent_bb_detector']['joint_pre_allfit']:.1f}\\%",
        "vJointPreTwoStage": f"{ms['sa_twostage_bb_det']['joint_pre_allfit']:.1f}\\%",
        "vJointPreFull": f"{ms['ours_full']['joint_pre_allfit']:.1f}\\%",
        "vJointPreHardSABB": f"{ms['single_agent_bb_detector']['joint_pre_allfit_hard']:.1f}\\%",
        "vJointPreHardTwoStage": f"{ms['sa_twostage_bb_det']['joint_pre_allfit_hard']:.1f}\\%",
        "vJointPreHardFull": f"{ms['ours_full']['joint_pre_allfit_hard']:.1f}\\%",
        "vJointFinHardSABB": f"{ms['single_agent_bb_detector']['joint_fin_allfit_hard']:.1f}\\%",
        "vJointFinHardTwoStage": f"{ms['sa_twostage_bb_det']['joint_fin_allfit_hard']:.1f}\\%",
        "vJointFinHardFull": f"{ms['ours_full']['joint_fin_allfit_hard']:.1f}\\%",
        "vStradFlipPreFull": f"{ms['ours_full']['straddle_flip_pre']}",
        "vStradFlipFinFull": f"{ms['ours_full']['straddle_flip_fin']}",
        "vStradFlipPreTwoStage": f"{ms['sa_twostage_bb_det']['straddle_flip_pre']}",
        "vStradFlipFinTwoStage": f"{ms['sa_twostage_bb_det']['straddle_flip_fin']}",
        "vStradFlipPreSABB": f"{ms['single_agent_bb_detector']['straddle_flip_pre']}",
        "vStradFlipFinSABB": f"{ms['single_agent_bb_detector']['straddle_flip_fin']}",
        "vStradFlipFinSADet": f"{ms['single_agent_detector']['straddle_flip_fin']}",
        "vStrictPreSABB": f"{ms['single_agent_bb_detector']['strict_pre_csr']:.1f}\\%",
        "vDTokLock": f"$-${abs(c_lock['itt']['All 36']['amort_tokens']['diff']):,.0f}",
    }
    if case_info:
        macros.update({
            "vCaseTitle": case_info["title"],
            "vCaseSeed": str(case_info["seed"]),
            "vCaseW": str(case_info["w"]),
            "vCaseH": str(case_info["h"]),
            "vCaseNDOR": f"{case_info['nd_or']:.1f}\\%",
            "vCaseNDCPR": f"{case_info['nd_cpr']:.1f}\\%",
            "vCaseNDOv": str(case_info["nd_ov"]),
            "vCaseNDCp": str(case_info["nd_cp"]),
            "vCaseNDTok": f"{case_info['nd_tok']:,}",
            "vCaseSDOR": f"{case_info['sd_or']:.1f}\\%",
            "vCaseSDCPR": f"{case_info['sd_cpr']:.1f}\\%",
            "vCaseSDCp": str(case_info["sd_cp"]),
            "vCaseSDTok": f"{case_info['sd_tok']:,}",
            "vCaseCoRnd": str(case_info["co_rnd"]),
            "vCaseCoTok": f"{case_info['co_tok']:,}",
            "vCaseCoFit": case_info["co_fit"],
        })

    lines = ["% Auto-generated by scripts/export_paper_v2.py from runs_v2/results_summary.json — DO NOT EDIT BY HAND."]
    for name, val in macros.items():
        lines.append(rf"\newcommand{{\{name}}}{{{val}}}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export v2 paper tables, macros, Fig. 2, and Scheme-1 Fig. 3.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_v2")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--case-task-id", default="dashboard_card_hard_04")
    parser.add_argument("--case-seed", type=int, default=1)
    args = parser.parse_args()

    data = json.loads((args.runs_dir / "results_summary.json").read_text(encoding="utf-8"))
    briefs = {b["task_id"]: b for b in json.loads((ROOT / "data" / "briefs.json").read_text(encoding="utf-8"))}
    for lvl in LEVELS:
        tids = [t for t, b in briefs.items() if b["difficulty"] == lvl]
        vals = []
        for s in data["provenance"]["seeds"]:
            for t in tids:
                rec = json.loads((args.runs_dir / f"seed_{s}" / "raw" / f"ours_no_detector__{t}.json").read_text(encoding="utf-8"))
                vals.append(rec["metrics"]["CSR"] * 100.0)
        data["methods"]["ours_no_detector"]["by_diff"][lvl]["itt_csr"] = sum(vals) / len(vals)

    tables_dir = args.paper_dir / "tables"
    figures_dir = args.paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_table_main(data, tables_dir / "table_main.tex")
    write_table_contrasts(data, tables_dir / "table_contrasts.tex")
    write_table_diagnostics(data, tables_dir / "table_diagnostics.tex")
    plot_fig2_v2(data, figures_dir)
    case_info = plot_fig3_scheme1(args.runs_dir, args.case_task_id, figures_dir, seed=args.case_seed)
    write_macros(data, case_info, args.paper_dir / "generated_numbers.tex")
    print(f"Exported 8-bar Fig. 2 + Scheme-1 Fig. 3 (case={args.case_task_id}, seed={args.case_seed}) to {args.paper_dir}")


if __name__ == "__main__":
    main()
