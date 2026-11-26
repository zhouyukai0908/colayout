"""Publication tables and the two-panel performance figure from grouped task means."""

from math import fsum
from pathlib import Path


METHOD_LABELS = {
    "single_agent": "Single-Agent",
    "self_refine": "Self-Refine",
    "naive_multi_agent": "Naive Multi-Agent",
    "single_agent_detector": "Single-Agent + Det.",
    "single_agent_bb_detector": "Single-Agent + BB + Det.",
    "ours_no_detector": "Ours w/o Detector",
    "ours_full": "CoLayout (Full)",
}
LEVELS = ("easy", "medium", "hard")
METRICS = ("OR", "BOR", "CPR", "AS", "pre_CSR", "CSR", "rounds_used", "total_tokens")
HIGHER_IS_BETTER = {"CPR", "AS", "pre_CSR", "CSR"}


def _metric_value(row: dict, field: str) -> float:
    if field == "pre_CSR":
        return float(row.get("pre_CSR", row["CSR"]))
    return float(row[field])


def _overall(groups: dict) -> dict:
    rows = {}
    for method in METHOD_LABELS:
        levels = [groups[method, level] for level in LEVELS]
        count = sum(row["n_tasks"] for row in levels)
        rows[method] = {
            field: fsum(row["n_tasks"] * _metric_value(row, field) for row in levels) / count
            for field in METRICS
        }
    return rows


def _ranked_column(rows: dict, field: str, *, bold_best: bool = True) -> dict[str, str]:
    """Compare the displayed precision so a displayed tie receives identical emphasis."""
    def format_value(value: float) -> str:
        if field == "rounds_used":
            return f"{value:.2f}"
        if field == "total_tokens":
            return f"{value:,.0f}"
        return f"{100 * value:.1f}"

    cells = {method: format_value(_metric_value(row, field)) for method, row in rows.items()}
    if not bold_best:
        return cells
    numeric = {method: float(cell.replace(",", "")) for method, cell in cells.items()}
    best = (max if field in HIGHER_IS_BETTER else min)(numeric.values())
    return {
        method: rf"\textbf{{{cell}}}" if numeric[method] == best else cell
        for method, cell in cells.items()
    }


def _table_rows(columns: list[dict[str, str]]) -> list[str]:
    rows = []
    for method, label in METHOD_LABELS.items():
        if method == "ours_no_detector":
            rows.append(r"\specialrule{0.35pt}{1.5pt}{1.5pt}")
        rows.append(" & ".join([label, *(column[method] for column in columns)]) + r" \\")
    return rows


def write_tables(groups: dict, tables_dir: Path, *, mock: bool = False) -> None:
    """Write the overall comparison and per-difficulty pre-arbiter CSR and token usage as LaTeX floats."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    provenance = "Offline mock data. " if mock else ""
    mean_rows = _overall(groups)
    overall_cells = [_ranked_column(mean_rows, field, bold_best=(field != "CSR")) for field in METRICS]
    headings = []
    for field in METRICS:
        heading = {
            "pre_CSR": r"Pre-CSR (\%)",
            "rounds_used": "Rounds",
            "total_tokens": "Tokens",
        }.get(field, field + r" (\%)")
        arrow = r"\uparrow" if field in HIGHER_IS_BETTER else r"\downarrow"
        headings.append(heading + " $" + arrow + "$")
    main = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{" + provenance + f"Mean scores and costs over "
        f"{sum(groups['single_agent', d]['n_tasks'] for d in LEVELS)} tasks. "
        r"OR, BOR, CPR, AS, and CSR score terminal states (CSR=100\% is "
        r"guaranteed by arbitration, unbolded); Pre-CSR is pre-arbitration; "
        r"Tokens average completed runs. Bold marks best.}",
        r"\label{tab:main}", r"\small", r"\setlength{\tabcolsep}{4.8pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
        " & ".join(["Method", *headings]) + r" \\", r"\midrule",
        *_table_rows(overall_cells),
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ]
    (tables_dir / "table_main.tex").write_text("\n".join(main) + "\n", encoding="utf-8")

    cells = []
    for level in LEVELS:
        rows = {method: groups[method, level] for method in METHOD_LABELS}
        cells.extend(_ranked_column(rows, field) for field in ("pre_CSR", "total_tokens"))
    difficulty = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + provenance + r"Pre-arbiter hard constraint satisfaction (Pre-CSR, \%) and mean token use by difficulty. "
        r"After priority arbitration, all three detector-guided methods reach 100.0\% CSR across all levels "
        r"(Ours w/o Detector: 75.0\%, 33.3\%, 0.0\%). Bold marks highest Pre-CSR and lowest tokens per level.}",
        r"\label{tab:difficulty}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        "Method & " + " & ".join(r"\multicolumn{2}{c}{" + level.title() + "}" for level in LEVELS)
        + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        " & " + " & ".join(["Pre-CSR & Tokens"] * len(LEVELS)) + r" \\", r"\midrule",
        *_table_rows(cells),
        r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}",
    ]
    (tables_dir / "table_difficulty.tex").write_text("\n".join(difficulty) + "\n", encoding="utf-8")


def _format_csr_label(value: float) -> str:
    return f"{value:.0f}%" if abs(value - round(value)) < 0.05 else f"{value:.1f}%"


def _format_token_label(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def plot_performance(groups: dict, figures_dir: Path, *, mock: bool = False) -> None:
    """Draw difficulty-grouped bars encoding pre-arbiter CSR, arbiter fallback gain, and token cost."""
    import matplotlib as mpl
    from matplotlib import colors as mcolors
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Borderless flat semantic two-scale cool palette: pure neutral cool grays
    # for detector-free methods (including Ours w/o Detector), academic steel
    # blues for detector-guided single-agent controls, and deep Prussian/Oxford
    # Navy (#142F54) as the sole dark visual anchor for CoLayout (Full).
    colors = ("#E2E7EE", "#C5CED9", "#9EAAB8", "#9BC0E0", "#4D83B3", "#6E7A89", "#142F54")
    with mpl.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.70,
    }):
        fig = Figure(figsize=(7.1, 2.6), dpi=300, facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(
            left=0.068, right=0.988, top=0.72,
            bottom=0.185 if mock else 0.135, wspace=0.22,
        )
        bar_step = 0.118
        bar_width = 0.100
        methods = list(METHOD_LABELS.items())
        has_arbiter_cap = False

        # Panel (a): Pre-Arbiter CSR (borderless flat solid) + Arbiter gain (tinted dashed cap).
        ax_csr, ax_tok = axes
        csr_series = {}
        for i, (method, label) in enumerate(methods):
            offset = (i - (len(methods) - 1) / 2) * bar_step
            positions = [x + offset for x in range(len(LEVELS))]
            pre_vals = [100 * _metric_value(groups[method, level], "pre_CSR") for level in LEVELS]
            fin_vals = [100 * _metric_value(groups[method, level], "CSR") for level in LEVELS]
            deltas = [max(0.0, f - p) for p, f in zip(pre_vals, fin_vals)]
            csr_series[method] = (positions, pre_vals, fin_vals)
            ax_csr.bar(
                positions, pre_vals, width=bar_width, label=label,
                color=colors[i], edgecolor="none", linewidth=0.0, zorder=3,
            )
            if any(d > 1e-6 for d in deltas):
                has_arbiter_cap = True
                ax_csr.bar(
                    positions, deltas, bottom=pre_vals, width=bar_width,
                    facecolor=mcolors.to_rgba(colors[i], 0.18),
                    edgecolor=colors[i],
                    linewidth=0.75, linestyle=(0, (2.5, 1.8)), zorder=3,
                )

        title_a = "(a) CSR (%, solid: pre-arbiter; dashed: +arbiter)" if has_arbiter_cap else "(a) Constraint satisfaction (CSR, %)"
        ax_csr.set_title(title_a, fontsize=8.8, fontweight="bold",
                         color="#1C2D40", loc="left", pad=6)
        ax_csr.set_xticks(range(len(LEVELS)), [level.title() for level in LEVELS])
        ax_csr.set_xlim(-0.53, len(LEVELS) - 0.47)
        ax_csr.set_ylim(0, 115)
        ax_csr.set_yticks((0, 25, 50, 75, 100))

        # Place pre-arbiter callouts with crisp contrast and zero collision.
        ours_pos, ours_pre, ours_fin = csr_series["ours_full"]
        for x, p_val, f_val in zip(ours_pos, ours_pre, ours_fin):
            if p_val >= 99.5 and f_val - p_val < 1e-6:
                ax_csr.text(
                    x, p_val + 2.0, f"{p_val:.0f}", ha="center", va="bottom",
                    fontsize=6.3, fontweight="bold", color="#142F54", zorder=5,
                )
            elif p_val >= 30.0:
                ax_csr.text(
                    x, p_val - 4.0, f"{p_val:.0f}", ha="center", va="top",
                    fontsize=6.0, fontweight="bold", color="#FFFFFF", zorder=5,
                )
            else:
                ax_csr.text(
                    x, max(p_val, f_val) + 2.4, _format_csr_label(p_val), ha="center", va="bottom",
                    fontsize=6.6, fontweight="bold", color="#142F54", zorder=5,
                )
        # Highlight the Hard-task three-step pre-arbiter ladder (25 -> 33 -> 75) with high contrast.
        for key, txt_color in (("single_agent_detector", "#1E293B"), ("single_agent_bb_detector", "#FFFFFF")):
            if key in csr_series:
                pos, pre, _ = csr_series[key]
                if 15.0 <= pre[2] < 50.0 and ours_pre[2] >= 50.0:
                    ax_csr.text(
                        pos[2], pre[2] - 3.5, f"{pre[2]:.0f}",
                        ha="center", va="top", fontsize=5.8, fontweight="bold",
                        color=txt_color, zorder=5,
                    )

        # Panel (b): Mean tokens per completed task.
        tok_series = {}
        for i, (method, label) in enumerate(methods):
            offset = (i - (len(methods) - 1) / 2) * bar_step
            positions = [x + offset for x in range(len(LEVELS))]
            values = [_metric_value(groups[method, level], "total_tokens") for level in LEVELS]
            tok_series[method] = (positions, values)
            ax_tok.bar(
                positions, values, width=bar_width, label=label,
                color=colors[i], edgecolor="none", linewidth=0.0, zorder=3,
            )

        ax_tok.set_title("(b) Mean tokens per task", fontsize=8.8, fontweight="bold",
                         color="#1C2D40", loc="left", pad=6)
        ax_tok.set_xticks(range(len(LEVELS)), [level.title() for level in LEVELS])
        ax_tok.set_xlim(-0.53, len(LEVELS) - 0.47)
        peak = max(v for _, vals in tok_series.values() for v in vals)
        top_limit = max(peak * 1.13, 1.0)
        ax_tok.set_ylim(0, top_limit)
        ax_tok.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
        ax_tok.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ours_tpos, ours_tvals = tok_series["ours_full"]
        y_pad = top_limit * 0.022
        for x, val in zip(ours_tpos, ours_tvals):
            ax_tok.text(
                x - bar_width * 0.40, val + y_pad, _format_token_label(val),
                ha="left", va="bottom", fontsize=6.9, fontweight="bold",
                color="#142F54", zorder=4,
            )

        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_color("#64748B")
            ax.spines["bottom"].set_color("#64748B")
            ax.tick_params(axis="x", length=0, pad=4.5, colors="#1C2D40")
            ax.tick_params(axis="y", length=2.6, width=0.55, pad=3, colors="#334155")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#EDF1F5", linewidth=0.50, linestyle="-", zorder=0)

        handles, labels = ax_csr.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.515, 0.992),
            ncol=4, fontsize=7.2, handlelength=1.10, handleheight=0.76,
            handletextpad=0.38, columnspacing=1.05, labelspacing=0.28, frameon=False,
        )
        if mock:
            fig.text(0.068, 0.025, "Offline mock data", fontsize=7.5, style="italic", color="#475569")
        for ext in ("pdf", "png"):
            fig.savefig(figures_dir / f"fig2_performance.{ext}", dpi=300, facecolor="white")
