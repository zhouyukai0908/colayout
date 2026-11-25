"""Build a paired, three-panel case study from detector-verified snapshots."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

from .detector import evaluate_layout
from .renderer import render_layout
from .schema import Conflict, LayoutState


PANELS = {
    "single_agent": "(a) Single-Agent",
    "naive_multi_agent": "(b) Naive Multi-Agent",
    "ours_full": "(c) CoLayout (Ours)",
}


def _conflict_counts(conflicts: list[Conflict]) -> Counter:
    # The comparison uses conflict type, severity, and involved elements.
    return Counter((c.conflict_type, c.severity, tuple(sorted(c.element_ids))) for c in conflicts)


def _verified_snapshot(row: dict) -> tuple[LayoutState, dict]:
    try:
        state = LayoutState.model_validate(row["final_state"])
        report = evaluate_layout(state)
        for key, value in report["metrics"].items():
            if not math.isclose(float(row["metrics"][key]), value, abs_tol=1e-12, rel_tol=1e-9):
                raise ValueError(f"saved {key} differs from the evaluated layout")
        saved = [Conflict.model_validate(c) for c in row["conflicts"]]
        if _conflict_counts(saved) != _conflict_counts(report["conflicts"]):
            raise ValueError("saved conflicts differ from the evaluated layout")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid case checkpoint {row['method']}/{row['task_id']}: {exc}") from exc
    return state, report


def _render_panel(state: LayoutState, folder: Path, annotated: bool) -> Image.Image:
    png = folder / "panel.png"
    render_layout(state, folder / "panel.html", show_conflicts=annotated, png_path=png)
    with Image.open(png) as rendered:
        return rendered.convert("RGB")


def _conflict_boxes(state: LayoutState, report: dict) -> list[tuple[float, float, float, float]]:
    """Collect bounding boxes for elements flagged by any hard conflict."""
    marked = {eid for conflict in report["conflicts"] for eid in conflict.element_ids}
    return [
        (el.x, el.y, el.w, el.h)
        for el in state.elements
        if el.id in marked and el.w > 0 and el.h > 0
    ]


def _write_figure(
    task: str,
    difficulty: str,
    panels: list[tuple[Image.Image, dict, list[tuple[float, float, float, float]]]],
    output: Path,
    *,
    mock: bool,
    show_legend: bool,
) -> None:
    # Use inches to keep text, gutters and margins consistent across canvas shapes.
    width = 7.1
    margin, gutter = 0.06, 0.20
    panel_width = (width - 2 * margin - 2 * gutter) / 3
    image_width, image_height = panels[0][0].size
    panel_height = panel_width * image_height / image_width
    bottom, top = 0.46, 0.26
    height = bottom + panel_height + top

    with rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    }):
        fig = Figure(figsize=(width, height), dpi=300, facecolor="white")
        FigureCanvasAgg(fig)
        panel_items = list(PANELS.items())
        for i, ((image, metrics, boxes), (method, title)) in enumerate(zip(panels, panel_items)):
            is_ours = method == "ours_full"
            left = margin + i * (panel_width + gutter)
            right = left + panel_width
            center = (left + panel_width / 2) / width
            ax = fig.add_axes([left / width, bottom / height, panel_width / width, panel_height / height])
            ax.imshow(image, interpolation="none")
            ax.set_xlim(-0.5, image_width - 0.5)
            ax.set_ylim(image_height - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])

            # Draw sharp vector conflict boxes inset cleanly so adjacent boxes stay distinct.
            for bx, by, bw, bh in boxes:
                inset_x = min(1.8, bw * 0.12)
                inset_y = min(1.8, bh * 0.12)
                rx = bx - 0.5 + inset_x
                ry = by - 0.5 + inset_y
                rw = max(1.0, bw - 2 * inset_x)
                rh = max(1.0, bh - 2 * inset_y)
                ax.add_patch(Rectangle(
                    (rx, ry), rw, rh,
                    fill=False,
                    edgecolor="#D62828",
                    linewidth=0.95,
                    linestyle=(0, (3.0, 1.8)),
                    joinstyle="miter",
                    clip_on=True,
                    zorder=5,
                ))

            for spine in ax.spines.values():
                spine.set_color("#1B365D" if is_ours else "#7C8B99")
                spine.set_linewidth(0.85 if is_ours else 0.55)

            fig.text(
                center, (height - 0.125) / height, title,
                fontsize=9.2, fontweight="bold",
                color="#1B365D" if is_ours else "#1C2D40",
                ha="center", va="center",
            )

            # Optically balance the three metrics across each panel width.
            metric_y = (bottom - 0.135) / height
            anchors = (
                ((left + 0.015) / width, "left", "OR"),
                ((left + 0.468 * panel_width) / width, "center", "BOR"),
                ((right - 0.015) / width, "right", "CPR"),
            )
            for mx, ha, key in anchors:
                val = metrics[key]
                failed = (val > 0) if key in ("OR", "BOR") else (val < 1.0)
                if failed:
                    color, weight = "#A61C1C", "semibold"
                elif is_ours:
                    color, weight = "#1B365D", "bold" if key == "CPR" else "normal"
                else:
                    color, weight = "#1F2937", "normal"
                fig.text(
                    mx, metric_y, f"{key}: {val:.1%}",
                    fontsize=7.9, fontweight=weight, color=color,
                    ha=ha, va="center",
                )

        # Subtle hairline separating panel metrics from the footer metadata and conflict legend.
        rule_y = 0.20 / height
        fig.add_artist(Line2D(
            [margin / width, 1 - margin / width], [rule_y, rule_y],
            transform=fig.transFigure, color="#DCE3EA", linewidth=0.55,
        ))

        footer_y = 0.088 / height
        label = f"Case: {task}  ·  Difficulty: {difficulty.capitalize()}"
        if mock:
            label += "  ·  Offline mock data"
        fig.text(margin / width, footer_y, label, fontsize=7.6, color="#475569", va="center")
        if show_legend:
            legend_text = "Red dashed boxes mark hard conflicts."
            text_right = 1 - margin / width
            legend_artist = fig.text(
                text_right, footer_y, legend_text, fontsize=7.6, color="#334155",
                ha="right", va="center",
            )
            # Measure exact rendered text bounds so the legend swatch sits cleanly beside it.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            text_box = fig.transFigure.inverted().transform_bbox(
                legend_artist.get_window_extent(renderer=renderer)
            )
            swatch_w = 0.145 / width
            swatch_h = 0.072 / height
            swatch_gap = 0.065 / width
            swatch_x = text_box.x0 - swatch_gap - swatch_w
            fig.add_artist(Rectangle(
                (swatch_x, footer_y - swatch_h / 2), swatch_w, swatch_h,
                transform=fig.transFigure, fill=False,
                edgecolor="#D62828", linewidth=0.95,
                linestyle=(0, (2.6, 1.5)),
            ))

        for extension in ("png", "pdf"):
            fig.savefig(output / f"fig3_case_study.{extension}", dpi=300, facecolor="white")


def export_case(
    records: dict[tuple[str, str], dict],
    case_task_id: str | None,
    figures_dir: Path,
    *,
    mock: bool = False,
) -> str:
    """Export the requested task, or the first qualifying hard task in sorted ID order."""
    tasks = sorted(task for method, task in records if method == "ours_full")
    if case_task_id is not None and case_task_id not in tasks:
        raise ValueError(f"Unknown case task {case_task_id!r}. Available tasks: {', '.join(tasks)}")
    candidates = ([case_task_id] if case_task_id is not None else
                  [task for task in tasks if records["ours_full", task]["difficulty"] == "hard"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=figures_dir) as scratch:
        for task in candidates:
            missing = [method for method in PANELS if (method, task) not in records]
            if missing:
                raise ValueError(f"Missing case checkpoints for {task}: {', '.join(missing)}")
            snapshots = [_verified_snapshot(records[method, task]) for method in PANELS]
            dimensions = {(s.canvas.width, s.canvas.height) for s, _ in snapshots}
            if len(dimensions) != 1:
                raise ValueError(f"Inconsistent canvas sizes for the three versions of {task}")
            ours = snapshots[-1][1]
            if ours["metrics"]["CSR"] != 1 or ours["conflicts"]:
                continue
            hard_baseline = any(
                c.severity == "hard" for _, r in snapshots[:-1] for c in r["conflicts"]
            )
            if case_task_id is None and not hard_baseline:
                continue

            panels = []
            visible = False
            for state, report in snapshots:
                clean = _render_panel(state, Path(scratch), annotated=False)
                boxes = _conflict_boxes(state, report) if report["conflicts"] else []
                if report["conflicts"]:
                    annotated = _render_panel(state, Path(scratch), annotated=True)
                    # Off-canvas violations can leave no visible red marks.
                    visible |= ImageChops.difference(clean, annotated).getbbox() is not None
                panels.append((clean, report["metrics"], boxes))
            if visible or case_task_id is not None:
                level = records["ours_full", task]["difficulty"]
                _write_figure(task, level, panels, figures_dir, mock=mock, show_legend=visible)
                return task

    if case_task_id is not None:
        raise ValueError(
            f"Case {case_task_id!r} requires CoLayout with CSR=1 and zero conflicts. "
            f"Available tasks: {', '.join(tasks)}"
        )
    raise ValueError(
        "No qualifying hard task. Automatic selection requires a visible baseline hard conflict "
        f"and CoLayout with CSR=1 and zero conflicts. Available tasks: {', '.join(tasks)}"
    )
