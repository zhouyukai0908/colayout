"""Render a CoLayout snapshot with optional conflict annotations and a PNG preview."""

import argparse
from html import escape
from pathlib import Path

from .detector import TEXT_TYPES, effective_background, evaluate_layout
from .schema import Element, LayoutState


def _geometry(element: Element) -> str:
    return (
        f"left:{element.x}px;top:{element.y}px;"
        f"width:{element.w}px;height:{element.h}px"
    )


def _fill(state: LayoutState, index: int) -> str | None:
    element = state.elements[index]
    if element.type == "divider":
        return element.bg_color or element.fg_color or "#000000"
    if element.type in TEXT_TYPES:
        return effective_background(state, index)
    return element.bg_color


def _html(state: LayoutState, order: list[int], marked: set[str]) -> str:
    canvas = state.canvas
    parts = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        "<title>CoLayout</title><style>",
        "*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif}",
        ".canvas{position:relative;overflow:hidden;isolation:isolate}",
        ".element,.conflict-box{position:absolute;overflow:hidden}",
        ".element{line-height:1.2;white-space:pre-line}",
        ".image{outline:1px solid #94a3b8;outline-offset:-1px}",
        ".conflict-box{border:2px dashed red;pointer-events:none}",
        "</style></head><body>",
        f'<main class="canvas" style="width:{canvas.width}px;height:{canvas.height}px;'
        f'background:{canvas.bg_color}">',
    ]
    for index in order:
        element = state.elements[index]
        fill = _fill(state, index) or "transparent"
        style = (
            f"{_geometry(element)};background:{fill};"
            f"color:{element.fg_color or '#000000'};font-size:{element.font_size or 16}px"
        )
        parts.append(
            f'<div class="element {element.type}" data-id="{escape(element.id)}" '
            f'style="{style}">{escape(element.text or "")}</div>'
        )
    # These siblings paint last, even when the affected element is behind a card.
    for element in state.elements:
        if element.id in marked:
            parts.append(
                f'<div class="conflict-box" aria-hidden="true" data-for="{escape(element.id)}" '
                f'style="{_geometry(element)}"></div>'
            )
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


def _raster_text(image, element: Element, bounds: tuple[int, int, int, int]) -> None:
    """Wrap at word boundaries and clip text to the visible part of its element."""
    from PIL import ImageDraw, ImageFont

    left, top, right, bottom = bounds
    visible = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    if visible[0] >= visible[2] or visible[1] >= visible[3]:
        return
    tile = image.crop(visible)
    draw = ImageDraw.Draw(tile)
    size = element.font_size or 16
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default(size=size)
    x, y = left - visible[0], top - visible[1]
    for paragraph in element.text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            if line and draw.textlength(candidate, font=font) > element.w:
                draw.text((x, y), line, fill=element.fg_color or "black", font=font)
                y += 1.2 * size
                line = word
            else:
                line = candidate
        draw.text((x, y), line, fill=element.fg_color or "black", font=font)
        y += 1.2 * size
    image.paste(tile, visible[:2])


def _dashes(start: int, stop: int, limit: int):
    """Yield visible dash intervals, preserving phase at clipped edges."""
    period, dash = 14, 8
    first = start + period * (max(0, -start) // period)
    for pos in range(first, min(limit, stop), period):
        yield pos, min(pos + dash, stop - 1)


def _png(state: LayoutState, order: list[int], marked: set[str], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("PNG export requires the optional dependency colayout[png]") from exc

    canvas = state.canvas
    image = Image.new("RGB", (canvas.width, canvas.height), canvas.bg_color)
    draw = ImageDraw.Draw(image)
    boxes = []
    for index in order:
        element = state.elements[index]
        left, top = round(element.x), round(element.y)
        right, bottom = round(element.x + element.w), round(element.y + element.h)
        if left >= right or top >= bottom:
            continue
        fill = _fill(state, index)
        outline = "#94a3b8" if element.type == "image" else None
        if fill or outline:
            draw.rectangle((left, top, right - 1, bottom - 1), fill=fill, outline=outline)
        if element.text:
            _raster_text(image, element, (left, top, right, bottom))
        if element.id in marked:
            boxes.append((left, top, right, bottom))

    stroke = 3
    for left, top, right, bottom in boxes:
        for start, end in _dashes(left, right, canvas.width):
            for y in (top, max(top, bottom - stroke)):
                draw.rectangle((start, y, end, min(y + stroke - 1, bottom - 1)), fill="#D62828")
        for start, end in _dashes(top, bottom, canvas.height):
            for x in (left, max(left, right - stroke)):
                draw.rectangle((x, start, min(x + stroke - 1, right - 1), end), fill="#D62828")
    image.save(output, format="PNG")


def render_layout(
    state: LayoutState,
    html_path: str | Path,
    *,
    show_conflicts: bool = False,
    png_path: str | Path | None = None,
) -> None:
    """Write standalone HTML and optionally a Pillow preview from a validated state."""
    marked = set()
    if show_conflicts:
        for conflict in evaluate_layout(state)["conflicts"]:
            marked.update(conflict.element_ids)
    # Stable sorting preserves JSON order within the same z-index.
    order = sorted(range(len(state.elements)), key=lambda i: state.elements[i].z_index)
    Path(html_path).write_text(_html(state, order, marked), encoding="utf-8")
    if png_path is not None:
        _png(state, order, marked, Path(png_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="layout JSON file")
    parser.add_argument("output", type=Path, help="destination HTML file")
    parser.add_argument("--show-conflicts", action="store_true", help="outline affected elements")
    parser.add_argument("--png", type=Path, help="also save a PNG preview (requires Pillow)")
    args = parser.parse_args()
    state = LayoutState.model_validate_json(args.input.read_bytes())
    render_layout(state, args.output, show_conflicts=args.show_conflicts, png_path=args.png)


if __name__ == "__main__":
    main()
