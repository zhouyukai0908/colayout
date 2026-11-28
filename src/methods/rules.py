"""Shared verbatim rule constants and non-ambiguous format templates.

All role-aligned methods share the exact same GEOMETRY_RULES, STYLE_RULES,
REVISION_RULES, and placeholder-based OUTPUT_FORMAT_* templates (with concrete
field names and placeholder tokens—never literal "field" keys and never
concrete hex colors that could bias color choices).
"""

from __future__ import annotations

GEOMETRY_RULES = (
    "Use positive pixel sizes. Use z_index 0 for cards and >=1 for foreground; "
    "type 'image' and all text types are foreground (z_index >= 1). "
    "Plan for disjoint foreground bounds, edge alignment, readable text and the "
    "specified header and footer zones."
)

STYLE_RULES = (
    "Use bg_color (hex or null) for containers and images, and explicit positive "
    "font_size and hex fg_color for all text types. Respect the brief's color constraints."
)

REVISION_RULES = (
    "Resolve the routed conflicts with targeted partial updates to the affected IDs. "
    "Preserve the design notes, alignment, and non-conflicting layout. Keep cards "
    "at z_index 0 and all other elements at z_index >= 1."
)

FIELD_PLACEHOLDERS: dict[str, str] = {
    "x": "<number>",
    "y": "<number>",
    "w": "<number>",
    "h": "<number>",
    "z_index": "<int>",
    "font_size": "<int>",
    "fg_color": '"<hex>"',
    "bg_color": '"<hex_or_null>"',
}
_FIELD_ORDER = ("x", "y", "w", "h", "z_index", "font_size", "fg_color", "bg_color")


def format_update_example(fields: set[str] | frozenset[str]) -> str:
    """Build a concrete-key placeholder JSON object for the allowed fields (no concrete colors)."""
    ordered = [f for f in _FIELD_ORDER if f in fields]
    pairs = ['"id":"<element_id>"', *(f'"{f}":{FIELD_PLACEHOLDERS[f]}' for f in ordered)]
    return "{" + ",".join(pairs) + "}"


OUTPUT_FORMAT_GEOMETRY_INIT = (
    'Initialize geometry. Return {"updates":['
    + format_update_example({"x", "y", "w", "h", "z_index"})
    + "]} with only those fields and one row per required element."
)

OUTPUT_FORMAT_STYLE_INIT_HEADER = (
    "Initialize typography and color for every ID. Provide fg_color and font_size "
    "for every text element and non-null bg_color for every card."
)

OUTPUT_FORMAT_PATCH_TEMPLATE = (
    'Use {{"updates":[{example}]}}. Put fields directly on each update object '
    "(no nested wrapper keys). Include only id and the needed fields from: {fields}. Use pixels."
)

OUTPUT_FORMAT_BB_SINGLE_INIT_TEMPLATE = (
    "Initialize geometry, typography, and color for every required element. "
    "Provide fg_color and font_size for every text element and non-null bg_color for every card. "
    'Return {{"updates":[{example}]}} with one row per required element, fields directly on each '
    "update object (no nested wrapper keys), and only the needed fields from: {fields}. Use pixels."
)

OUTPUT_FORMAT_FULL_LAYOUT_INIT = (
    "Design the complete layout described in the brief below. The LayoutState JSON "
    "object has fields canvas {width, height, bg_color}, elements, and version "
    "(initially 0). Use the exact canvas and one element per required id and type. "
    "Each element has id, type, x, y, w, h (top-left coordinates and sizes in pixels), "
    "and z_index. Text types (title, subtitle, body, button, tag) have text copied "
    "verbatim from text_or_desc. Other types use text_or_desc as a design instruction. "
    "Respect design_notes."
)

OUTPUT_FORMAT_FULL_LAYOUT_REVISION = (
    "Resolve all hard conflicts reported by the deterministic detector while preserving "
    "the design notes, alignment, and non-conflicting elements. Return only the complete "
    "revised LayoutState JSON object with top-level keys canvas, elements, and version."
)


def patch_format_instruction(fields: set[str] | frozenset[str]) -> str:
    return OUTPUT_FORMAT_PATCH_TEMPLATE.format(
        example=format_update_example(fields),
        fields=", ".join(sorted(fields)),
    )


def bb_single_init_format_instruction(fields: set[str] | frozenset[str]) -> str:
    return OUTPUT_FORMAT_BB_SINGLE_INIT_TEMPLATE.format(
        example=format_update_example(fields),
        fields=", ".join(sorted(fields)),
    )
