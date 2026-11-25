"""Pydantic v2 models for a CoLayout snapshot and its conflict reports."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, FiniteFloat, field_validator


def _expand_hex(color: str) -> str:
    if len(color) == 4:
        return "#" + "".join(channel * 2 for channel in color[1:])
    return color


HexColor = Annotated[
    str, Field(pattern=r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"), AfterValidator(_expand_hex)
]
ElementType = Literal["title", "subtitle", "body", "image", "button", "tag", "card", "divider"]


class Canvas(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    bg_color: HexColor


class Element(BaseModel):
    id: str = Field(min_length=1)
    type: ElementType
    x: FiniteFloat
    y: FiniteFloat
    w: FiniteFloat = Field(ge=0)
    h: FiniteFloat = Field(ge=0)
    z_index: int = Field(default=1, ge=0)
    text: str | None = None
    font_size: int | None = Field(default=None, gt=0)
    fg_color: HexColor | None = "#000000"
    bg_color: HexColor | None = None


class LayoutState(BaseModel):
    canvas: Canvas
    elements: list[Element]
    version: int = 0

    @field_validator("elements")
    @classmethod
    def unique_ids(cls, elements: list[Element]) -> list[Element]:
        seen: set[str] = set()
        for element in elements:
            if element.id in seen:
                raise ValueError(f"Duplicate element id {element.id!r}")
            seen.add(element.id)
        return elements


class Conflict(BaseModel):
    conflict_type: Literal["overlap", "overflow", "contrast", "min_font"]
    element_ids: list[str]
    severity: Literal["hard", "soft"]
    detail: str
    numeric_value: FiniteFloat
