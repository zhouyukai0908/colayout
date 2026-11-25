import json
import re
from collections import Counter
from typing import get_args

from src.generate_briefs import DEFAULT_PATH, generate_briefs, write_briefs
from src.schema import ElementType


def test_checked_in_dataset_matches_repeated_generation(tmp_path):
    out = tmp_path / "data" / "briefs.json"
    write_briefs(out)
    first = out.read_bytes()
    write_briefs(out)
    assert first == out.read_bytes() == DEFAULT_PATH.read_bytes()
    assert json.loads(first) == generate_briefs()


def test_dataset_balance_and_content():
    briefs = generate_briefs()
    domains = ("poster", "dashboard_card", "editorial_hero")
    counts = {"easy": (4, 5), "medium": (7, 9), "hard": (10, 14)}
    assert len(briefs) == len({brief["task_id"] for brief in briefs}) == 36
    assert Counter((b["domain"], b["difficulty"]) for b in briefs) == {
        (domain, difficulty): 4 for domain in domains for difficulty in counts
    }
    titles = set()
    for b in briefs:
        assert b["canvas_width"] > 0 and b["canvas_height"] > 0
        assert re.fullmatch(r"#[\da-fA-F]{6}", b["canvas_bg_color"])
        assert b["design_notes"].strip()
        elements = b["required_elements"]
        lower, upper = counts[b["difficulty"]]
        assert lower <= len(elements) <= upper
        assert len({e["id"] for e in elements}) == len(elements)
        assert all(e["id"] and e["text_or_desc"].strip() for e in elements)
        types = Counter(e["type"] for e in elements)
        assert set(types) <= set(get_args(ElementType))
        assert types["title"] == 1
        titles.add(next(e["text_or_desc"] for e in elements if e["type"] == "title"))
        if b["difficulty"] != "easy":
            assert {"title", "subtitle", "body", "image", "button", "tag", "card"} <= types.keys()
        if b["difficulty"] == "hard":
            assert types["button"] >= 2 and types["tag"] >= 2
            assert "top 64 px" in b["design_notes"] and "bottom 56 px" in b["design_notes"]
    assert len(titles) == 36
