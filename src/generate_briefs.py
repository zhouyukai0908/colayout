"""Generate the fixed English-language CoLayout benchmark."""

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "briefs.json"


@dataclass(frozen=True)
class Copy:
    title: str
    subtitle: str
    image: str
    action: str
    body: str = ""
    tags: tuple[str, ...] = ()
    secondary_action: str = ""
    caption: str = ""


# Entries are authored separately so each benchmark item has its own subject,
# audience, copy lengths, and visual material. Order fixes the public task IDs.
CATALOG = {
    "poster": {
        "easy": [
            Copy("Seed Swap Sunday", "12 May | Glasshouse Yard", "Envelopes of seeds on a potting bench",
                 "Reserve a table"),
            Copy("River After Dark", "Lantern walk | East Quay", "Lantern reflections on a tidal river",
                 "Join the walk", "Meet the guide at the old ferry steps at dusk."),
            Copy("Quiet Hour", "Every Thursday | 09:00-10:00", "Sunlight falling across empty gallery benches",
                 "Plan a visit"),
            Copy("Fix Your Radio", "Repair cafe | Workshop 3", "An open transistor radio beside a soldering mat",
                 "Book a bench", "Bring one battery-powered radio and work alongside a volunteer."),
        ],
        "medium": [
            Copy("Print the Tide", "Two-color linocut workshop", "Ink rollers and a blue wave print",
                 "Book a place", "Carve a coastal motif and take home three hand-pulled prints.", ("Studio B",)),
            Copy("Fruit Tree Census", "Field training | 18 June", "A surveyor labeling a pear tree",
                 "Register", "Learn to identify orchard varieties before the summer survey.",
                 ("Field skills", "Free")),
            Copy("Listening Room", "An evening of field recordings", "A hydrophone suspended below a pier",
                 "Reserve a seat", "Hear a harbor soundscape assembled from underwater recordings.",
                 ("Pier Hall", "All ages", "Live")),
            Copy("Comet Watch", "West Ridge Observatory | 21:00", "A comet trail above the ridgeline",
                 "Get a pass", "Join the public telescope session and pick up a printed sky chart.",
                 ("Outdoors", "Clear skies only")),
        ],
        "hard": [
            Copy("Clay After Firing", "Kiln opening | Makers Courtyard", "Glazed bowls on cooling shelves",
                 "Book entry", "Compare the surfaces of salt-fired and wood-fired stoneware.",
                 ("Ceramics", "Hands-on"), "See the schedule", "Cooling rack 4"),
            Copy("Night Ferry Stories", "Dockside reading | Berth Four", "A moored ferry against the pier lights",
                 "Get tickets", "Six local writers read short stories set along the estuary.",
                 ("Readings", "Step-free", "Evening"), "Find the pier", "Final ferry at 22:40"),
            Copy("Build a Rain Garden", "Community planting day | Alder Lane", "A plan of a curbside planting bed",
                 "Join a team", "Help shape a planted basin that catches runoff from the school roof.",
                 ("Outdoors", "Tools provided"), "View the plan", "Planting bed B"),
            Copy("The Last Reel", "Silent film with live piano", "A film reel and its handwritten cue sheet",
                 "Reserve seats", "A restored travelogue returns to the screen with a new live score.",
                 ("Screening", "35 mm", "One night"), "Read film notes", "Projection starts at 19:30"),
        ],
    },
    "dashboard_card": {
        "easy": [
            Copy("Bike Dock Availability", "18 of 24 spaces free", "A strip chart of hourly dock occupancy",
                 "Open station map"),
            Copy("Tank Level", "72% full | North reservoir", "A gauge showing the current water level",
                 "View readings", "Level is inside the target band for this week."),
            Copy("Batch Queue", "6 jobs waiting", "A sparkline of queued jobs over the last hour",
                 "Inspect queue"),
            Copy("Library Returns", "148 items this morning", "A small bar chart of returns by hour",
                 "View branches", "The busiest return point is the East Annex."),
        ],
        "medium": [
            Copy("Archive Digitization", "2,430 pages scanned", "A bar chart of pages scanned per weekday",
                 "Open batch log", "The map collection accounts for half of this week's scans.", ("Weekly",)),
            Copy("Greenhouse Climate", "24.8 C | Bay Three", "A temperature trace with a shaded target band",
                 "Inspect sensors", "Roof vents opened when the temperature crossed the upper threshold.",
                 ("Live", "Vent open")),
            Copy("Transit Reliability", "92% on time", "A line chart comparing three bus routes",
                 "View route detail", "Route 8 recovered after the morning diversion.",
                 ("Weekdays", "North line", "AM peak")),
            Copy("Kitchen Surplus", "86 portions redistributed", "Stacked bars of portions by collection site",
                 "Open collection log", "The evening pickup added two neighborhood fridges.",
                 ("This week", "Four sites")),
        ],
        "hard": [
            Copy("Pump Station Load", "68 kW | Pump group A", "A power trace for the last twelve hours",
                 "Inspect load", "The standby pump cut in during the afternoon peak.",
                 ("Telemetry", "Standby active"), "Compare pumps", "Window 06:00-18:00"),
            Copy("Lab Freezer Watch", "Cold store 2 | -76 C", "A temperature series with a marked door event",
                 "Review event", "Temperature returned to the operating range after a short door opening.",
                 ("Recovered", "Door sensor", "Verified"), "View history", "Door open for 42 s"),
            Copy("Parcel Sorting", "1,240 items per hour", "A belt throughput chart with a maintenance interval",
                 "Open shift log", "Line B resumed sorting after a scheduled belt inspection.",
                 ("Shift B", "Resumed"), "Compare lines", "Inspection at 14:10"),
            Copy("Air Sensor Coverage", "21 of 24 stations online", "A grid map of monitoring stations",
                 "Locate outage", "Three hillside stations missed their last hourly transmission.",
                 ("Hourly", "3 offline", "Uplink"), "Export report", "Hill district"),
        ],
    },
    "editorial_hero": {
        "easy": [
            Copy("Paper From Reeds", "Material studies | Issue 6", "Reed fibers spread across a paper screen",
                 "Read the essay"),
            Copy("Where Moss Keeps Time", "Field journal | Rowan Vale", "Moss covering a stone survey marker",
                 "Open journal", "A botanist follows a slow change along an old boundary wall."),
            Copy("Letters at Dawn", "Photo essay | Night shift", "Mail sacks being loaded before sunrise",
                 "View photographs"),
            Copy("A Seat at the Loom", "Profiles | The weaving room", "A shuttle passing through a red warp",
                 "Read the profile", "An apprentice restores the rhythm of a neighborhood workshop."),
        ],
        "medium": [
            Copy("The Color of Old Glass", "Objects | Written by Mira Lee", "A row of green and amber glass fragments",
                 "Read feature", "Trace minerals tell a story that the bottle labels no longer can.", ("Materials",)),
            Copy("Along the Snow Line", "A mountain field diary", "Layered snowpack exposed in a shallow pit",
                 "Read the diary", "At the end of winter a survey crew retraces a route marked by wooden poles.",
                 ("Fieldwork", "Alpine")),
            Copy("Beneath the Arcade", "City notebook | Market streets", "Stone arches framing a produce stall",
                 "Explore the story", "A shaded walkway holds a record of the city's changing trades.",
                 ("Architecture", "Markets", "History")),
            Copy("Reading the Mud", "Estuary notes | Low tide", "Bird tracks crossing a rippled mudflat",
                 "Read field notes", "Footprints left before sunrise mark a feeding route across the inlet.",
                 ("Ecology", "Coast")),
        ],
        "hard": [
            Copy("How a Map Remembers", "Atlas desk | An annotated essay", "A folded map with erased pencil routes",
                 "Read the essay", "Successive surveys left faint paths beneath the modern street grid.",
                 ("Cartography", "Archives"), "View map plates", "Sheet 12, east quarter"),
            Copy("The Wind Readers", "Portraits | Harbor pilots", "A harbor pilot looking past a windsock",
                 "Read portraits", "Before each crossing, pilots compare the instruments with the water.",
                 ("Navigation", "Work", "Coast"), "Hear interviews", "Pilot station at daybreak"),
            Copy("Thread by Thread", "Craft archive | Repair traditions", "A close detail of a darned wool sleeve",
                 "Read feature", "A nearly invisible mend preserves the shape of a garment in daily use.",
                 ("Textiles", "Repair"), "View close-ups", "Detail of a woven patch"),
            Copy("The Orchard Ledger", "Long read | Orchard records", "A grower's notebook beside labeled apples",
                 "Read article", "Decades of handwritten harvest notes reveal the fortunes of a family orchard.",
                 ("Agriculture", "Records", "Seasons"), "Browse facsimiles", "Autumn entries, folio 8"),
        ],
    },
}

# Canvas sizes are in pixels. The compact hard canvases leave room for a header,
# two body columns and a button row while increasing density over the easy tasks.
SIZES = {
    "poster":         {"easy": (600, 800), "medium": (560, 720), "hard": (460, 560)},
    "dashboard_card": {"easy": (800, 600), "medium": (680, 460), "hard": (500, 360)},
    "editorial_hero": {"easy": (800, 600), "medium": (760, 480), "hard": (640, 360)},
}
PALETTES = {
    "poster": ("#FFF9ED", "#33223D"),
    "dashboard_card": ("#F3F6FA", "#142B3D"),
    "editorial_hero": ("#FAF8F1", "#23342F"),
}
ART_DIRECTION = {
    "poster": "Make the event title legible from a distance and lead the eye to the invitation.",
    "dashboard_card": "Keep the reading and its unit together. Align the trend graphic with the reading.",
    "editorial_hero": "Give the headline and lead image visual priority. Keep publication metadata subordinate.",
}


def _build(domain: str, difficulty: str, number: int, copy: Copy) -> dict:
    width, height = SIZES[domain][difficulty]
    background, panel = PALETTES[domain]
    elements = []

    def add(id_: str, kind: str, content: str) -> None:
        elements.append({"id": id_, "type": kind, "text_or_desc": content})

    notes = ART_DIRECTION[domain] + " Use the supplied text verbatim."
    if difficulty != "easy":
        add("panel", "card", "Background container for the content group.")
    add("headline", "title", copy.title)
    add("subhead", "subtitle", copy.subtitle)
    if copy.body:
        add("description", "body", copy.body)
    add("visual", "image", copy.image)
    for index, tag in enumerate(copy.tags, 1):
        add(f"tag_{index}", "tag", tag)
    add("primary_action", "button", copy.action)

    if difficulty == "easy":
        notes += " Leave generous space between the content and the canvas edges."
    elif difficulty == "medium":
        notes += " Use panel at z_index 0 to group the description and tags. Give all foreground items clearance."
    elif difficulty == "hard":
        add("secondary_action", "button", copy.secondary_action)
        add("caption", "body", copy.caption)
        notes += (
            " Reserve the top 64 px for headline and subhead and the bottom 56 px for the"
            " two buttons. Place both buttons side by side. Keep the remaining elements"
            f" inside the middle zone. Give panel z_index 0 and bg_color {panel}."
            " Enclose the description, caption and all tags in panel. Use light text"
            " with inherited backgrounds there. Give visual at least one third of the"
            " canvas width and leave at least 8 px between foreground items."
        )
    return {
        "task_id": f"{domain}_{difficulty}_{number:02d}",
        "domain": domain, "difficulty": difficulty,
        "canvas_width": width, "canvas_height": height, "canvas_bg_color": background,
        "required_elements": elements, "design_notes": notes,
    }


def generate_briefs() -> list[dict]:
    """Return fresh brief dictionaries in stable benchmark order."""
    return [
        _build(domain, difficulty, number, copy)
        for domain, levels in CATALOG.items()
        for difficulty, entries in levels.items()
        for number, copy in enumerate(entries, 1)
    ]


def write_briefs(path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(generate_briefs(), indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


if __name__ == "__main__":
    write_briefs()
    print(f"Wrote 36 design briefs to {DEFAULT_PATH}")
