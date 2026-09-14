"""Rendering one self-contained HTML report per page."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, PackageLoader
from markupsafe import Markup
from PIL import Image, UnidentifiedImageError

from manyhands import __version__
from manyhands.vote import Slot

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from manyhands.pipeline import PageResult

#: Line crops are inlined as JPEG at this width at most, which keeps a busy page's
#: report well under a megabyte while staying legible at 1:1.
CROP_MAX_WIDTH = 1400
CROP_QUALITY = 78
CROP_PADDING = 8


@lru_cache(maxsize=1)
def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("manyhands", "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["script_json"] = _script_json
    return env


def _script_json(value: Any) -> Markup:
    """Serialise a value for a ``<script type="application/json">`` block.

    HTML-escaping would corrupt the JSON, so the three characters that could close the
    script element are escaped as JSON unicode instead and the result is marked safe.
    """
    payload = json.dumps(value, ensure_ascii=False)
    for char, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        payload = payload.replace(char, escaped)
    return Markup(payload)


def render_page(result: PageResult, siblings: Sequence[str] = ()) -> str:
    """Render the self-contained ``report.html`` for one page.

    :param siblings: Every page stem in the run, in page order, for the next/previous links.
    """
    previous, following = _neighbours(result.stem, siblings)
    slots = result.ballot.slots
    lines = _group_lines(slots)
    crops = _band_crops(result)
    summary = _summary(result)
    return _environment().get_template("report.html.j2").render(
        page=result.name,
        stem=result.stem,
        previous=previous,
        following=following,
        blank=result.blank,
        backends=result.ballot.backends,
        lines=lines,
        crops=crops,
        slot_data=_slot_data(slots, result.ballot.backends),
        summary=summary,
        dropped=result.dropped,
        notes={
            transcript.backend: transcript.notes
            for transcript in result.transcripts
            if transcript.notes
        },
        timings={
            transcript.backend: round(transcript.seconds, 2) for transcript in result.transcripts
        },
        version=__version__,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _neighbours(stem: str, siblings: Sequence[str]) -> tuple[str | None, str | None]:
    stems = list(siblings)
    if stem not in stems:
        return None, None
    at = stems.index(stem)
    return (stems[at - 1] if at else None), (stems[at + 1] if at + 1 < len(stems) else None)


def render_index(results: Sequence[PageResult]) -> str:
    """Render the folder-level index that links to every page report."""
    rows = [
        {
            "page": result.name,
            "stem": result.stem,
            "slots": len(result.ballot.slots),
            "flagged": result.flagged,
            "flag_rate": (result.flagged / len(result.ballot.slots)) if result.ballot.slots else 0.0,
            "blank": result.blank,
            "dropped": sorted(result.dropped),
        }
        for result in results
    ]
    total_slots = sum(row["slots"] for row in rows)
    total_flagged = sum(row["flagged"] for row in rows)
    return _environment().get_template("index.html.j2").render(
        rows=rows,
        total_slots=total_slots,
        total_flagged=total_flagged,
        flag_rate=(total_flagged / total_slots) if total_slots else 0.0,
        version=__version__,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def _group_lines(slots: Sequence[Slot]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Slot]] = {}
    for slot in slots:
        grouped.setdefault(slot.line, []).append(slot)
    return [
        {
            "line": line,
            "slots": [
                {
                    "i": slot.index,
                    # U+2205 would be the honest glyph but Georgia has no empty set, so it tofus.
                    "text": slot.consensus or "·",
                    "band": slot.band,
                    "agreement": slot.agreement,
                    "confirmed": slot.confirmed,
                    "contested": slot.contested,
                    "empty": not slot.consensus,
                }
                for slot in grouped[line]
            ],
        }
        for line in sorted(grouped)
    ]


def _slot_data(slots: Sequence[Slot], backends: Sequence[str]) -> dict[str, Any]:
    return {
        str(slot.index): {
            "line": slot.line,
            "bbox": list(slot.bbox) if slot.bbox else None,
            "agreement": round(slot.agreement, 4),
            "contested": slot.contested,
            "confirmed": slot.confirmed,
            "consensus": slot.consensus,
            "votes": slot.votes,
            "readings": [
                [
                    name,
                    (slot.readings[name].text if slot.readings.get(name) else None),
                    name in slot.winners,
                ]
                for name in backends
            ],
        }
        for slot in slots
        if slot.flagged
    }


def _summary(result: PageResult) -> dict[str, Any]:
    slots = result.ballot.slots
    total = len(slots)
    flagged = sum(1 for slot in slots if slot.flagged)
    bands = {"red": 0, "amber": 0, "pale": 0}
    for slot in slots:
        if slot.band in bands:
            bands[slot.band] += 1
    return {
        "total": total,
        "flagged": flagged,
        "flag_rate": (flagged / total) if total else 0.0,
        "bands": bands,
        "confirmed": sum(1 for slot in slots if slot.confirmed),
        "voters": len(result.ballot.backends),
    }


def _band_crops(result: PageResult) -> dict[str, dict[str, Any]]:
    """Return one inlined crop per written line that carries a flagged word.

    Crops are keyed by the top of the line band, not by the consensus line: a backend
    returns a paragraph per entry, so one consensus line covers several written lines and
    each of them needs its own crop.
    """
    wanted = {slot.bbox[1] for slot in result.ballot.slots if slot.flagged and slot.bbox}
    if not wanted:
        return {}
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for slot in result.ballot.slots:
        if slot.bbox is None or slot.bbox[1] not in wanted:
            continue
        x, y, width, height = slot.bbox
        current = boxes.get(y)
        if current is None:
            boxes[y] = (x, y, x + width, y + height)
        else:
            boxes[y] = (
                min(current[0], x),
                y,
                max(current[2], x + width),
                max(current[3], y + height),
            )

    crops: dict[str, dict[str, Any]] = {}
    try:
        with Image.open(result.image_path) as page:
            for top, box in sorted(boxes.items()):
                crop = _crop_box(page, box)
                if crop is not None:
                    crops[str(top)] = crop
    except (OSError, UnidentifiedImageError, ValueError):
        return {}
    return crops


def _crop_box(page: Image.Image, box: tuple[int, int, int, int]) -> dict[str, Any] | None:
    """Inline one line region as a base64 JPEG, with the geometry to draw a box on it."""
    left = max(0, box[0] - CROP_PADDING)
    top = max(0, box[1] - CROP_PADDING)
    right = min(page.width, box[2] + CROP_PADDING)
    bottom = min(page.height, box[3] + CROP_PADDING)
    if right <= left or bottom <= top:
        return None
    crop = page.crop((left, top, right, bottom)).convert("RGB")
    scale = 1.0
    if crop.width > CROP_MAX_WIDTH:
        scale = CROP_MAX_WIDTH / crop.width
        crop = crop.resize((CROP_MAX_WIDTH, max(1, round(crop.height * scale))))
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=CROP_QUALITY, optimize=True)
    return {
        "src": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "origin": [left, top],
        "scale": round(scale, 5),
        "size": [crop.width, crop.height],
    }
