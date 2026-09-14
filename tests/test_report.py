"""The structure of the HTML a page report has to have."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from conftest import PAGES, transcripts
from manyhands.pipeline import PageResult, vote_page
from manyhands.report import render_index, render_page
from manyhands.vote import Ballot, variant_key

PAGE = PAGES / "journal-p001.jpg"

FOUR_AGREE = "JOURNAL\nDE\nCELESTINE DONIAU-DANEST\nSUR LES DEBUTS DE LA GUERRE\n1914-1918"
ONE_DIFFERS = FOUR_AGREE.replace("DONIAU-DANEST", "DONIAU-DANESI")


@pytest.fixture(scope="module")
def result() -> PageResult:
    """One real page voted by five backends, four of which agree."""
    return vote_page(
        "journal-p001.jpg",
        PAGE,
        transcripts(
            ("model-a", FOUR_AGREE),
            ("model-b", FOUR_AGREE),
            ("model-c", FOUR_AGREE),
            ("model-d", FOUR_AGREE),
            ("model-e", ONE_DIFFERS),
        ),
    )


@pytest.fixture(scope="module")
def html(result: PageResult) -> str:
    return render_page(result)


def json_block(html: str, element_id: str) -> dict:
    """Return the parsed contents of one of the report's embedded JSON blocks."""
    match = re.search(rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None, f"no {element_id} block in the report"
    return json.loads(match.group(1))


def test_the_report_is_one_self_contained_file(html):
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<link" not in html
    assert 'src="http' not in html
    assert "<style>" in html


def test_the_report_links_to_the_index_and_to_the_pages_either_side(html, result):
    assert 'href="../index.html"' in html
    assert "/report.html" not in html

    middle = render_page(result, ["journal-p000", result.stem, "journal-p002"])

    assert 'href="../journal-p000/report.html"' in middle
    assert 'href="../journal-p002/report.html"' in middle


def test_every_flagged_slot_is_clickable_and_banded(html, result):
    flagged = [slot for slot in result.ballot.slots if slot.flagged]
    assert len(flagged) == 1

    for slot in flagged:
        assert f'data-slot="{slot.index}"' in html
        assert f'class="w {slot.band}"' in html

    unflagged = next(slot for slot in result.ballot.slots if not slot.flagged)
    assert f'data-slot="{unflagged.index}"' not in html


def test_the_disagreements_can_be_stepped_through_from_the_keyboard(html):
    assert "n / p step through the disagreements" in html
    assert "if (event.key === 'n') step(1);" in html
    assert "else if (event.key === 'p') step(-1);" in html


def test_the_three_bands_are_all_defined_in_the_stylesheet(html):
    for band in ("red", "amber", "pale"):
        assert f"--{band}:" in html
        assert f".w.{band}" in html


def test_slot_data_carries_one_row_per_backend(html, result):
    data = json_block(html, "slot-data")
    flagged = next(slot for slot in result.ballot.slots if slot.flagged)
    entry = data[str(flagged.index)]

    assert [name for name, _, _ in entry["readings"]] == result.ballot.backends
    assert entry["consensus"] == "DONIAU-DANEST"
    assert entry["agreement"] == pytest.approx(0.8)
    assert {name: text for name, text, _ in entry["readings"]}["model-e"] == "DONIAU-DANESI"
    assert entry["bbox"] is not None


def test_slot_data_marks_the_backends_that_won_the_slot(html, result):
    """The panel bolds winners by flag, not by string match, so a backend that agreed but
    capitalised differently still reads as agreeing."""
    data = json_block(html, "slot-data")
    flagged = next(slot for slot in result.ballot.slots if slot.flagged)
    entry = data[str(flagged.index)]

    won = {name for name, _, winner in entry["readings"] if winner}
    assert won == set(result.ballot.backends) - {"model-e"}


def test_unflagged_slots_are_left_out_of_the_click_data(html, result):
    data = json_block(html, "slot-data")
    assert set(data) == {str(slot.index) for slot in result.ballot.slots if slot.flagged}


def test_each_flagged_band_carries_an_inline_base64_crop(html, result):
    crops = json_block(html, "crop-data")
    flagged_bands = {str(slot.bbox[1]) for slot in result.ballot.slots if slot.flagged and slot.bbox}

    assert set(crops) == flagged_bands
    for crop in crops.values():
        assert crop["src"].startswith("data:image/jpeg;base64,")
        assert len(crop["size"]) == 2
        assert len(crop["origin"]) == 2


def test_an_unreadable_page_says_why_there_is_no_crop(tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg")

    readings = transcripts(("model-a", FOUR_AGREE), ("model-b", ONE_DIFFERS))
    page = vote_page("broken.jpg", broken, readings)
    html = render_page(page)

    assert all(slot.bbox is None for slot in page.ballot.slots)
    assert json_block(html, "crop-data") == {}
    assert "could not find its line on the page image" in html


def test_the_crop_is_actually_the_page_region_it_claims(html, result):
    import base64
    import io

    crops = json_block(html, "crop-data")
    crop = next(iter(crops.values()))
    payload = base64.b64decode(crop["src"].split(",", 1)[1])
    image = Image.open(io.BytesIO(payload))
    page = Image.open(PAGE)

    assert image.format == "JPEG"
    assert crop["origin"][0] >= 0
    assert crop["origin"][1] + image.height / crop["scale"] <= page.height + 32


def test_the_tick_is_only_explained_when_a_reading_was_confirmed(html, result):
    assert "you confirmed this reading" not in html

    key = variant_key(Counter({"DONIAU-DANEST": 4, "DONIAU-DANESI": 1}))
    confirmed = vote_page(result.name, PAGE, result.transcripts, glossary={key: "DONIAU-DANESI"})

    assert "you confirmed this reading" in render_page(confirmed)


def test_the_header_counts_match_the_ballot(html, result):
    assert f"<b>{len(result.ballot.slots)}</b>slots" in html
    assert f"<b>{result.flagged}</b>disagreed" in html
    assert " · ".join(result.ballot.backends) in html


def test_the_footer_states_the_limits_and_the_timings(html):
    assert "Multi-column and table pages" in html
    assert "not accuracy" in html
    assert "<h2>Timings</h2>" in html
    assert "model-a · 1.0s" in html


def test_a_dropped_backend_is_named_in_the_footer(result):
    result.dropped["model-f"] = "dropped from the vote: looped"
    try:
        html = render_page(result)
    finally:
        result.dropped.pop("model-f")

    assert "Backends kept out of this page's vote" in html
    assert "model-f" in html
    assert "dropped from the vote: looped" in html


def test_a_blank_page_says_so_and_invents_nothing():
    result = PageResult(
        name="blank.jpg",
        stem="blank",
        image_path=PAGE,
        ballot=Ballot(slots=[], backends=["model-a", "model-b"], blank=True),
        transcripts=transcripts(("model-a", ""), ("model-b", "")),
        blank=True,
    )
    html = render_page(result)

    assert "rather than whatever one model might have invented" in html
    assert "data-slot=" not in html


def test_backend_names_are_escaped_not_injected():
    result = vote_page(
        "x.jpg",
        PAGE,
        transcripts(("<script>bad</script>", "alpha"), ("model-b", "beta")),
    )
    html = render_page(result)

    assert "<script>bad</script>" not in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html


def test_the_index_links_every_page(result):
    html = render_index([result, result])

    assert html.count(f'href="{result.stem}/report.html"') == 2
    assert "<!DOCTYPE html>" in html
    assert f"{result.flagged * 2}" in html


def test_the_index_counts_dropped_backends_in_plain_english(result):
    result.dropped["model-f"] = "dropped from the vote: looped"
    try:
        one = render_index([result])
        result.dropped["model-g"] = "dropped from the vote: crashed"
        two = render_index([result])
    finally:
        result.dropped.clear()

    assert "1 backend dropped" in one
    assert "2 backends dropped" in two


def test_the_index_of_an_empty_run_still_renders():
    html = render_index([])

    assert "0 pages transcribed" in html
    assert "<table>" in html


def test_the_templates_ship_inside_the_wheel():
    import manyhands

    templates = Path(manyhands.__file__).parent / "templates"
    assert sorted(path.name for path in templates.glob("*.j2")) == [
        "index.html.j2",
        "report.html.j2",
    ]
