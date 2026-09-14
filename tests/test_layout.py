"""Line-band detection and the slot-to-pixel mapping behind the report crops."""

from __future__ import annotations

from PIL import Image, ImageDraw

from conftest import PAGES
from manyhands.layout import Band, detect_bands, place_slots


def ruled_page(lines: int = 4, *, width: int = 800, height: int = 600) -> Image.Image:
    """A white page with evenly spaced black bars standing in for lines of writing."""
    page = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(page)
    pitch = height // (lines + 1)
    for index in range(lines):
        top = pitch // 2 + index * pitch
        draw.rectangle([width // 10, top, width - width // 10, top + pitch // 3], fill=0)
    return page


def test_a_blank_page_has_no_bands():
    assert detect_bands(Image.new("L", (400, 300), 255)) == []


def test_each_drawn_line_becomes_one_band():
    bands = detect_bands(ruled_page(4))

    assert len(bands) == 4
    assert [band.y for band in bands] == sorted(band.y for band in bands)
    assert all(30 < band.height < 50 for band in bands)
    assert all(70 < band.x < 90 for band in bands)


def test_a_band_stops_at_the_ink_not_at_the_page_edge():
    band = detect_bands(ruled_page(1, width=800))[0]

    assert band.x >= 70
    assert band.x + band.width <= 730


def test_specks_are_not_mistaken_for_lines():
    page = ruled_page(2)
    ImageDraw.Draw(page).point([(400, 30), (402, 32)], fill=0)

    assert len(detect_bands(page)) == 2


def test_a_large_page_is_profiled_downscaled_but_reports_original_coordinates():
    bands = detect_bands(ruled_page(3, width=4000, height=3000))
    drawn = [375, 1125, 1875]

    assert len(bands) == 3
    assert all(abs(band.y - top) <= 4 for band, top in zip(bands, drawn, strict=True))
    assert bands[0].width > 3000


def test_a_scan_border_does_not_swallow_the_lines():
    plain = ruled_page(4)
    framed = plain.copy()
    ImageDraw.Draw(framed).rectangle([0, 0, framed.width - 1, framed.height - 1], outline=0, width=6)

    assert [band.height for band in detect_bands(framed)] == [band.height for band in detect_bands(plain)]


def test_a_band_reaches_past_the_dense_core_to_the_ascenders():
    page = Image.new("L", (800, 300), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle([100, 140, 700, 170], fill=0)
    draw.rectangle([120, 110, 150, 140], fill=0)

    band = detect_bands(page)[0]

    assert band.y <= 110
    assert band.y + band.height >= 170


def test_a_real_scanned_page_yields_plausible_lines():
    with Image.open(PAGES / "journal-p002.jpg") as page:
        bands = detect_bands(page)
        height = page.height

    assert 10 <= len(bands) <= 60
    assert all(0 <= band.y < band.y + band.height <= height for band in bands)


def test_each_line_gets_the_band_it_was_written_on():
    bands = [Band(0, y, 100, 20) for y in (0, 40, 80)]
    boxes = place_slots([["one"], ["two"], ["three"]], bands)

    assert [box[0][1] for box in boxes] == [0, 40, 80]


def test_a_long_line_is_laid_over_the_several_bands_it_covers():
    bands = [Band(0, y, 100, 20) for y in (0, 40, 80, 120)]
    boxes = place_slots([["Samedi"], ["a " * 40], ["Lundi"]], bands)

    assert boxes[0][0][1] == 0
    assert boxes[1][0][1] == 40
    assert boxes[2][0][1] == 120


def test_a_line_that_runs_on_puts_its_later_words_on_a_later_band():
    boxes = place_slots(
        [["Buried", "this", "day", "John", "Fenwick"]],
        [
            Band(0, 0, 100, 20),
            Band(0, 40, 100, 20),
        ],
    )[0]

    assert boxes[0][1] == 0
    assert boxes[-1][1] == 40


def test_more_lines_than_bands_falls_back_to_the_last_band():
    boxes = place_slots([["a"], ["b"], ["c"]], [Band(0, 0, 100, 20)])

    assert [box[0][1] for box in boxes] == [0, 0, 0]


def test_a_page_with_no_bands_places_nothing():
    assert place_slots([["a"], ["b"]], []) == [[], []]


def test_words_tile_their_band_in_order_without_overlapping():
    band = Band(x=100, y=50, width=300, height=40)
    boxes = place_slots([["Buried", "this", "day", "John", "Fenwick"]], [band])[0]

    assert len(boxes) == 5
    assert boxes[0][0] == band.x
    assert boxes[-1][0] + boxes[-1][2] == band.x + band.width
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left[0] + left[2] <= right[0]
    assert all(y == band.y and height == band.height for _, y, _, height in boxes)


def test_a_longer_word_gets_a_wider_box():
    boxes = place_slots([["a", "Fenwickshire"]], [Band(0, 0, 260, 20)])[0]

    assert boxes[1][2] > boxes[0][2] * 4


def test_an_empty_line_has_no_boxes():
    assert place_slots([[]], [Band(0, 0, 100, 20)]) == [[]]


def test_an_empty_word_still_gets_a_box_to_click():
    boxes = place_slots([["", "word"]], [Band(0, 0, 100, 20)])[0]

    assert len(boxes) == 2
    assert boxes[0][2] >= 1
