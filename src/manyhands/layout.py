"""Line-band detection, and the slot-to-pixel mapping the report crops use.

The backends return text, not coordinates, so manyhands finds the line bands itself. A
horizontal ink profile gives the vertical extent of each written line, and the consensus
text is then laid over those bands by character count. The band is measured; the position
along it is an estimate, which is why the report shows the whole line crop with the slot
marked rather than a tight crop.

A two-column page produces bands that span both columns. v1 documents that limit rather
than guessing a cross-column reading order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Pages are profiled at this longest-side size; bands are scaled back afterwards.
_PROFILE_MAX_DIM = 1600

#: An edge row or column at least this inked is the scanner's dark frame, not writing.
_BORDER_INK_FRACTION = 0.5

#: Once a frame is found, its softened edge is trimmed down to this share as well.
_BORDER_RAMP_FRACTION = 0.12

#: A row joins a line's core when it holds at least this share of the busiest row's ink.
_ROW_INK_FRACTION = 0.06

#: A core then grows outward over every row holding at least this share.
_BAND_EDGE_FRACTION = 0.01

#: Within a band, ink is counted over a character-wide window and the columns holding
#: less than this share of the busiest window are not part of the line.
_COLUMN_INK_FRACTION = 0.1

#: Bands thinner than this share of the profiled page height are specks, not lines.
_MIN_BAND_FRACTION = 0.004


@dataclass(frozen=True, slots=True)
class Band:
    """The pixel extent of one detected line of writing, in original image coordinates."""

    x: int
    y: int
    width: int
    height: int


def detect_bands(image: Image.Image) -> list[Band]:
    """Return the line bands of a page image, top to bottom.

    :param image: The page image.
    :returns: One band per detected line of writing. Empty for a blank page.
    """
    grey = image.convert("L")
    scale = max(grey.width, grey.height) / _PROFILE_MAX_DIM
    if scale > 1.0:
        grey = grey.resize((max(1, round(grey.width / scale)), max(1, round(grey.height / scale))))
    else:
        scale = 1.0

    pixels = np.asarray(grey, dtype=np.uint8)
    # Inclusive: the Otsu level is the top of the ink class, and a bitonal scan puts
    # every ink pixel at exactly that level.
    ink = pixels <= _otsu_threshold(pixels)
    height, width = ink.shape
    top, bottom = _solid_edges(ink.sum(axis=1) / max(1, width))
    left_edge, right_edge = _solid_edges(ink.sum(axis=0) / max(1, height))
    body = ink[top:bottom, left_edge:right_edge]
    rows = body.sum(axis=1)
    if not rows.any():
        return []

    peak = int(rows.max())
    min_height = max(2, round(body.shape[0] * _MIN_BAND_FRACTION))
    cores = [
        run
        for run in _runs(rows >= max(1.0, peak * _ROW_INK_FRACTION), gap=min_height // 2)
        if run[1] - run[0] >= min_height
    ]
    bands: list[Band] = []
    for start, stop in _grow_cores(cores, rows, max(1.0, peak * _BAND_EDGE_FRACTION)):
        # The frame fades into the page rather than stopping at a row, so a band that
        # reaches the trimmed edge is the rest of the frame and not a line of writing.
        if (start == 0 and top > 0) or (stop == rows.size and bottom < height):
            continue
        span = _ink_span(body[start:stop].sum(axis=0), stop - start)
        if span is None:
            continue
        left, right = span
        bands.append(
            Band(
                x=round((left_edge + left) * scale),
                y=round((top + start) * scale),
                width=max(1, round((right - left) * scale)),
                height=max(1, round((stop - start) * scale)),
            )
        )
    return bands


def _ink_span(columns: np.ndarray, reach: int) -> tuple[int, int] | None:
    """Return the ``(start, stop)`` columns a line of writing occupies.

    A speck, a gutter smudge or the show-through from the facing page would otherwise
    stretch the band across the whole scan, so ink is counted over a window about one
    character wide and the quiet ends of the line are dropped.
    """
    window = np.convolve(columns, np.ones(max(1, reach)), "same")
    if not window.any():
        return None
    inked = np.flatnonzero((columns > 0) & (window >= window.max() * _COLUMN_INK_FRACTION))
    if inked.size == 0:
        return None
    return int(inked[0]), int(inked[-1]) + 1


def _solid_edges(profile: np.ndarray) -> tuple[int, int]:
    """Return the ``(start, stop)`` of ``profile`` with its scan-border ends removed."""
    start, stop = _trim_ends(profile, 0, profile.size, _BORDER_INK_FRACTION)
    if start or stop < profile.size:
        # The frame is anti-aliased into the page. Left in, that ramp reads as a line of
        # writing along the edge.
        start, stop = _trim_ends(profile, start, stop, _BORDER_RAMP_FRACTION)
    return start, stop


def _trim_ends(profile: np.ndarray, start: int, stop: int, threshold: float) -> tuple[int, int]:
    """Move ``start`` and ``stop`` inward past every value at or above ``threshold``."""
    while start < stop and profile[start] >= threshold:
        start += 1
    while stop > start and profile[stop - 1] >= threshold:
        stop -= 1
    return start, stop


def _grow_cores(
    cores: list[tuple[int, int]], rows: np.ndarray, faint: float
) -> list[tuple[int, int]]:
    """Grow each line core over its faint rows, stopping halfway to the next core.

    The core is the dense middle of a line of writing. Ascenders, descenders and the tail
    of a long s live in the faint rows either side, and a crop without them is unreadable.
    """
    grown: list[tuple[int, int]] = []
    for index, (start, stop) in enumerate(cores):
        upper = (cores[index - 1][1] + start) // 2 if index else 0
        lower = (stop + cores[index + 1][0] + 1) // 2 if index + 1 < len(cores) else rows.size
        while start > upper and rows[start - 1] > faint:
            start -= 1
        while stop < lower and rows[stop] > faint:
            stop += 1
        grown.append((start, stop))
    return grown


def _otsu_threshold(pixels: np.ndarray) -> int:
    """Return the grey level that best separates ink from paper."""
    histogram = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 128
    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(histogram)
    weight_high = total - weight_low
    sum_low = np.cumsum(histogram * levels)
    sum_total = sum_low[-1]
    valid = (weight_low > 0) & (weight_high > 0)
    if not valid.any():
        return 128
    mean_low = np.divide(sum_low, weight_low, out=np.zeros(256), where=valid)
    mean_high = np.divide(sum_total - sum_low, weight_high, out=np.zeros(256), where=valid)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -1.0
    return int(np.argmax(between))


def _runs(mask: np.ndarray, *, gap: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` runs of True, merging runs separated by at most ``gap``."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > max(1, gap))
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    stops = np.concatenate((indices[breaks], [indices[-1]])) + 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def place_slots(
    lines: Sequence[Sequence[str]], bands: Sequence[Band]
) -> list[list[tuple[int, int, int, int]]]:
    """Estimate a box for every consensus word on a page.

    A document VLM returns a paragraph per entry, not a box per word, so one consensus
    line routinely runs over several written lines. The page is treated as a ribbon:
    bands are handed out to lines in proportion to their character count, then the words
    of a line are laid along the bands it was given.

    :param lines: The consensus words of each line, in reading order.
    :param bands: The detected line bands, top to bottom.
    :returns: One ``(x, y, width, height)`` box per word, grouped as ``lines`` was.
    """
    if not bands:
        return [[] for _ in lines]
    runs = _share_bands([_ribbon_length(words) for words in lines], len(bands))
    return [
        _lay_out(words, bands[start:stop])
        for words, (start, stop) in zip(lines, runs, strict=True)
    ]


def _ribbon_length(words: Sequence[str]) -> int:
    """Return the character length of a line, counting one space between words."""
    return sum(max(1, len(word)) for word in words) + max(0, len(words) - 1)


def _share_bands(lengths: Sequence[int], count: int) -> list[tuple[int, int]]:
    """Split ``count`` bands between lines in proportion to their length.

    A line always gets at least one band. With more lines than bands, later lines reuse
    the last band rather than falling off the page.
    """
    total = sum(lengths) or len(lengths) or 1
    runs: list[tuple[int, int]] = []
    cursor = 0.0
    stop = 0
    for length in lengths:
        start = min(stop, count - 1)
        cursor += count * max(1, length) / total
        stop = min(count, max(start + 1, round(cursor)))
        runs.append((start, stop))
    return runs


def _lay_out(words: Sequence[str], bands: Sequence[Band]) -> list[tuple[int, int, int, int]]:
    """Spread one line's words across the bands it covers, in proportion to their length."""
    if not words or not bands:
        return []
    widths = [max(1, len(word)) for word in words]
    total = sum(widths) + len(widths) - 1
    capacity = sum(band.width for band in bands)
    boxes: list[tuple[int, int, int, int]] = []
    cursor = 0
    for width in widths:
        boxes.append(
            _span(bands, capacity * cursor / total, capacity * (cursor + width) / total)
        )
        cursor += width + 1
    return boxes


def _span(bands: Sequence[Band], start: float, stop: float) -> tuple[int, int, int, int]:
    """Return the box for a ribbon interval, on whichever band the interval opens."""
    offset = 0
    for band in bands[:-1]:
        if start < offset + band.width:
            break
        offset += band.width
    else:
        band = bands[-1]
    left = band.x + min(band.width - 1, round(start - offset))
    right = band.x + min(band.width, round(stop - offset))
    return (left, band.y, max(1, right - left), band.height)
