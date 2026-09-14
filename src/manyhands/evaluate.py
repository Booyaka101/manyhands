"""Scoring the flags against ALTO or PAGE-XML ground truth.

The question this answers is not "how good is the consensus", it is "if a human only
looks at the words manyhands highlighted, how many of the real errors do they see, and
how much do they have to read to see them". So the two headline numbers are the
error-capture rate and the flag rate, with CER alongside for context.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path, PureWindowsPath
from typing import Any

from manyhands.pipeline import IMAGE_SUFFIXES
from manyhands.text import fold, levenshtein
from manyhands.vote import Ballot, group_by_line

ALTO_NAMESPACE_HINT = "alto"
PAGE_NAMESPACE_HINT = "pagecontent"

#: The bar the brief set: catch most of the real errors without flagging most of the page.
TARGET_CAPTURE = 0.70
TARGET_FLAG = 0.20

_WHITESPACE = re.compile(r"[ \t]+")


class GroundTruthError(RuntimeError):
    """Raised when a ground-truth file cannot be read as ALTO or PAGE-XML."""


@dataclass(slots=True)
class GroundTruth:
    """Transcribed lines taken from one ground-truth file.

    :param lines: Line transcriptions in document order.
    :param image: The image file name the XML points at, when it names one.
    :param schema: ``"alto"`` or ``"page"``.
    """

    lines: list[str]
    image: str | None
    schema: str

    @property
    def text(self) -> str:
        """Return the ground-truth page as plain text."""
        return "\n".join(self.lines)


@dataclass(slots=True)
class PageScore:
    """How the flags did on one page.

    :param page: Page name.
    :param slots: Number of consensus slots.
    :param flagged: Number of slots the backends disagreed on.
    :param error_chars: Consensus characters that differ from the ground truth.
    :param captured_chars: Of those, how many sit inside a flagged slot.
    :param cer: Character error rate of the consensus against the ground truth.
    :param gt_chars: Length of the ground-truth text.
    """

    page: str
    slots: int
    flagged: int
    error_chars: int
    captured_chars: int
    cer: float
    gt_chars: int

    @property
    def flag_rate(self) -> float:
        """Return the share of slots that were flagged."""
        return self.flagged / self.slots if self.slots else 0.0

    @property
    def capture_rate(self) -> float:
        """Return the share of erroneous characters that a flagged slot covers."""
        return self.captured_chars / self.error_chars if self.error_chars else 1.0


@dataclass(slots=True)
class DatasetScore:
    """Aggregate scores over a dataset.

    :param pages: Per-page scores in dataset order.
    """

    pages: list[PageScore] = field(default_factory=list)

    @property
    def slots(self) -> int:
        """Return the total number of slots scored."""
        return sum(page.slots for page in self.pages)

    @property
    def flagged(self) -> int:
        """Return the total number of flagged slots."""
        return sum(page.flagged for page in self.pages)

    @property
    def error_chars(self) -> int:
        """Return the total number of erroneous consensus characters."""
        return sum(page.error_chars for page in self.pages)

    @property
    def captured_chars(self) -> int:
        """Return the total number of erroneous characters inside a flagged slot."""
        return sum(page.captured_chars for page in self.pages)

    @property
    def flag_rate(self) -> float:
        """Return the share of all slots that were flagged."""
        return self.flagged / self.slots if self.slots else 0.0

    @property
    def capture_rate(self) -> float:
        """Return the share of all erroneous characters that were flagged."""
        return self.captured_chars / self.error_chars if self.error_chars else 1.0

    @property
    def cer(self) -> float:
        """Return the character error rate over the whole dataset."""
        total = sum(page.gt_chars for page in self.pages)
        if not total:
            return 0.0
        return sum(page.cer * page.gt_chars for page in self.pages) / total


def load_ground_truth(path: Path) -> GroundTruth:
    """Read one ALTO or PAGE-XML file into ordered line transcriptions.

    :param path: Path to the XML file.
    :raises GroundTruthError: If the file is unreadable or is neither schema.
    """
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError) as exc:
        message = f"cannot read ground truth {path.name}: {exc}"
        raise GroundTruthError(message) from exc

    root = tree.getroot()
    namespace = root.tag.split("}")[0].strip("{").casefold() if "}" in root.tag else ""
    local = root.tag.split("}")[-1].casefold()

    if local == "alto" or ALTO_NAMESPACE_HINT in namespace:
        return _read_alto(root)
    if local == "pcgts" or PAGE_NAMESPACE_HINT in namespace:
        return _read_page(root)
    message = (
        f"{path.name} is neither ALTO nor PAGE-XML (root element <{local}>). "
        "manyhands eval reads the two schemas HTR-United datasets publish."
    )
    raise GroundTruthError(message)


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _read_alto(root: ET.Element) -> GroundTruth:
    lines: list[str] = []
    for element in root.iter():
        if _local(element.tag) != "TextLine":
            continue
        pieces: list[str] = []
        spaced = False
        for child in element:
            name = _local(child.tag)
            if name == "String":
                pieces.append(child.get("CONTENT", ""))
            elif name == "SP":
                pieces.append(" ")
                spaced = True
            elif name == "HYP":
                # A line-break hyphen belongs to the word in front of it, never spaced off.
                glyph = child.get("CONTENT", "-")
                if pieces:
                    pieces[-1] += glyph
                else:
                    pieces.append(glyph)
        # Some exporters put one String per word with explicit <SP/>, others put the whole
        # line in a single String. Only join on spaces when the file did not supply them.
        joined = "".join(pieces) if spaced else " ".join(piece for piece in pieces if piece)
        text = _normalise_gt(joined)
        if text:
            lines.append(text)

    image = None
    for element in root.iter():
        if _local(element.tag) == "fileName" and element.text:
            image = element.text.strip()
            break
    return GroundTruth(lines=lines, image=image, schema="alto")


def _read_page(root: ET.Element) -> GroundTruth:
    lines: list[str] = []
    for element in root.iter():
        if _local(element.tag) != "TextLine":
            continue
        text = ""
        for equiv in element:
            if _local(equiv.tag) != "TextEquiv":
                continue
            for unicode_element in equiv:
                if _local(unicode_element.tag) == "Unicode" and unicode_element.text:
                    text = unicode_element.text
                    break
            if text:
                break
        cleaned = _normalise_gt(text)
        if cleaned:
            lines.append(cleaned)

    image = None
    for element in root.iter():
        if _local(element.tag) == "Page":
            image = element.get("imageFilename")
            break
    return GroundTruth(lines=lines, image=image, schema="page")


def _normalise_gt(text: str) -> str:
    return _WHITESPACE.sub(" ", fold(text).replace("\n", " ")).strip()


def consensus_spans(ballot: Ballot) -> tuple[str, list[tuple[int, int, bool]]]:
    """Render the consensus and record where each slot's characters landed.

    A slot the backends disagreed about but left empty still gets a span, a zero-width
    one, because the report highlights it and a reviewer would look there.

    The text this returns is ``consensus.txt`` folded, character for character, so the
    score describes the file the run actually shipped.

    :returns: The consensus text, and one ``(start, end, flagged)`` per slot with a place
        in the text.
    """
    pieces: list[str] = []
    spans: list[tuple[int, int, bool]] = []
    cursor = 0
    for line in group_by_line(ballot.slots):
        started = False
        for slot in line:
            if not slot.consensus:
                if slot.flagged:
                    spans.append((cursor, cursor, True))
                continue
            separator = " " if started else ("\n" if pieces else "")
            pieces.append(separator)
            cursor += len(separator)
            started = True
            text = fold(slot.consensus)
            pieces.append(text)
            spans.append((cursor, cursor + len(text), slot.flagged))
            cursor += len(text)
    return "".join(pieces), spans


def score_page(page: str, ballot: Ballot, truth: GroundTruth) -> PageScore:
    """Score one page's flags against its ground truth."""
    hypothesis, spans = consensus_spans(ballot)
    reference = "\n".join(truth.lines)

    flagged_chars = bytearray(len(hypothesis))
    flagged_gaps: set[int] = set()
    for start, end, flagged in spans:
        if not flagged:
            continue
        if end == start:
            flagged_gaps.add(start)
            continue
        for position in range(start, min(end, len(hypothesis))):
            flagged_chars[position] = 1

    error_chars = 0
    captured = 0
    matcher = SequenceMatcher(a=hypothesis, b=reference, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            for position in range(i1, i2):
                error_chars += 1
                captured += flagged_chars[position]
        if tag in {"replace", "insert"}:
            missing = (j2 - j1) - (i2 - i1) if tag == "replace" else j2 - j1
            if missing > 0:
                # Characters the consensus dropped are charged to the slot either side of
                # the hole, since that is the word a reviewer would have to look at.
                neighbours = [p for p in (i1 - 1, i1) if 0 <= p < len(hypothesis)]
                seen = any(flagged_chars[p] for p in neighbours) or bool(
                    flagged_gaps & {i1 - 1, i1, i1 + 1}
                )
                error_chars += missing
                captured += missing if seen else 0

    cer = levenshtein(hypothesis, reference) / len(reference) if reference else 0.0
    return PageScore(
        page=page,
        slots=len(ballot.slots),
        flagged=sum(1 for slot in ballot.slots if slot.flagged),
        error_chars=error_chars,
        captured_chars=captured,
        cer=cer,
        gt_chars=len(reference),
    )


def mismatched_image(image_path: Path, truth: GroundTruth) -> str | None:
    """Return a warning when a ground-truth file names a page other than the one it was paired with.

    Pairing is by file stem, so a dataset that renamed its images without rewriting the
    XML scores a real transcript against the wrong page and reports it as error.

    :returns: The warning, or ``None`` when the pair is consistent or the XML names nothing.
    """
    if not truth.image:
        return None
    # PureWindowsPath so a path written with either separator yields the same file name.
    named = PureWindowsPath(truth.image).name
    if PureWindowsPath(named).stem.casefold() == image_path.stem.casefold():
        return None
    return (
        f"{image_path.name} is paired with ground truth that names {named}. "
        "Pairing is by file stem, so this scores against a different page."
    )


def find_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    """Match page images to their ground-truth XML by file stem, recursively.

    :param dataset: Dataset folder.
    :returns: ``(image, xml)`` pairs sorted by image name.
    :raises GroundTruthError: If the folder holds no matched pair.
    """
    if not dataset.is_dir():
        message = f"not a folder: {dataset}"
        raise GroundTruthError(message)

    images: dict[str, Path] = {}
    xmls: dict[str, Path] = {}
    for path in sorted(dataset.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix in IMAGE_SUFFIXES:
            images.setdefault(path.stem, path)
        elif suffix == ".xml":
            xmls.setdefault(path.stem, path)

    pairs = [(images[stem], xmls[stem]) for stem in sorted(images) if stem in xmls]
    if not pairs:
        message = (
            f"{dataset} holds no image/XML pairs. manyhands eval expects each page image "
            "next to an ALTO or PAGE-XML file with the same stem, for example "
            "page001.jpg and page001.xml."
        )
        raise GroundTruthError(message)
    return pairs


def format_report(
    score: DatasetScore,
    *,
    target_capture: float = TARGET_CAPTURE,
    target_flag: float = TARGET_FLAG,
) -> str:
    """Format the dataset score as the block ``manyhands eval`` prints."""
    lines = [
        f"pages scored          {len(score.pages)}",
        f"slots                 {score.slots}",
        f"flagged slots         {score.flagged}",
        "",
        f"error-capture rate    {score.capture_rate:.1%}  "
        f"({score.captured_chars} of {score.error_chars} wrong characters sit in a flagged word)",
        f"flag rate             {score.flag_rate:.1%}  "
        f"({score.flagged} of {score.slots} words highlighted for review)",
        f"consensus CER         {score.cer:.1%}",
        "",
        f"target                capture >= {target_capture:.0%}, flag rate < {target_flag:.0%}  "
        f"[{'met' if meets_target(score, target_capture, target_flag) else 'not met'}]",
    ]
    return "\n".join(lines)


def meets_target(score: DatasetScore, target_capture: float, target_flag: float) -> bool:
    """Return True when a dataset score clears both halves of the target."""
    return score.capture_rate >= target_capture and score.flag_rate < target_flag


def score_payload(
    score: DatasetScore,
    dataset: Path,
    backends: Sequence[str],
    *,
    target_capture: float = TARGET_CAPTURE,
    target_flag: float = TARGET_FLAG,
) -> dict[str, Any]:
    """Build the document ``manyhands eval --json`` writes."""
    return {
        "dataset": str(dataset),
        "backends": list(backends),
        "target_capture": target_capture,
        "target_flag": target_flag,
        "met": meets_target(score, target_capture, target_flag),
        "capture_rate": round(score.capture_rate, 4),
        "flag_rate": round(score.flag_rate, 4),
        "cer": round(score.cer, 4),
        "slots": score.slots,
        "flagged": score.flagged,
        "error_chars": score.error_chars,
        "captured_chars": score.captured_chars,
        "pages": [
            {
                "page": page.page,
                "slots": page.slots,
                "flagged": page.flagged,
                "capture_rate": round(page.capture_rate, 4),
                "flag_rate": round(page.flag_rate, 4),
                "cer": round(page.cer, 4),
                "error_chars": page.error_chars,
                "captured_chars": page.captured_chars,
                "gt_chars": page.gt_chars,
            }
            for page in score.pages
        ],
    }


def format_pages(pages: Sequence[PageScore]) -> str:
    """Format the per-page table ``manyhands eval --per-page`` prints."""
    header = f"{'page':<34}{'slots':>7}{'flagged':>9}{'flag%':>8}{'capture%':>10}{'CER%':>8}"
    rows = [
        f"{page.page[:33]:<34}{page.slots:>7}{page.flagged:>9}"
        f"{page.flag_rate * 100:>7.1f}%{page.capture_rate * 100:>9.1f}%{page.cer * 100:>7.1f}%"
        for page in pages
    ]
    return "\n".join([header, "-" * len(header), *rows])
