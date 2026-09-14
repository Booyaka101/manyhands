"""Walking a folder, running the ensemble over every page, and writing the artifacts."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from manyhands import report
from manyhands.align import align
from manyhands.backends import (
    BackendLike,
    Transcript,
    cache_dir,
    read_cached,
    repetition_ratio,
)
from manyhands.layout import detect_bands, place_slots
from manyhands.text import plural
from manyhands.vote import Ballot, Slot, render_consensus, tally

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}

#: A backend whose token count exceeds this multiple of the page median is not reading
#: the same page as the others, so it is dropped from the vote.
OUTLIER_RATIO = 3.0

#: And one under this share of the median stopped somewhere down the page, so its silence
#: over the rest is not a disagreement either.
THIN_RATIO = 1 / 3

#: Above this share of looped n-grams the transcript is a hallucination, not a reading.
REPETITION_CUTOFF = 0.5

#: A page where every backend read at most this many tokens is treated as blank.
BLANK_TOKENS = 2

#: Below this many usable backends the outlier rules are suspended; there is nothing to
#: compare against and dropping one would leave no vote at all.
MIN_VOTERS = 3


@dataclass(slots=True)
class PageResult:
    """Everything one page produced.

    :param name: The page's original file name.
    :param stem: The page's slug, used for the output folder and the cache.
    :param image_path: Path to the image actually transcribed.
    :param ballot: The voted page.
    :param transcripts: Every transcript, including the failed and dropped ones.
    :param dropped: Backend name to the reason it was kept out of the vote.
    :param blank: True when every backend read the page as empty.
    :param out_dir: Folder the artifacts were written to.
    """

    name: str
    stem: str
    image_path: Path
    ballot: Ballot
    transcripts: list[Transcript]
    dropped: dict[str, str] = field(default_factory=dict)
    blank: bool = False
    out_dir: Path | None = None

    @property
    def flagged(self) -> int:
        """Return the number of slots the backends did not agree on."""
        return sum(1 for slot in self.ballot.slots if slot.flagged)


class PipelineError(RuntimeError):
    """Raised for input problems the user needs to fix before a run can start."""


def discover_pages(folder: Path, workspace: Path) -> list[tuple[str, Path]]:
    """Find every page in a folder, rasterising PDFs into the workspace.

    :param folder: Folder of page images and/or PDFs.
    :param workspace: The ``.manyhands`` workspace, used to hold rasterised PDF pages.
    :returns: ``(display name, image path)`` pairs in a stable order.
    :raises PipelineError: If the folder is missing or holds nothing readable.
    """
    if not folder.exists():
        message = f"no such folder: {folder}"
        raise PipelineError(message)
    if not folder.is_dir():
        message = f"not a folder: {folder}"
        raise PipelineError(message)

    pages: list[tuple[str, Path]] = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.casefold()
        if suffix in IMAGE_SUFFIXES:
            pages.append((path.name, path))
        elif suffix in PDF_SUFFIXES:
            pages.extend(rasterise_pdf(path, workspace / "pdf-pages"))

    if not pages:
        wanted = ", ".join(sorted(IMAGE_SUFFIXES | PDF_SUFFIXES))
        message = f"{folder} holds no pages. manyhands reads: {wanted}"
        raise PipelineError(message)
    return pages


def rasterise_pdf(pdf_path: Path, out_dir: Path, *, dpi: int = 300) -> list[tuple[str, Path]]:
    """Render a PDF to one PNG per page, skipping pages already rendered.

    :raises PipelineError: If the PDF cannot be opened.
    """
    try:
        import pypdfium2
    except ImportError as exc:  # pragma: no cover - pypdfium2 is a hard dependency
        message = "reading PDFs needs pypdfium2: pip install pypdfium2"
        raise PipelineError(message) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        document = pypdfium2.PdfDocument(pdf_path)
    except Exception as exc:  # noqa: BLE001 - pypdfium2 raises several unrelated types
        message = f"cannot read {pdf_path.name}: {exc}"
        raise PipelineError(message) from exc

    pages: list[tuple[str, Path]] = []
    try:
        for index in range(len(document)):
            target = out_dir / f"{pdf_path.stem}-p{index + 1:04d}.png"
            if not target.is_file():
                page = document[index]
                page.render(scale=dpi / 72).to_pil().save(target)
            pages.append((f"{pdf_path.name}#{index + 1}", target))
    finally:
        document.close()
    return pages


def transcribe_pages(
    pages: Sequence[tuple[str, Path]],
    backends: Sequence[BackendLike],
    workspace: Path,
    *,
    reuse: bool = True,
    on_transcript: Callable[[str, Transcript], None] | None = None,
) -> list[list[Transcript]]:
    """Transcribe every page with every backend, one model at a time.

    Backend-major order, not page-major: the default five-model ensemble does not fit in
    24 GB of VRAM together, so each model is loaded, run over the whole folder, and
    released before the next one loads. Every transcript is cached the moment it lands
    and a cached page is not read again, so a run killed halfway resumes for free.

    :param pages: ``(display name, image path)`` pairs.
    :param backends: Resolved backends, in rank order.
    :param workspace: The ``.manyhands`` workspace holding the transcript cache.
    :param reuse: Set False to re-transcribe pages that are already in the cache.
    :param on_transcript: Called with ``(page name, transcript)`` as each one lands.
    :returns: One list of transcripts per page, in backend rank order.
    """
    per_page: list[list[Transcript]] = [[] for _ in pages]
    for backend in backends:
        directory = cache_dir(workspace, backend.model_id)
        try:
            for index, (name, image_path) in enumerate(pages):
                transcript = read_cached(directory, image_path.stem) if reuse else None
                if transcript is None:
                    transcript = backend.transcribe(image_path)
                    if transcript.ok:
                        cache_transcript(transcript, image_path.stem, workspace)
                else:
                    # The vote labels backends by what the user asked for, not by whatever
                    # name the cache file was written under.
                    transcript.backend = backend.model_id
                per_page[index].append(transcript)
                if on_transcript is not None:
                    on_transcript(name, transcript)
        finally:
            backend.release()
    return per_page


def vote_page(
    name: str,
    image_path: Path,
    transcripts: Sequence[Transcript],
    *,
    glossary: dict[str, str] | None = None,
    image: Image.Image | None = None,
) -> PageResult:
    """Align, vote and locate one page from transcripts that are already in hand.

    Backend failures are recorded, never raised: a page with one model down is still
    worth reading with the rest.
    """
    usable, dropped = _select_voters(list(transcripts))

    columns = align([transcript.tokens for transcript in usable])
    ballot = tally(columns, [transcript.backend for transcript in usable], glossary=glossary)
    blank = bool(usable) and all(len(transcript.tokens) <= BLANK_TOKENS for transcript in usable)
    if blank:
        ballot = Ballot(slots=[], backends=ballot.backends, blank=True)

    if not blank and ballot.slots:
        _locate_slots(ballot.slots, image_path, image=image)

    return PageResult(
        name=name,
        stem=image_path.stem,
        image_path=image_path,
        ballot=ballot,
        transcripts=list(transcripts),
        dropped=dropped,
        blank=blank,
    )


def _select_voters(transcripts: list[Transcript]) -> tuple[list[Transcript], dict[str, str]]:
    """Return the transcripts that vote, and why each of the others does not."""
    dropped: dict[str, str] = {
        transcript.backend: transcript.error or "backend failed"
        for transcript in transcripts
        if not transcript.ok
    }
    alive = [transcript for transcript in transcripts if transcript.ok]
    if any(transcript.tokens for transcript in alive):
        # A backend that returned nothing where the others found text has failed at the page,
        # whatever its exit status. Its silence would otherwise dissent from every word.
        for transcript in alive:
            if not transcript.tokens:
                dropped[transcript.backend] = "dropped from the vote: returned no text"
        alive = [transcript for transcript in alive if transcript.tokens]
    if len(alive) < MIN_VOTERS:
        return alive, dropped

    counts = [len(transcript.tokens) for transcript in alive]
    median = statistics.median(counts)
    kept: list[Transcript] = []
    for transcript in alive:
        count = len(transcript.tokens)
        looped = repetition_ratio(transcript.tokens)
        if looped > REPETITION_CUTOFF:
            dropped[transcript.backend] = (
                f"dropped from the vote: {looped:.0%} of the transcript is a repeated phrase, "
                "which is what a hallucination loop looks like"
            )
            continue
        if median > 0 and not median * THIN_RATIO <= count <= median * OUTLIER_RATIO:
            dropped[transcript.backend] = (
                f"dropped from the vote: returned {plural(count, 'token')} "
                f"against a page median of {median:.0f}, so it was not reading the same page "
                "as the others"
            )
            continue
        kept.append(transcript)

    if len(kept) < MIN_VOTERS:
        # Dropping would leave too few voters to mean anything, so keep everyone and say so.
        for transcript in alive:
            if transcript.backend in dropped:
                transcript.notes.append(
                    "kept in the vote despite looking like an outlier: too few backends left"
                )
                dropped.pop(transcript.backend, None)
        return alive, dropped
    return kept, dropped


def _locate_slots(slots: list[Slot], image_path: Path, *, image: Image.Image | None = None) -> None:
    """Attach an estimated bounding box to every slot, in place."""
    try:
        if image is not None:
            bands = detect_bands(image)
        else:
            with Image.open(image_path) as page:
                bands = detect_bands(page)
    except (OSError, UnidentifiedImageError, ValueError):
        return
    if not bands:
        return

    by_line: dict[int, list[Slot]] = {}
    for slot in slots:
        by_line.setdefault(slot.line, []).append(slot)
    ordered = [by_line[line] for line in sorted(by_line)]
    placed = place_slots([[slot.consensus or " " for slot in line] for line in ordered], bands)
    for line, boxes in zip(ordered, placed, strict=True):
        for slot, box in zip(line, boxes, strict=True):
            slot.bbox = box


def agreement_payload(result: PageResult) -> dict[str, Any]:
    """Build the ``agreement.json`` document for one page."""
    return {
        "page": result.name,
        "blank": result.blank,
        "backends": result.ballot.backends,
        "dropped": result.dropped,
        "slots": [
            {
                "i": slot.index,
                "consensus": slot.consensus,
                "agreement": round(slot.agreement, 4),
                "votes": slot.votes,
                "bbox": list(slot.bbox) if slot.bbox else None,
                "line": slot.line,
                "contested": slot.contested,
                "confirmed": slot.confirmed,
                "readings": {
                    name: (token.text if token is not None else None)
                    for name, token in slot.readings.items()
                },
            }
            for slot in result.ballot.slots
        ],
        "timings": {
            transcript.backend: round(transcript.seconds, 3) for transcript in result.transcripts
        },
    }


def write_page(result: PageResult, out_root: Path, siblings: Sequence[str] = ()) -> Path:
    """Write ``consensus.txt``, ``agreement.json`` and ``report.html`` for one page.

    :param siblings: Every page stem in the run, in page order, for the report's navigation.
    """
    out_dir = out_root / result.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "consensus.txt").write_text(render_consensus(result.ballot) + "\n", encoding="utf-8")
    (out_dir / "agreement.json").write_text(
        json.dumps(agreement_payload(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.html").write_text(
        report.render_page(result, siblings), encoding="utf-8"
    )
    result.out_dir = out_dir
    return out_dir


def cache_transcript(transcript: Transcript, stem: str, workspace: Path) -> None:
    """Store one transcript so a later run can re-vote the page without a GPU."""
    directory = cache_dir(workspace, transcript.backend)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_folder_index(out_root: Path, results: Sequence[PageResult]) -> Path:
    """Write the ``index.html`` that links every page report in a run."""
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "index.html"
    path.write_text(report.render_index(results), encoding="utf-8")
    return path


def write_index(workspace: Path, results: Sequence[PageResult]) -> None:
    """Record where each page's report went, so ``manyhands confirm`` can find it."""
    index = {
        result.stem: {
            "page": result.name,
            "image": str(result.image_path),
            "out_dir": str(result.out_dir) if result.out_dir else None,
        }
        for result in results
    }
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_folder(
    folder: Path,
    backends: Sequence[BackendLike],
    out_root: Path,
    workspace: Path,
    *,
    glossary: dict[str, str] | None = None,
    reuse: bool = True,
    on_transcript: Callable[[str, Transcript], None] | None = None,
    on_page: Callable[[PageResult], None] | None = None,
) -> list[PageResult]:
    """Run the whole pipeline over a folder and write every artifact.

    :param folder: Folder of page images and/or PDFs.
    :param backends: Resolved backends, in rank order.
    :param out_root: Folder the per-page artifacts are written under.
    :param workspace: The ``.manyhands`` workspace for cache, glossary and index.
    :param glossary: Variant key to accepted reading, consulted for contested slots.
    :param reuse: Set False to re-transcribe pages that are already in the cache.
    :param on_transcript: Called with ``(page name, transcript)`` as each one lands.
    :param on_page: Called with each page result as soon as it is written.
    :returns: One result per page, in page order.
    :raises PipelineError: If the folder holds nothing readable.
    """
    pages = discover_pages(folder, workspace)
    per_page = transcribe_pages(pages, backends, workspace, reuse=reuse, on_transcript=on_transcript)

    stems = [image_path.stem for _, image_path in pages]
    results: list[PageResult] = []
    for (name, image_path), transcripts in zip(pages, per_page, strict=True):
        result = vote_page(name, image_path, transcripts, glossary=glossary)
        write_page(result, out_root, stems)
        results.append(result)
        if on_page is not None:
            on_page(result)
    write_index(workspace, results)
    write_folder_index(out_root, results)
    return results

