"""The ``manyhands`` command line."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from manyhands import __version__
from manyhands.backends import (
    ALIASES,
    DEFAULT_MODELS,
    PAGE_TIMEOUT,
    BackendUnavailable,
    Transcript,
    licence_notice,
    resolve_backends,
    resolve_model,
)
from manyhands.evaluate import (
    DatasetScore,
    GroundTruth,
    GroundTruthError,
    find_pairs,
    format_pages,
    format_report,
    load_ground_truth,
    score_page,
    score_payload,
)
from manyhands.glossary import Glossary, GlossaryError
from manyhands.pipeline import (
    PageResult,
    PipelineError,
    run_folder,
    transcribe_pages,
    vote_page,
    write_folder_index,
    write_index,
    write_page,
)
from manyhands.text import plural
from manyhands.vote import variant_key

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Run several local OCR models over the same page and show which words they "
        "disagree on. Nothing leaves the machine and no model weights ship with this tool."
    ),
)

WORKSPACE_DIRNAME = ".manyhands"

MODELS_HELP = f"How many of the {len(DEFAULT_MODELS)} default models to use."
BACKENDS_HELP = (
    "Comma-separated backends, overriding --models. Accepts Hugging Face model ids, the "
    f"aliases ({', '.join(sorted(ALIASES))}), or the name of a cached transcript folder."
)
NO_CACHE_HELP = "Ignore stored transcripts and re-run every model."
TIMEOUT_HELP = "Seconds a model may spend generating one page, 0 for no limit."


def _use_utf8_output() -> None:
    """Print readings without tripping over the console codepage.

    A redirected stdout on Windows encodes with the legacy codepage, and the archaic
    glyphs this tool exists to show are exactly what it cannot encode.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and (stream.encoding or "").lower() not in {"utf-8", "utf8"}:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _err(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.RED)


def _fail(message: str, code: int = 2) -> typer.Exit:
    _err(message)
    return typer.Exit(code=code)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"manyhands {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Print the version."),
    ] = False,
) -> None:
    """Cross-model OCR consensus for handwritten and historical pages."""
    _use_utf8_output()


def _backend_names(models: int, backends: str | None) -> list[str]:
    if backends:
        return [name.strip() for name in backends.split(",") if name.strip()]
    return list(DEFAULT_MODELS[:models])


def _report_transcript(page_name: str, transcript: Transcript) -> None:
    """Print one transcript as it lands, because a five-model run is otherwise hours of silence."""
    state = (
        f"failed: {transcript.error}"
        if not transcript.ok
        else plural(len(transcript.tokens), "token")
    )
    typer.echo(f"  {transcript.backend} · {page_name}: {state} ({transcript.seconds:.1f}s)")


def _report_page(result: PageResult) -> None:
    """Print one page's flag count and any backend kept out of its vote."""
    total = len(result.ballot.slots)
    rate = (result.flagged / total * 100) if total else 0.0
    state = "blank" if result.blank else f"{result.flagged}/{total} flagged ({rate:.0f}%)"
    typer.echo(f"  {result.name}: {state}")
    for name, reason in result.dropped.items():
        typer.secho(f"    {name}: {reason}", fg=typer.colors.YELLOW)


def _announce_licences(names: list[str]) -> None:
    for name in names:
        notice = licence_notice(resolve_model(name))
        if notice:
            typer.secho(f"notice: {notice}", fg=typer.colors.YELLOW)


@app.command()
def run(
    folder: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder of page images (png/jpg/tif) and/or PDFs.",
        ),
    ],
    models: Annotated[
        int, typer.Option("--models", "-n", min=1, max=len(DEFAULT_MODELS), help=MODELS_HELP)
    ] = len(DEFAULT_MODELS),
    backends: Annotated[
        str | None, typer.Option("--backends", "-b", help=BACKENDS_HELP)
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Where the reports go. Default: <folder>/manyhands-out")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help=NO_CACHE_HELP)] = False,
    no_glossary: Annotated[
        bool, typer.Option("--no-glossary", help="Ignore confirmed readings for this run.")
    ] = False,
    page_timeout: Annotated[
        float,
        typer.Option("--page-timeout", min=0, help=TIMEOUT_HELP),
    ] = PAGE_TIMEOUT,
) -> None:
    """Transcribe every page in FOLDER with several models and report where they differ."""
    workspace = folder / WORKSPACE_DIRNAME
    names = _backend_names(models, backends)
    if len(names) < 2:
        raise _fail("manyhands needs at least two backends; one model cannot disagree with itself.")
    _announce_licences(names)

    resolved, problems = resolve_backends(
        names, workspace, allow_cached=not no_cache, timeout=page_timeout or None
    )
    for problem in problems:
        _err(problem)
    if len(resolved) < 2:
        raise _fail("fewer than two backends resolved, so there is nothing to compare.")

    out_root = out or (folder / "manyhands-out")
    try:
        lookup = {} if no_glossary else Glossary.load(workspace / "glossary.json").as_lookup()
    except GlossaryError as exc:
        raise _fail(str(exc)) from exc

    try:
        results = run_folder(
            folder,
            resolved,
            out_root,
            workspace,
            glossary=lookup,
            reuse=not no_cache,
            on_transcript=_report_transcript,
            on_page=_report_page,
        )
    except PipelineError as exc:
        raise _fail(str(exc)) from exc
    except BackendUnavailable as exc:
        raise _fail(str(exc), code=3) from exc

    if results and not any(t.ok for result in results for t in result.transcripts):
        raise _fail(
            "every backend failed on every page, so there was nothing to compare. "
            "The reason for each is listed above.",
            code=3,
        )

    slots = sum(len(result.ballot.slots) for result in results)
    flagged = sum(result.flagged for result in results)
    typer.echo("")
    typer.secho(
        f"{plural(len(results), 'page')}, {flagged} of {slots} slots flagged "
        f"({(flagged / slots * 100) if slots else 0:.1f}%)",
        bold=True,
    )
    typer.echo(f"reports: {out_root / 'index.html'}")


@app.command()
def confirm(
    page: Annotated[
        Path,
        typer.Argument(
            exists=True, help="A page image, or the report folder manyhands wrote for it."
        ),
    ],
    slot: Annotated[
        int | None, typer.Option("--slot", help="Confirm one slot without prompting.")
    ] = None,
    reading: Annotated[
        str | None, typer.Option("--reading", help="The reading to accept for --slot.")
    ] = None,
    workspace_option: Annotated[
        Path | None, typer.Option("--workspace", help="The .manyhands folder to write into.")
    ] = None,
) -> None:
    """Record the readings you accept into the corpus glossary."""
    if reading is not None and slot is None:
        raise _fail("--reading needs --slot: it says which slot the reading is for.")
    try:
        agreement_path, workspace = _locate_page(page, workspace_option)
    except PipelineError as exc:
        raise _fail(str(exc)) from exc

    try:
        payload = json.loads(agreement_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _fail(f"cannot read {agreement_path}: {exc}") from exc

    slots = [entry for entry in payload.get("slots", []) if entry.get("agreement", 1.0) < 1.0]
    if not slots:
        name = payload.get("page", agreement_path.parent.name)
        typer.echo(f"{name}: nothing to confirm, every backend agreed.")
        return

    try:
        glossary = Glossary.load(workspace / "glossary.json")
    except GlossaryError as exc:
        raise _fail(str(exc)) from exc
    page_name = str(payload.get("page", agreement_path.parent.name))

    if slot is not None:
        entry = next((item for item in slots if item.get("i") == slot), None)
        if entry is None:
            raise _fail(f"slot {slot} is not a flagged slot on this page.")
        choice = reading if reading is not None else str(entry.get("consensus", ""))
        _record(glossary, entry, choice, page_name)
        glossary.save()
        typer.echo(f"confirmed slot {slot} as {choice!r} in {glossary.path}")
        return

    if not _interactive():
        raise _fail(
            "confirm is interactive. Run it in a terminal, or pass --slot N --reading TEXT."
        )

    typer.echo(
        f"{page_name}: {plural(len(slots), 'slot')} to review. A number accepts that reading, "
        "Enter leaves the slot alone, 'q' stops."
    )
    changed = 0
    for entry in slots:
        try:
            choice = _prompt_slot(entry)
        except _Stop:
            break
        if choice is None:
            continue
        _record(glossary, entry, choice, page_name)
        changed += 1
    glossary.save()
    typer.echo(f"{plural(changed, 'reading')} written to {glossary.path}")


def _interactive() -> bool:
    """Return True when there is a human at the keyboard to answer a prompt."""
    return sys.stdin.isatty()


class _Stop(Exception):
    """Raised when the reviewer asks to stop going through a page."""


def _prompt_slot(entry: dict[str, Any]) -> str | None:
    """Ask about one slot. Returns the accepted reading, or None to leave it alone."""
    votes: dict[str, int] = entry.get("votes", {})
    options = [reading for reading in votes if reading]
    typer.echo("")
    typer.secho(f"slot {entry.get('i')} · agreement {entry.get('agreement')}", bold=True)
    for position, option in enumerate(options, start=1):
        typer.echo(f"  {position}. {option}   ({plural(votes[option], 'vote')})")
    answer = typer.prompt("choose", default="", show_default=False).strip()
    if answer.casefold() in {"q", "quit"}:
        raise _Stop
    if not answer or answer.casefold() in {"s", "skip"}:
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return answer


def _record(glossary: Glossary, entry: dict[str, Any], choice: str, page: str) -> None:
    counts = Counter({reading: count for reading, count in entry.get("votes", {}).items()})
    key = variant_key(counts)
    glossary.confirm(key, choice, rejected=[reading for reading in counts if reading], page=page)


def _locate_page(page: Path, workspace_option: Path | None) -> tuple[Path, Path]:
    """Resolve a user-supplied page reference to its agreement.json and workspace."""
    candidates: list[Path] = []
    if page.is_dir():
        candidates.append(page / "agreement.json")
    elif page.name == "agreement.json":
        candidates.append(page)
    elif page.is_file():
        workspace = workspace_option or _find_workspace(page.parent)
        if workspace is not None:
            index_path = workspace / "index.json"
            if index_path.is_file():
                try:
                    index = json.loads(index_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    index = {}
                out_dir = (index.get(page.stem) or {}).get("out_dir")
                if out_dir:
                    candidates.append(Path(out_dir) / "agreement.json")
            candidates.append(page.parent / "manyhands-out" / page.stem / "agreement.json")

    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        message = (
            f"no report found for {page}. Run `manyhands run` on the folder first, or point "
            "confirm at the page's report folder."
        )
        raise PipelineError(message)

    workspace = workspace_option or _find_workspace(found.parent) or _find_workspace(page.parent)
    if workspace is None:
        message = f"cannot find a {WORKSPACE_DIRNAME} workspace near {page}. Pass --workspace."
        raise PipelineError(message)
    return found, workspace


def _find_workspace(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / WORKSPACE_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


@app.command("eval")
def evaluate_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder of page images each next to an ALTO or PAGE-XML file with the same stem.",
        ),
    ],
    models: Annotated[
        int, typer.Option("--models", "-n", min=1, max=len(DEFAULT_MODELS), help=MODELS_HELP)
    ] = len(DEFAULT_MODELS),
    backends: Annotated[str | None, typer.Option("--backends", "-b", help=BACKENDS_HELP)] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Score only the first N pages.")
    ] = None,
    per_page: Annotated[bool, typer.Option("--per-page", help="Print a row per page.")] = False,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Also write the per-page reports here.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the scores as JSON.")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help=NO_CACHE_HELP)] = False,
    page_timeout: Annotated[
        float, typer.Option("--page-timeout", min=0, help=TIMEOUT_HELP)
    ] = PAGE_TIMEOUT,
) -> None:
    """Score the flags against ALTO/PAGE-XML ground truth: error capture and flag rate."""
    workspace = dataset / WORKSPACE_DIRNAME
    names = _backend_names(models, backends)
    _announce_licences(names)
    resolved, problems = resolve_backends(
        names, workspace, allow_cached=not no_cache, timeout=page_timeout or None
    )
    for problem in problems:
        _err(problem)
    if len(resolved) < 2:
        raise _fail("fewer than two backends resolved, so there is nothing to compare.")

    try:
        pairs = find_pairs(dataset)
    except GroundTruthError as exc:
        raise _fail(str(exc)) from exc
    if limit is not None:
        pairs = pairs[:limit]

    truths: dict[Path, GroundTruth] = {}
    for image_path, xml_path in pairs:
        try:
            truth = load_ground_truth(xml_path)
        except GroundTruthError as exc:
            _err(str(exc))
            continue
        if not truth.lines:
            _err(f"{xml_path.name} has no transcribed lines; skipped.")
            continue
        truths[image_path] = truth

    scored_pages = [(path.name, path) for path in truths]
    if not scored_pages:
        raise _fail("no pages could be scored: every ground-truth file was unreadable or empty.")

    try:
        transcribed = transcribe_pages(
            scored_pages, resolved, workspace, reuse=not no_cache, on_transcript=_report_transcript
        )
    except BackendUnavailable as exc:
        raise _fail(str(exc), code=3) from exc

    score = DatasetScore()
    stems = [path.stem for _, path in scored_pages]
    results = []
    for (name, image_path), transcripts in zip(scored_pages, transcribed, strict=True):
        result = vote_page(name, image_path, transcripts)
        results.append(result)
        if out is not None:
            write_page(result, out, stems)
        page_score = score_page(name, result.ballot, truths[image_path])
        score.pages.append(page_score)
        typer.echo(
            f"  {name}: capture {page_score.capture_rate:.0%}, "
            f"flag {page_score.flag_rate:.0%}, CER {page_score.cer:.1%}"
        )

    if not score.pages:
        raise _fail("no pages could be scored.")
    if out is not None:
        write_index(workspace, results)
        write_folder_index(out, results)

    typer.echo("")
    if out is not None:
        typer.echo(f"reports: {out / 'index.html'}")
        typer.echo("")
    if per_page:
        typer.echo(format_pages(score.pages))
        typer.echo("")
    typer.echo(format_report(score))

    if json_out is not None:
        payload = score_payload(score, dataset, [backend.model_id for backend in resolved])
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"\nscores: {json_out}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
