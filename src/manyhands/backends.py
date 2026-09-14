"""Backend adapters: run one local vision-OCR model and normalise what it returns.

churro-ocr owns model loading, prompting and the per-model postprocessors, so nothing
here touches transformers. What it does own is the mess that arrives afterwards: the
five default models return the same page as markdown with tables, as a bare block of
text, as HTML-ish layout markup, or as XML lines, and each has its own idea of where a
line ends. :func:`to_lines` flattens all of that to display lines so the aligner sees
one grammar.

Diversity in the ensemble comes only from using different models. Of the five defaults,
four resolve to ``do_sample: False`` and olmOCR-2 to ``temperature: 0.1``, so resampling
one model gives either the same text twice or a near copy of it. There is deliberately
no temperature path here.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from manyhands.text import Token, split_lines, tokenize

#: Default ensemble. Permissively licensed, all runnable locally through churro-ocr.
DEFAULT_MODELS: tuple[str, ...] = (
    "allenai/olmOCR-2-7B-1025",
    "PaddlePaddle/PaddleOCR-VL-1.5",
    "kristaller486/dots.ocr-1.5",
    "opendatalab/MinerU2.5-2509-1.2B",
    "nanonets/Nanonets-OCR2-3B",
)

#: Short names accepted on the command line, in the same order as DEFAULT_MODELS.
ALIASES: dict[str, str] = {
    "olmocr": "allenai/olmOCR-2-7B-1025",
    "olmocr-fp8": "allenai/olmOCR-2-7B-1025-FP8",
    "paddle": "PaddlePaddle/PaddleOCR-VL-1.5",
    "dots": "kristaller486/dots.ocr-1.5",
    "dots-mocr": "rednote-hilab/dots.mocr",
    "mineru": "opendatalab/MinerU2.5-2509-1.2B",
    "nanonets": "nanonets/Nanonets-OCR2-3B",
    "chandra": "datalab-to/chandra-ocr-2",
    "deepseek": "deepseek-ai/DeepSeek-OCR-2",
    "glm": "zai-org/GLM-OCR",
    "infinity": "infly/Infinity-Parser-7B",
    "lfm": "LiquidAI/LFM2.5-VL-1.6B",
    "churro": "stanford-oval/churro-3B",
}

#: Seconds one page may generate for before the model is stopped and whatever it has
#: produced so far is kept. A document VLM that falls into a repetition loop will
#: otherwise run to its token ceiling, which on a 200-page folder is the difference
#: between an overnight run and an unfinished one.
PAGE_TIMEOUT = 600.0

#: Consecutive page failures after which a model is dropped for the rest of the run.
#: dots.ocr-1.5 runs out of VRAM on every archive scan on a 24 GB card, and each attempt
#: can cost ten minutes before it does.
GIVE_UP_AFTER = 3

#: How much of an exception message reaches the report footer.
ERROR_MESSAGE_LIMIT = 200

#: Models whose weights are not under a permissive licence. Opt-in, with a notice.
RESTRICTED_LICENCES: dict[str, str] = {
    "stanford-oval/churro-3B": "Qwen-research",
}

_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_MARKDOWN_QUOTE = re.compile(r"^\s{0,3}>\s?")
_MARKDOWN_BULLET = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,3}[.)])\s+")
_MARKDOWN_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|`)")
_HTML_TAG = re.compile(r"<[^>\n]{1,120}>")
_XML_LINE = re.compile(r"<\s*(?:Line|TextLine|line)\b[^>]*>(.*?)<\s*/\s*(?:Line|TextLine|line)\s*>", re.S)
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_MD = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LATEX_DELIM = re.compile(r"\$\$?")


@dataclass(slots=True)
class Transcript:
    """One backend's reading of one page.

    :param backend: Display name of the backend, normally the model id.
    :param raw: Exactly what the backend returned, before normalisation.
    :param lines: Normalised display lines.
    :param tokens: Flat token stream built from ``lines``.
    :param seconds: Wall clock spent on this page.
    :param error: Failure message when the backend did not produce usable output.
    :param notes: Quality warnings raised while normalising, shown in the report footer.
    """

    backend: str
    raw: str = ""
    lines: list[str] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when this transcript can take part in the vote."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form used for the transcript cache."""
        return {
            "backend": self.backend,
            "raw": self.raw,
            "lines": self.lines,
            "seconds": round(self.seconds, 3),
            "error": self.error,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Transcript:
        """Rebuild a transcript from its cached form."""
        lines = [str(line) for line in payload.get("lines") or []]
        return cls(
            backend=str(payload["backend"]),
            raw=str(payload.get("raw") or ""),
            lines=lines,
            tokens=tokenize(lines),
            seconds=float(payload.get("seconds") or 0.0),
            error=payload.get("error"),
            notes=[str(note) for note in payload.get("notes") or []],
        )

    @classmethod
    def failed(cls, backend: str, message: str, seconds: float = 0.0) -> Transcript:
        """Build the transcript recorded when a backend crashed or timed out."""
        return cls(backend=backend, seconds=seconds, error=message)


def resolve_model(name: str) -> str:
    """Return the Hugging Face model id for a short alias or a full id."""
    cleaned = name.strip()
    if cleaned.startswith("hf:"):
        cleaned = cleaned[3:]
    return ALIASES.get(cleaned.casefold(), cleaned)


def licence_notice(model_id: str) -> str | None:
    """Return the notice to print before running a model with a restricted licence."""
    licence = RESTRICTED_LICENCES.get(model_id)
    if licence is None:
        return None
    return (
        f"{model_id} is opt-in: its weights are released under the {licence} licence, "
        "not a permissive open-source licence. Check the terms before using its output."
    )


def to_lines(raw: str) -> tuple[list[str], list[str]]:
    """Normalise one backend's raw output to display lines.

    :param raw: Text as churro-ocr's postprocessor handed it over.
    :returns: The display lines, and any notes worth showing in the report footer.
    """
    notes: list[str] = []
    text = raw.strip()
    if not text:
        return [], notes

    xml_lines = [match.group(1) for match in _XML_LINE.finditer(text)]
    if xml_lines:
        notes.append("output was XML line markup; line elements used as display lines")
        return [clean for clean in (_clean_inline(line) for line in xml_lines) if clean], notes

    text = _strip_front_matter(text)
    lines: list[str] = []
    in_table = False
    for line in split_lines(text):
        if _MARKDOWN_RULE.match(line) or _TABLE_DIVIDER.match(line):
            in_table = in_table or bool(_TABLE_DIVIDER.match(line))
            continue
        if line.startswith("|") and line.count("|") >= 2:
            in_table = True
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cleaned = _clean_inline(" ".join(cell for cell in cells if cell))
            if cleaned:
                lines.append(cleaned)
            continue
        stripped = _MARKDOWN_HEADING.sub("", line)
        stripped = _MARKDOWN_QUOTE.sub("", stripped)
        stripped = _MARKDOWN_BULLET.sub("", stripped)
        cleaned = _clean_inline(stripped)
        if cleaned:
            lines.append(cleaned)
    if in_table:
        notes.append("output contained a markdown table; cells were flattened into lines")
    return lines, notes


def _strip_front_matter(text: str) -> str:
    """Drop the YAML front matter olmOCR-style outputs put above the page text."""
    if not text.startswith("---"):
        return text
    parts = text.split("\n")
    for index in range(1, len(parts)):
        if parts[index].strip() in {"---", "..."}:
            return "\n".join(parts[index + 1 :]).strip()
    return text


def _clean_inline(line: str) -> str:
    """Strip inline markup that carries no reading, leaving the words untouched."""
    cleaned = _IMAGE_MD.sub(" ", line)
    cleaned = _LINK_MD.sub(r"\1", cleaned)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = _EMPHASIS.sub("", cleaned)
    cleaned = _LATEX_DELIM.sub("", cleaned)
    return " ".join(cleaned.split())


def build_transcript(backend: str, raw: str, seconds: float = 0.0) -> Transcript:
    """Normalise raw backend output into a transcript ready for alignment."""
    lines, notes = to_lines(raw)
    return Transcript(
        backend=backend,
        raw=raw,
        lines=lines,
        tokens=tokenize(lines),
        seconds=seconds,
        notes=notes,
    )


def repetition_ratio(tokens: list[Token], window: int = 6) -> float:
    """Return the share of the stream covered by an immediately repeated n-gram.

    Historical-document VLMs fail by looping: the model latches onto a phrase and emits
    it until it runs out of tokens. A high ratio means the transcript is a loop, not a
    reading.
    """
    if len(tokens) < window * 2:
        return 0.0
    keys = [token.key for token in tokens]
    repeated = 0
    index = 0
    while index + window * 2 <= len(keys):
        if keys[index : index + window] == keys[index + window : index + window * 2]:
            repeated += window
            index += window
        else:
            index += 1
    return repeated / len(keys)


class _Deadline:
    """A transformers stopping criterion that remembers whether it fired.

    ``MaxTimeCriteria`` reports nothing back, so without this a page that simply ran long
    looks the same as one the deadline cut short. They are not the same: several churro
    profiles reach ``generate`` by a path that never consults ``stopping_criteria``.
    """

    def __init__(self, seconds: float) -> None:
        from transformers import MaxTimeCriteria

        self._criteria = MaxTimeCriteria(seconds)
        self.fired = False

    def restart(self) -> None:
        """Start the clock again for a new page."""
        self._criteria.initial_timestamp = time.time()
        self.fired = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        done = self._criteria(input_ids, scores, **kwargs)
        if bool(done.all()) if hasattr(done, "all") else bool(done):
            self.fired = True
        return done


class OCRRunner:
    """Runs one model over page images through churro-ocr's Hugging Face backend.

    The model is loaded once and reused for every page, because loading a 7B VLM costs
    far more than transcribing a page with it.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        timeout: float | None = PAGE_TIMEOUT,
    ) -> None:
        """Create a runner for one model.

        :param model_id: Hugging Face model id passed to churro-ocr.
        :param device_map: ``device_map`` forwarded to transformers.
        :param torch_dtype: ``torch_dtype`` forwarded to transformers.
        :param timeout: Seconds one page may generate for, or ``None`` for no limit.
        """
        self.model_id = model_id
        self.timeout = timeout
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._client: Any | None = None
        self._deadline: Any | None = None
        self._load_seconds = 0.0
        self._loaded_page = False
        self._skip_reason: str | None = None
        self._failures = 0

    def _generation_kwargs(self) -> dict[str, Any]:
        """Return the generation overrides, holding on to the per-page deadline.

        Merged into whatever the model's churro profile already pins, so the profiles
        that fix ``do_sample`` keep fixing it.
        """
        if self.timeout is None:
            return {}
        from transformers import StoppingCriteriaList

        self._deadline = _Deadline(self.timeout)
        return {"stopping_criteria": StoppingCriteriaList([self._deadline])}

    def _load(self) -> Any:
        if self._client is not None:
            return self._client
        started = time.perf_counter()
        try:
            from churro_ocr.ocr import OCRClient
            from churro_ocr.providers import (
                HuggingFaceOptions,
                OCRBackendSpec,
                build_ocr_backend,
            )

            # Inside the guard because this reaches for transformers, which is half of what
            # the extra installs and is missing exactly when churro's own imports are not.
            generation_kwargs = self._generation_kwargs()
        except ImportError as exc:
            message = (
                "Local model inference needs churro-ocr's Hugging Face extra. "
                "Install it with: pip install 'manyhands[hf]' (and a torch build for your GPU)."
            )
            raise BackendUnavailable(message) from exc

        from churro_ocr.providers import hf as hf_presets

        repair_preset_super(hf_presets)

        backend = build_ocr_backend(
            OCRBackendSpec(
                provider="hf",
                model=self.model_id,
                options=HuggingFaceOptions(
                    model_kwargs={"device_map": self._device_map, "torch_dtype": self._torch_dtype},
                    generation_kwargs=generation_kwargs,
                ),
            )
        )
        self._client = OCRClient(backend)
        self._load_seconds = time.perf_counter() - started
        return self._client

    def transcribe(self, image_path: Path) -> Transcript:
        """Transcribe one page image, returning a failed transcript instead of raising."""
        started = time.perf_counter()
        if self._skip_reason is not None:
            return Transcript.failed(self.model_id, self._skip_reason)
        try:
            from churro_ocr.page_detection import DocumentPage

            client = self._load()
        except BackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unloadable model must not take the run down
            return self._give_up(_short_error(exc), started, after=1)
        if self._deadline is not None:
            self._deadline.restart()
        try:
            page = client.ocr(DocumentPage.from_image_path(image_path))
            raw = page.text or ""
        except Exception as exc:  # noqa: BLE001 - a page must never take the run down
            return self._give_up(_short_error(exc), started, after=GIVE_UP_AFTER)
        self._failures = 0
        elapsed = time.perf_counter() - started
        transcript = build_transcript(self.model_id, raw, elapsed)
        transcript.notes.extend(self._timing_notes(elapsed))
        return transcript

    def _give_up(self, message: str, started: float, *, after: int) -> Transcript:
        """Record a failure and stop calling this model once it has failed ``after`` pages.

        A model that will not download, or that runs out of VRAM on every scan in a folder,
        does the same on page two. Retrying it per page costs hours and buys no votes.
        """
        self._failures += 1
        if self._failures >= after:
            self._skip_reason = f"{message} (not retried on later pages)"
            message = self._skip_reason
        return Transcript.failed(self.model_id, message, time.perf_counter() - started)

    def _timing_notes(self, elapsed: float) -> list[str]:
        """Describe a page's wall clock, once, for the report footer."""
        notes = []
        generating = elapsed - self._load_seconds
        if self._deadline is not None and self._deadline.fired:
            notes.append(
                f"stopped at the {self.timeout:.0f}s page timeout; the reading may be cut short"
            )
        elif self.timeout is not None and generating >= self.timeout:
            # Some churro profiles reach generate() by a path that never consults
            # stopping_criteria, so the deadline is set and simply not honoured.
            notes.append(
                f"generated for {generating:.0f}s, past the {self.timeout:.0f}s page timeout: "
                "this backend does not stop when asked"
            )
        if not self._loaded_page:
            # churro-ocr defers most of the weight load to the first ocr() call, so the
            # honest thing to say is which page paid for it, not how many seconds it cost.
            notes.append("first page for this model, so it includes the one-off weight load")
            self._loaded_page = True
        self._load_seconds = 0.0
        return notes

    def release(self) -> None:
        """Drop the loaded model and free its VRAM.

        The default ensemble does not fit in 24 GB all at once, so the pipeline runs one
        model over the whole folder and calls this before loading the next.
        """
        if self._client is None:
            return
        self._client = None
        self._deadline = None
        self._load_seconds = 0.0
        self._loaded_page = False
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - needs a GPU
            torch.cuda.empty_cache()


#: churro-ocr 0.3.0 preset backends whose overrides call zero-argument ``super()``.
_STALE_SUPER_PRESETS = (
    "ChandraOCR2OCRBackend",
    "DotsOCR15OCRBackend",
    "PaddleOCRVL15OCRBackend",
    "LFM25VLOCRBackend",
)


def repair_preset_super(module: Any) -> list[str]:
    """Re-point the stale ``__class__`` cell in churro-ocr's slotted preset backends.

    ``@dataclass(slots=True)`` builds a replacement class object, so the ``__class__``
    cell captured by the original class body still names the discarded class and any
    zero-argument ``super()`` inside raises TypeError. Four churro-ocr 0.3.0 presets are
    affected, two of them in the default ensemble, so without this dots.ocr-1.5 and
    PaddleOCR-VL-1.5 fail on every page. CPython 3.13 repoints the cell itself, so on 3.13
    this finds nothing and returns an empty list. Reported upstream; see docs/churro-issue.md.

    :param module: ``churro_ocr.providers.hf``.
    :returns: The names of the classes that needed repairing.
    """
    repaired: list[str] = []
    for name in _STALE_SUPER_PRESETS:
        target = getattr(module, name, None)
        if not isinstance(target, type):
            continue
        for attribute in vars(target).values():
            code = getattr(attribute, "__code__", None)
            if code is None or "__class__" not in code.co_freevars:
                continue
            cell = attribute.__closure__[code.co_freevars.index("__class__")]
            if cell.cell_contents is not target:
                cell.cell_contents = target
                repaired.append(name)
                break
    return repaired


class BackendUnavailable(RuntimeError):
    """Raised when local inference is not installed at all, which no page can recover from."""


def _short_error(exc: BaseException) -> str:
    """Return one line naming the exception, short enough for a report footer.

    A CUDA out-of-memory message runs to several hundred characters of allocator
    statistics, and the part that tells you what happened is the first sentence.
    """
    text = str(exc).strip() or exc.__class__.__name__
    first = text.splitlines()[0]
    if len(first) > ERROR_MESSAGE_LIMIT:
        sentence = first.rfind(". ", 0, ERROR_MESSAGE_LIMIT)
        first = first[: sentence + 1] if sentence > 0 else first[:ERROR_MESSAGE_LIMIT].rstrip() + "..."
    return f"{exc.__class__.__name__}: {first}"


class CachedBackend:
    """Serves transcripts that a previous run stored, instead of loading a model.

    This is how ``manyhands run`` re-votes a folder after a glossary change without
    spending another GPU hour, and how the test fixtures drive the whole pipeline.
    """

    def __init__(self, name: str, directory: Path) -> None:
        """Create a cached backend.

        :param name: Backend name as it appears in the report.
        :param directory: Folder holding ``<page-stem>.json`` transcripts.
        """
        self.model_id = name
        self.directory = directory

    def transcribe(self, image_path: Path) -> Transcript:
        """Return the stored transcript for a page, or a failed one if it is not usable."""
        transcript = read_cached(self.directory, image_path.stem)
        if transcript is None:
            path = self.directory / f"{image_path.stem}.json"
            return Transcript.failed(self.model_id, f"no usable cached transcript at {path}")
        transcript.backend = self.model_id
        return transcript

    def release(self) -> None:
        """Nothing to free; present so backends are interchangeable."""


BackendLike = OCRRunner | CachedBackend


def cache_dir(workspace: Path, backend: str) -> Path:
    """Return the transcript cache folder for one backend inside a workspace."""
    return workspace / "transcripts" / _safe_name(backend)


def read_cached(directory: Path, stem: str) -> Transcript | None:
    """Return the transcript stored for one page, or None when there is nothing usable.

    A cache file that will not parse is treated as absent rather than fatal, so a run
    interrupted mid-write costs one page rather than the folder.
    """
    path = directory / f"{stem}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Transcript.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _safe_name(backend: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", backend)


def resolve_backends(
    names: Iterable[str],
    workspace: Path,
    *,
    allow_cached: bool = True,
    timeout: float | None = PAGE_TIMEOUT,
) -> tuple[list[BackendLike], list[str]]:
    """Turn command-line backend names into runnable backends.

    A name resolves to a model when churro-ocr knows it or it looks like a Hugging Face
    id; otherwise it resolves to a cached backend if that cache exists. Anything else is
    an error, reported to the caller rather than raised.

    A model id always resolves to the model, never to its cache. Skipping pages that are
    already transcribed is the pipeline's job, so a half-finished cache resumes instead
    of pinning the backend to the pages it happens to hold.

    Two names for one model resolve once. ``--backends olmocr,allenai/olmOCR-2-7B-1025``
    is one model, and letting it vote twice would report its reading as corroborated when
    nothing corroborated it.

    :param names: Backend names or aliases as typed by the user.
    :param workspace: The ``.manyhands`` workspace holding the transcript cache.
    :param allow_cached: Set False to refuse cached transcripts and force real inference.
    :param timeout: Seconds a model may spend generating one page, or ``None`` for no limit.
    :returns: The resolved backends, and a message for every name that did not resolve.
    """
    resolved: list[BackendLike] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for name in names:
        cleaned = name.strip()
        if not cleaned:
            continue
        model_id = resolve_model(cleaned)
        first = seen.get(model_id)
        if first is not None:
            problems.append(
                f"{cleaned!r} is {model_id}, which {first!r} already covers; "
                "it votes once. A model cannot corroborate itself."
            )
            continue
        if "/" in model_id:
            seen[model_id] = cleaned
            resolved.append(OCRRunner(model_id, timeout=timeout))
            continue
        cached = cache_dir(workspace, cleaned)
        if allow_cached and cached.is_dir() and any(cached.glob("*.json")):
            seen[model_id] = cleaned
            resolved.append(CachedBackend(cleaned, cached))
            continue
        problems.append(
            f"unknown backend {cleaned!r}: use a Hugging Face model id such as "
            f"'{DEFAULT_MODELS[0]}', one of the aliases ({', '.join(sorted(ALIASES))}), "
            f"or the name of a cached transcript folder under {workspace / 'transcripts'}"
        )
    return resolved, problems
