"""Normalising what each backend's output grammar actually looks like."""

from __future__ import annotations

import json
import sys

import pytest

from conftest import PAGES
from manyhands.backends import (
    ALIASES,
    DEFAULT_MODELS,
    GIVE_UP_AFTER,
    CachedBackend,
    OCRRunner,
    Transcript,
    _Deadline,
    _short_error,
    build_transcript,
    cache_dir,
    licence_notice,
    repair_preset_super,
    repetition_ratio,
    resolve_backends,
    resolve_model,
    to_lines,
)
from manyhands.text import tokenize

OLMOCR_OUTPUT = """---
primary_language: fr
is_rotation_valid: true
rotation_correction: 0
is_table: false
---
# Journal de Celestine

Samedi 25 Juillet 1914

Des bruits de guerre, beaucoup plus precis.
"""

NANONETS_OUTPUT = """<watermark>ARCHIVES</watermark>

| Date | Entry |
| --- | --- |
| 25 Juillet | Des bruits de guerre |
| 26 Juillet | La mobilisation |

**Samedi** 25 _Juillet_ 1914

![image](data:image/png;base64,AAAA)
"""

CHURRO_OUTPUT = """<HistoricalDocument>
  <Line>Samedi 25 Juillet 1914</Line>
  <Line>Des bruits de guerre, beaucoup plus precis</Line>
</HistoricalDocument>"""

PADDLE_OUTPUT = "Samedi 25 Juillet 1914\nDes bruits de guerre, beaucoup plus precis"


def test_olmocr_front_matter_is_dropped_and_headings_are_kept_as_text():
    lines, notes = to_lines(OLMOCR_OUTPUT)

    assert lines == [
        "Journal de Celestine",
        "Samedi 25 Juillet 1914",
        "Des bruits de guerre, beaucoup plus precis.",
    ]
    assert notes == []


def test_markdown_tables_are_flattened_and_the_report_is_told():
    lines, notes = to_lines(NANONETS_OUTPUT)

    assert lines == [
        "ARCHIVES",
        "Date Entry",
        "25 Juillet Des bruits de guerre",
        "26 Juillet La mobilisation",
        "Samedi 25 Juillet 1914",
    ]
    assert notes == ["output contained a markdown table; cells were flattened into lines"]


def test_xml_line_markup_becomes_one_display_line_per_element():
    lines, notes = to_lines(CHURRO_OUTPUT)

    assert lines == ["Samedi 25 Juillet 1914", "Des bruits de guerre, beaucoup plus precis"]
    assert notes == ["output was XML line markup; line elements used as display lines"]


def test_bare_text_passes_through_untouched():
    lines, notes = to_lines(PADDLE_OUTPUT)

    assert lines == PADDLE_OUTPUT.split("\n")
    assert notes == []


def test_every_grammar_reduces_to_the_same_reading():
    from_xml, _ = to_lines(CHURRO_OUTPUT)
    from_text, _ = to_lines(PADDLE_OUTPUT)

    assert from_xml == from_text


def test_bullets_quotes_and_horizontal_rules_are_stripped():
    lines, _ = to_lines("- first item\n> a quote\n***\n1. numbered\n")
    assert lines == ["first item", "a quote", "numbered"]


def test_empty_output_yields_no_lines():
    assert to_lines("") == ([], [])
    assert to_lines("   \n\n  ") == ([], [])


def test_build_transcript_tokenizes_with_line_indices():
    transcript = build_transcript("model-a", PADDLE_OUTPUT, seconds=2.5)

    assert transcript.ok is True
    assert transcript.seconds == 2.5
    assert transcript.tokens[0].text == "Samedi"
    assert transcript.tokens[0].line == 0
    assert transcript.tokens[-1].line == 1


def test_a_transcript_round_trips_through_the_cache_format():
    original = build_transcript("model-a", NANONETS_OUTPUT)
    restored = Transcript.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.backend == original.backend
    assert restored.lines == original.lines
    assert [token.text for token in restored.tokens] == [token.text for token in original.tokens]


def test_a_failed_transcript_does_not_vote():
    failed = Transcript.failed("model-a", "CUDA out of memory")

    assert failed.ok is False
    assert failed.error == "CUDA out of memory"
    assert failed.tokens == []


def test_repetition_ratio_catches_a_hallucination_loop():
    looped = tokenize(["the parish register " * 20])
    clean = tokenize(["Des bruits de guerre beaucoup plus precis que tous ceux entendus jusqu ici"])

    assert repetition_ratio(looped) > 0.5
    assert repetition_ratio(clean) == 0.0
    assert repetition_ratio(tokenize(["short"])) == 0.0


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_every_alias_resolves_to_a_hugging_face_id(alias):
    assert "/" in resolve_model(alias)


def test_full_model_ids_and_the_hf_prefix_pass_through():
    assert resolve_model("allenai/olmOCR-2-7B-1025") == "allenai/olmOCR-2-7B-1025"
    assert resolve_model("hf:nanonets/Nanonets-OCR2-3B") == "nanonets/Nanonets-OCR2-3B"


def test_the_five_defaults_are_the_documented_ones():
    assert DEFAULT_MODELS == (
        "allenai/olmOCR-2-7B-1025",
        "PaddlePaddle/PaddleOCR-VL-1.5",
        "kristaller486/dots.ocr-1.5",
        "opendatalab/MinerU2.5-2509-1.2B",
        "nanonets/Nanonets-OCR2-3B",
    )
    assert all(licence_notice(model) is None for model in DEFAULT_MODELS)


def test_a_long_error_is_cut_at_a_sentence_for_the_report_footer():
    """A CUDA out-of-memory message is mostly allocator statistics."""
    oom = RuntimeError(
        "CUDA out of memory. Tried to allocate 27.76 GiB. GPU 0 has a total capacity of "
        "23.99 GiB of which 0 bytes is free. Of the allocated memory 36.29 GiB is allocated "
        "by PyTorch, and 205.16 MiB is reserved by PyTorch but unallocated."
    )

    message = _short_error(oom)

    assert message.startswith("RuntimeError: CUDA out of memory.")
    assert message.endswith("of which 0 bytes is free.")
    assert len(message) < 220


def test_an_unbroken_error_is_cut_with_an_ellipsis():
    message = _short_error(ValueError("x" * 400))

    assert message.endswith("...")
    assert len(message) < 220


def test_churro_carries_a_licence_notice():
    notice = licence_notice(resolve_model("churro"))

    assert notice is not None
    assert "Qwen-research" in notice


def test_a_cached_backend_serves_the_stored_transcript_under_the_requested_name(tmp_path):
    directory = cache_dir(tmp_path, "stub-a")
    directory.mkdir(parents=True)
    payload = build_transcript("some/other-model", PADDLE_OUTPUT).to_dict()
    (directory / "p001.json").write_text(json.dumps(payload), encoding="utf-8")

    transcript = CachedBackend("stub-a", directory).transcribe(tmp_path / "p001.jpg")

    assert transcript.ok is True
    assert transcript.backend == "stub-a"
    assert transcript.lines == PADDLE_OUTPUT.split("\n")


def test_a_missing_cache_entry_fails_that_page_rather_than_the_run(tmp_path):
    transcript = CachedBackend("stub-a", tmp_path).transcribe(tmp_path / "p001.jpg")

    assert transcript.ok is False
    assert "no usable cached transcript" in transcript.error


def test_a_corrupt_cache_entry_fails_that_page_rather_than_the_run(tmp_path):
    (tmp_path / "p001.json").write_text("{not json", encoding="utf-8")

    transcript = CachedBackend("stub-a", tmp_path).transcribe(tmp_path / "p001.jpg")

    assert transcript.ok is False
    assert "no usable cached transcript" in transcript.error


def test_a_model_id_resolves_to_the_model_even_when_it_has_a_half_filled_cache(tmp_path):
    """Pinning a model to its cache would make the pages it is missing unreachable."""
    directory = cache_dir(tmp_path, "allenai/olmOCR-2-7B-1025")
    directory.mkdir(parents=True)
    (directory / "p001.json").write_text(json.dumps({"backend": "x", "lines": ["a"]}), encoding="utf-8")

    resolved, problems = resolve_backends(["allenai/olmOCR-2-7B-1025"], tmp_path)

    assert problems == []
    assert isinstance(resolved[0], OCRRunner)


def test_a_cache_folder_name_resolves_to_its_transcripts_unless_the_cache_is_refused(tmp_path):
    directory = cache_dir(tmp_path, "stub-a")
    directory.mkdir(parents=True)
    (directory / "p001.json").write_text(json.dumps({"backend": "stub-a", "lines": ["a"]}), encoding="utf-8")

    cached, problems = resolve_backends(["stub-a"], tmp_path)
    refused, refusals = resolve_backends(["stub-a"], tmp_path, allow_cached=False)

    assert problems == []
    assert isinstance(cached[0], CachedBackend)
    assert refused == []
    assert "unknown backend 'stub-a'" in refusals[0]


def test_an_unknown_backend_is_reported_rather_than_raised(tmp_path):
    resolved, problems = resolve_backends(["stub-a", " ", "nanonets"], tmp_path)

    assert len(resolved) == 1
    assert len(problems) == 1
    assert "unknown backend 'stub-a'" in problems[0]
    assert str(tmp_path / "transcripts") in problems[0]


def test_repair_preset_super_is_needed_on_3_12_only_and_is_idempotent():
    """CPython 3.13 repoints the cell itself when the slotted dataclass is rebuilt."""
    hf = pytest.importorskip("churro_ocr.providers.hf")
    needed = sys.version_info < (3, 13)

    first = repair_preset_super(hf)
    second = repair_preset_super(hf)

    assert bool(first) is needed
    assert ("PaddleOCRVL15OCRBackend" in first) is needed
    assert second == []


def test_repair_preset_super_ignores_a_module_without_the_presets():
    assert repair_preset_super(object()) == []


def test_repair_preset_super_fixes_every_override_in_the_class():
    hf = pytest.importorskip("churro_ocr.providers.hf")

    repair_preset_super(hf)
    target = hf.PaddleOCRVL15OCRBackend
    cells = [
        attribute.__closure__[attribute.__code__.co_freevars.index("__class__")]
        for attribute in vars(target).values()
        if getattr(attribute, "__code__", None) and "__class__" in attribute.__code__.co_freevars
    ]

    assert len(cells) > 1
    assert all(cell.cell_contents is target for cell in cells)


class FakePage:
    text = "Buried this day John Fenwick"


class FakeClient:
    on_ocr = None

    def ocr(self, page):  # noqa: ANN001, ARG002 - stands in for churro's OCRClient
        if self.on_ocr is not None:
            self.on_ocr()
        return FakePage()


def loaded_runner(load_seconds: float) -> OCRRunner:
    """An OCRRunner with the model already in hand, as if a load had just cost that long."""
    runner = OCRRunner("fake/model")
    runner._client = FakeClient()
    runner._load_seconds = load_seconds
    return runner


def test_the_page_that_paid_for_the_model_load_says_so():
    runner = loaded_runner(93.4)
    image = PAGES / "journal-p001.jpg"

    first = runner.transcribe(image)
    second = runner.transcribe(image)

    assert first.notes == ["first page for this model, so it includes the one-off weight load"]
    assert second.notes == []
    assert first.lines == ["Buried this day John Fenwick"]


def test_releasing_a_runner_drops_the_model_and_is_safe_to_repeat():
    runner = loaded_runner(1.0)
    runner.release()
    runner.release()

    assert runner._client is None
    assert runner._load_seconds == 0.0
    assert runner._loaded_page is False


def test_a_backend_that_raises_mid_page_fails_only_that_page(tmp_path):
    runner = loaded_runner(0.0)
    runner._client = None
    runner._load = lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory\nsecond line"))

    transcript = runner.transcribe(tmp_path / "p001.jpg")

    assert transcript.ok is False
    assert transcript.error == "RuntimeError: CUDA out of memory (not retried on later pages)"
    assert transcript.tokens == []


def test_a_model_that_fails_three_pages_running_is_dropped_for_the_rest():
    runner = loaded_runner(0.0)
    attempts = []

    def out_of_memory():
        attempts.append(1)
        raise RuntimeError("CUDA out of memory")

    runner._client.on_ocr = out_of_memory
    failures = [runner.transcribe(PAGES / "journal-p001.jpg") for _ in range(5)]

    assert len(attempts) == GIVE_UP_AFTER
    assert all(not transcript.ok for transcript in failures)
    assert failures[0].error == "RuntimeError: CUDA out of memory"
    assert failures[-1].error == "RuntimeError: CUDA out of memory (not retried on later pages)"


def test_a_page_that_works_clears_the_earlier_failures():
    runner = loaded_runner(0.0)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) in (1, 2, 4, 5):
            raise RuntimeError("CUDA out of memory")

    runner._client.on_ocr = flaky
    results = [runner.transcribe(PAGES / "journal-p001.jpg") for _ in range(6)]

    assert [transcript.ok for transcript in results] == [False, False, True, False, False, True]


def test_a_model_that_will_not_load_fails_every_page_without_retrying():
    attempts = []

    class Unloadable(OCRRunner):
        def _load(self):
            attempts.append(1)
            raise OSError("We couldn't connect to huggingface.co")

    runner = Unloadable("fake/model")
    failures = [runner.transcribe(PAGES / name) for name in ("journal-p001.jpg", "journal-p002.jpg")]

    assert len(attempts) == 1
    assert all(not transcript.ok for transcript in failures)
    assert all("couldn't connect" in transcript.error for transcript in failures)


def test_a_page_the_deadline_cut_short_keeps_what_it_read_and_says_so():
    pytest.importorskip("transformers")
    runner = loaded_runner(0.0)
    runner.timeout = 30.0
    runner._deadline = _Deadline(30.0)
    # The criterion fires mid-generation, which is after transcribe() restarts the clock.
    runner._client.on_ocr = lambda: setattr(runner._deadline, "fired", True)

    transcript = runner.transcribe(PAGES / "journal-p001.jpg")

    assert transcript.ok
    assert transcript.lines == ["Buried this day John Fenwick"]
    assert transcript.notes == [
        "stopped at the 30s page timeout; the reading may be cut short",
        "first page for this model, so it includes the one-off weight load",
    ]


def test_a_backend_that_ran_past_the_timeout_without_stopping_is_named():
    runner = loaded_runner(0.0)
    runner.timeout = 0.0

    transcript = runner.transcribe(PAGES / "journal-p001.jpg")

    assert transcript.notes == [
        "generated for 0s, past the 0s page timeout: this backend does not stop when asked",
        "first page for this model, so it includes the one-off weight load",
    ]


def test_the_deadline_records_the_moment_it_fires():
    torch = pytest.importorskip("torch")
    # A negative budget is past before the first token, so the clock granularity that makes
    # a zero-second budget flaky on Windows cannot reach this.
    deadline = _Deadline(-1.0)
    deadline.restart()

    assert deadline.fired is False
    assert bool(deadline(torch.zeros((1, 4), dtype=torch.long), None).all()) is True
    assert deadline.fired is True


def test_the_deadline_is_a_transformers_stopping_criterion_restarted_per_page():
    pytest.importorskip("transformers")
    runner = OCRRunner("fake/model", timeout=30.0)

    kwargs = runner._generation_kwargs()
    first = runner._deadline._criteria.initial_timestamp
    runner._client = FakeClient()
    runner.transcribe(PAGES / "journal-p001.jpg")

    assert list(kwargs) == ["stopping_criteria"]
    assert kwargs["stopping_criteria"][0] is runner._deadline
    assert runner._deadline._criteria.max_time == 30.0
    assert runner._deadline._criteria.initial_timestamp >= first


def test_no_timeout_leaves_generation_alone():
    assert OCRRunner("fake/model", timeout=None)._generation_kwargs() == {}


def test_an_alias_and_its_full_id_resolve_to_one_voter(tmp_path):
    """Two names for one model used to load it twice and let it corroborate itself."""
    resolved, problems = resolve_backends(
        ["olmocr", "allenai/olmOCR-2-7B-1025", "nanonets"], tmp_path
    )

    assert [backend.model_id for backend in resolved] == [
        "allenai/olmOCR-2-7B-1025",
        "nanonets/Nanonets-OCR2-3B",
    ]
    assert len(problems) == 1
    assert "'allenai/olmOCR-2-7B-1025' is allenai/olmOCR-2-7B-1025" in problems[0]
    assert "'olmocr' already covers" in problems[0]


def test_the_same_cached_backend_named_twice_resolves_once(tmp_path):
    directory = cache_dir(tmp_path, "stub-a")
    directory.mkdir(parents=True)
    (directory / "p001.json").write_text('{"backend": "stub-a", "lines": ["a"]}', encoding="utf-8")

    resolved, problems = resolve_backends(["stub-a", "stub-a"], tmp_path)

    assert [backend.model_id for backend in resolved] == ["stub-a"]
    assert len(problems) == 1

