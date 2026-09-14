"""The whole path, end to end, on the two committed sample pages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from conftest import PAGES, transcripts
from manyhands.backends import Transcript, resolve_backends
from manyhands.pipeline import (
    BLANK_TOKENS,
    MIN_VOTERS,
    PipelineError,
    agreement_payload,
    cache_transcript,
    discover_pages,
    rasterise_pdf,
    run_folder,
    run_payload,
    transcribe_pages,
    vote_page,
    write_page,
)

STUBS = ["stub-a", "stub-b", "stub-c"]
ARTIFACTS = ["agreement.json", "consensus.txt", "report.html"]


@pytest.fixture
def folder(tmp_path):
    """A writable copy of the committed sample pages, cached transcripts included."""
    target = tmp_path / "pages"
    shutil.copytree(PAGES, target)
    return target


def contested(results):
    """The first page the backends actually disagreed on."""
    return next(result for result in results if result.flagged)


@pytest.fixture
def run(folder):
    """The full pipeline over the sample pages, driven by the cached stub transcripts."""
    workspace = folder / ".manyhands"
    backends, problems = resolve_backends(STUBS, workspace)
    assert problems == []
    return folder, run_folder(folder, backends, folder / "manyhands-out", workspace)


def test_the_committed_fixtures_cover_every_stub_and_page():
    cache = PAGES / ".manyhands" / "transcripts"
    pages = sorted(path.stem for path in PAGES.glob("*.jpg"))

    assert len(pages) == 2
    for stub in STUBS:
        assert sorted(path.stem for path in (cache / stub).glob("*.json")) == pages


def test_every_page_gets_the_three_artifacts(run):
    folder, results = run

    assert len(results) == 2
    for result in results:
        for name in ARTIFACTS:
            path = folder / "manyhands-out" / result.stem / name
            assert path.is_file()
            assert path.stat().st_size > 0
    assert (folder / "manyhands-out" / "index.html").is_file()


def test_the_run_finds_real_disagreements_on_real_pages(run):
    _, results = run
    total = sum(len(result.ballot.slots) for result in results)
    flagged = sum(result.flagged for result in results)

    assert total > 200
    assert 0 < flagged < total
    assert all(result.ballot.backends == STUBS for result in results)


def test_agreement_json_has_the_documented_shape(run):
    folder, results = run
    page = contested(results)
    payload = json.loads(
        (folder / "manyhands-out" / page.stem / "agreement.json").read_text(encoding="utf-8")
    )

    assert payload["page"] == page.name
    assert payload["backends"] == STUBS
    assert set(payload["timings"]) == set(STUBS)
    slot = next(entry for entry in payload["slots"] if entry["agreement"] < 1.0)
    assert set(slot) == {
        "i",
        "consensus",
        "agreement",
        "votes",
        "bbox",
        "line",
        "contested",
        "confirmed",
        "readings",
    }
    assert sum(slot["votes"].values()) == len(STUBS)
    assert len(slot["bbox"]) == 4


def test_the_consensus_text_is_what_the_report_shows(run):
    folder, results = run
    result = results[0]
    consensus = (folder / "manyhands-out" / result.stem / "consensus.txt").read_text(encoding="utf-8")
    words = consensus.split()

    assert words
    assert all(word in consensus for word in words[:20])
    assert consensus.endswith("\n")


def test_the_workspace_index_points_at_the_reports(run):
    folder, results = run
    index = json.loads((folder / ".manyhands" / "index.json").read_text(encoding="utf-8"))

    assert set(index) == {result.stem for result in results}
    for result in results:
        assert index[result.stem]["page"] == result.name
        assert (folder / "manyhands-out" / result.stem).samefile(index[result.stem]["out_dir"])


def test_the_run_summary_totals_match_the_pages_it_lists(run):
    folder, results = run
    payload = run_payload(results, folder / "manyhands-out")

    assert payload["slots"] == sum(len(result.ballot.slots) for result in results)
    assert payload["flagged"] == sum(result.flagged for result in results)
    assert payload["flag_rate"] == round(payload["flagged"] / payload["slots"], 4)


def test_a_run_that_found_no_pages_summarises_as_zero_rather_than_dividing_by_it():
    payload = run_payload([], Path("out"))

    assert payload["pages"] == []
    assert payload["flag_rate"] == 0.0


def test_a_rerun_from_the_cache_reproduces_the_same_consensus(run):
    folder, results = run
    first = (folder / "manyhands-out" / results[0].stem / "consensus.txt").read_text(encoding="utf-8")

    backends, _ = resolve_backends(STUBS, folder / ".manyhands")
    again = run_folder(folder, backends, folder / "second", folder / ".manyhands")
    second = (folder / "second" / results[0].stem / "consensus.txt").read_text(encoding="utf-8")

    assert first == second
    assert [len(r.ballot.slots) for r in results] == [len(r.ballot.slots) for r in again]


def test_a_glossary_confirmation_changes_the_next_run(run):
    folder, results = run
    payload = json.loads(
        (folder / "manyhands-out" / contested(results).stem / "agreement.json").read_text(encoding="utf-8")
    )
    slot = next(entry for entry in payload["slots"] if entry["agreement"] < 1.0 and entry["consensus"])
    key = "|".join(sorted({reading.casefold().strip(".,;:'\"") for reading in slot["votes"] if reading}))

    backends, _ = resolve_backends(STUBS, folder / ".manyhands")
    again = run_folder(
        folder,
        backends,
        folder / "second",
        folder / ".manyhands",
        glossary={key: "MANYHANDS-CONFIRMED"},
    )
    changed = [
        slot for result in again for slot in result.ballot.slots if slot.consensus == "MANYHANDS-CONFIRMED"
    ]

    assert changed
    assert all(slot.confirmed for slot in changed)


def test_transcribe_pages_runs_one_backend_at_a_time_and_releases_it(folder):
    order: list[str] = []

    class Recorder:
        def __init__(self, name):
            self.model_id = name

        def transcribe(self, image_path):
            order.append(f"{self.model_id}:{image_path.stem}")
            return Transcript(backend=self.model_id, lines=["alpha beta"])

        def release(self):
            order.append(f"{self.model_id}:release")

    pages = discover_pages(folder, folder / ".manyhands")
    transcribe_pages(pages, [Recorder("one"), Recorder("two")], folder / ".manyhands")

    assert order == [
        "one:journal-p001",
        "one:journal-p002",
        "one:release",
        "two:journal-p001",
        "two:journal-p002",
        "two:release",
    ]


def test_a_transcript_is_cached_as_soon_as_it_lands(folder, tmp_path):
    workspace = tmp_path / "ws"
    pages = discover_pages(folder, workspace)

    class One:
        model_id = "just-one"

        def transcribe(self, image_path):
            return Transcript(backend=self.model_id, lines=["alpha"])

        def release(self):
            pass

    transcribe_pages(pages[:1], [One()], workspace)
    stored = json.loads((workspace / "transcripts" / "just-one" / "journal-p001.json").read_text("utf-8"))

    assert stored["lines"] == ["alpha"]


def test_a_crashed_backend_does_not_take_the_page_down():
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        [
            *transcripts(("model-a", "alpha beta gamma"), ("model-b", "alpha beta gamma")),
            Transcript.failed("model-c", "CUDA out of memory"),
        ],
    )

    assert result.ballot.backends == ["model-a", "model-b"]
    assert result.dropped == {"model-c": "CUDA out of memory"}
    assert [slot.consensus for slot in result.ballot.slots] == ["alpha", "beta", "gamma"]


def test_a_backend_with_more_than_three_times_the_median_is_dropped():
    short = "alpha beta gamma delta"
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        transcripts(
            ("model-a", short),
            ("model-b", short),
            ("model-c", short),
            ("model-d", " ".join(["word"] * 13)),
        ),
    )

    assert "model-d" in result.dropped
    assert "13 tokens against a page median of 4" in result.dropped["model-d"]
    assert result.ballot.backends == ["model-a", "model-b", "model-c"]


def test_a_looping_backend_is_dropped_as_a_hallucination():
    real = "Samedi 25 Juillet 1914 des bruits de guerre beaucoup plus precis que tous"
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        transcripts(
            ("model-a", real),
            ("model-b", real),
            ("model-c", real),
            ("model-d", "de la guerre de la guerre de la guerre de la guerre de la guerre de la guerre"),
        ),
    )

    assert "model-d" in result.dropped
    assert "hallucination loop" in result.dropped["model-d"]


def test_a_backend_that_stopped_a_third_of_the_way_down_is_dropped():
    full = " ".join(f"word{index}" for index in range(30))
    thin = " ".join(f"word{index}" for index in range(8))
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        transcripts(("model-a", full), ("model-b", full), ("model-c", full), ("model-d", thin)),
    )

    assert "8 tokens against a page median of 30" in result.dropped["model-d"]
    assert result.ballot.backends == ["model-a", "model-b", "model-c"]


def test_a_backend_that_returned_nothing_does_not_dissent_from_every_word():
    real = "Samedi 25 Juillet 1914"
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        transcripts(("model-a", real), ("model-b", real), ("model-c", "")),
    )

    assert result.dropped["model-c"] == "dropped from the vote: returned no text"
    assert [slot.agreement for slot in result.ballot.slots] == [1.0] * 4


def test_outlier_rules_are_suspended_when_too_few_backends_are_left():
    result = vote_page(
        "p.jpg",
        PAGES / "journal-p001.jpg",
        transcripts(("model-a", "alpha beta gamma"), ("model-b", " ".join(["word"] * 40))),
    )

    assert len(result.ballot.backends) < MIN_VOTERS
    assert result.dropped == {}


def test_a_blank_page_produces_an_empty_consensus_and_a_blank_flag(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (600, 400), "white").save(blank)
    result = vote_page("blank.png", blank, transcripts(("model-a", ""), ("model-b", "")))

    assert result.blank is True
    assert result.ballot.slots == []
    assert agreement_payload(result)["blank"] is True
    assert all(len(t.tokens) <= BLANK_TOKENS for t in result.transcripts)

    out = write_page(result, tmp_path / "out")
    assert (out / "consensus.txt").read_text(encoding="utf-8").strip() == ""


def test_discover_pages_explains_an_empty_folder(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing to read", encoding="utf-8")

    with pytest.raises(PipelineError) as caught:
        discover_pages(tmp_path, tmp_path / ".manyhands")

    assert "holds no pages" in str(caught.value)
    assert ".pdf" in str(caught.value)


def test_discover_pages_explains_a_missing_folder(tmp_path):
    with pytest.raises(PipelineError) as caught:
        discover_pages(tmp_path / "nope", tmp_path / ".manyhands")

    assert "no such folder" in str(caught.value)


def test_discover_pages_skips_the_workspace_and_hidden_files(folder):
    pages = discover_pages(folder, folder / ".manyhands")

    assert [name for name, _ in pages] == ["journal-p001.jpg", "journal-p002.jpg"]


def test_a_pdf_is_rasterised_once_and_reused(tmp_path):
    pypdfium2 = pytest.importorskip("pypdfium2")
    source = tmp_path / "book.pdf"
    document = pypdfium2.PdfDocument.new()
    for _ in range(2):
        document.new_page(300, 400)
    document.save(source)
    document.close()

    pages = rasterise_pdf(source, tmp_path / "rendered", dpi=72)
    stamps = [path.stat().st_mtime_ns for _, path in pages]
    again = rasterise_pdf(source, tmp_path / "rendered", dpi=72)

    assert [name for name, _ in pages] == ["book.pdf#1", "book.pdf#2"]
    assert [path.stat().st_mtime_ns for _, path in again] == stamps


def test_a_broken_pdf_is_reported_not_raised_raw(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")

    with pytest.raises(PipelineError) as caught:
        rasterise_pdf(broken, tmp_path / "rendered")

    assert "cannot read broken.pdf" in str(caught.value)


def test_a_page_that_is_not_really_an_image_fails_that_page_only(tmp_path):
    fake = tmp_path / "p001.jpg"
    fake.write_bytes(b"not an image")
    words = "alpha beta gamma"
    result = vote_page("p001.jpg", fake, transcripts(("model-a", words), ("model-b", words)))

    assert result.ballot.slots[0].bbox is None
    assert result.ballot.slots[0].consensus == "alpha"


class _Counting:
    """A backend that records which pages it was actually asked to read."""

    model_id = "counter"

    def __init__(self):
        self.asked: list[str] = []

    def transcribe(self, image_path):
        self.asked.append(image_path.stem)
        return Transcript(backend=self.model_id, lines=["fresh"])

    def release(self):
        pass


@pytest.mark.parametrize("reuse", [True, False])
def test_a_cached_page_is_read_again_only_when_the_cache_is_refused(folder, reuse):
    """A run killed halfway resumes: the pages already done cost nothing the second time."""
    workspace = folder / ".manyhands"
    pages = discover_pages(folder, workspace)
    stems = [path.stem for _, path in pages]
    cache_transcript(Transcript(backend=_Counting.model_id, lines=["cached"]), stems[0], workspace)
    backend = _Counting()

    per_page = transcribe_pages(pages, [backend], workspace, reuse=reuse)

    assert backend.asked == (stems[1:] if reuse else stems)
    assert per_page[0][0].lines == (["cached"] if reuse else ["fresh"])
