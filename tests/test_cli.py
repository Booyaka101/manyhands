"""The command line: exit codes, and the message a user gets when something is wrong."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FENWICK_LINE, PAGES, transcripts
from manyhands import __version__, cli
from manyhands.cli import app
from manyhands.pipeline import cache_transcript, vote_page, write_index, write_page

runner = CliRunner()

STUBS = "stub-a,stub-b,stub-c"


def invoke(*args, **kwargs):
    return runner.invoke(app, [str(arg) for arg in args], **kwargs)


#: Rich draws typer's error panels with these, and wraps the text inside them to the
#: terminal width, which splits phrases a test is looking for when tmp_path is long.
BOX_ART = str.maketrans("", "", "│╭╮╰╯─")

#: On a CI runner rich decides it has colour, so every panel border also carries an SGR
#: pair. Stripping the box art alone leaves those between the words and the phrase never
#: matches.
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def output(result) -> str:
    """Everything the command printed, unwrapped and uncoloured, whichever stream it chose."""
    printed = result.output + (result.stderr if result.stderr_bytes else "")
    return " ".join(ANSI.sub("", printed).translate(BOX_ART).split())


@pytest.fixture
def page_folder(tmp_path):
    """A one-page folder with three cached transcripts that disagree on the surname."""
    folder = tmp_path / "pages"
    folder.mkdir()
    source = PAGES / "journal-p001.jpg"
    (folder / "p001.jpg").write_bytes(source.read_bytes())

    wrong = FENWICK_LINE.replace("Fenwick", "Renwick")
    for transcript in transcripts(("stub-a", FENWICK_LINE), ("stub-b", wrong), ("stub-c", wrong)):
        cache_transcript(transcript, "p001", folder / ".manyhands")
    return folder


def test_version_prints_the_shipped_version():
    result = invoke("--version")

    assert result.exit_code == 0
    assert result.output.strip() == f"manyhands {__version__}"


def test_help_lists_the_three_commands():
    result = invoke("--help")

    assert result.exit_code == 0
    for command in ("run", "confirm", "eval"):
        assert command in result.output


def test_no_arguments_prints_help_rather_than_a_traceback():
    result = invoke()

    assert result.exit_code != 0
    assert "Usage" in output(result)


def test_run_writes_the_three_artifacts_and_exits_zero(page_folder):
    result = invoke("run", page_folder, "--backends", STUBS)
    out = page_folder / "manyhands-out" / "p001"

    assert result.exit_code == 0
    assert (out / "consensus.txt").is_file()
    assert (out / "agreement.json").is_file()
    assert (out / "report.html").is_file()
    assert "1 page," in result.output
    assert "slots flagged" in result.output


def test_run_honours_an_explicit_output_folder(page_folder, tmp_path):
    result = invoke("run", page_folder, "--backends", STUBS, "--out", tmp_path / "elsewhere")

    assert result.exit_code == 0
    assert (tmp_path / "elsewhere" / "p001" / "report.html").is_file()
    assert not (page_folder / "manyhands-out").exists()


def test_run_names_a_backend_it_cannot_resolve_and_carries_on_with_the_rest(page_folder):
    result = invoke("run", page_folder, "--backends", "stub-a,stub-b,stub-missing")

    assert result.exit_code == 0
    assert "unknown backend 'stub-missing'" in output(result)
    assert (page_folder / "manyhands-out" / "p001" / "report.html").is_file()


def test_run_refuses_a_single_backend(page_folder):
    result = invoke("run", page_folder, "--backends", "stub-a")

    assert result.exit_code == 2
    assert "at least two backends" in output(result)


def test_run_fails_when_no_backend_read_anything(page_folder):
    """Zero flags because nothing was transcribed is a failed run, not a clean one."""
    (page_folder / "p001.jpg").rename(page_folder / "p404.jpg")

    result = invoke("run", page_folder, "--backends", STUBS)

    assert result.exit_code == 3
    assert "every backend failed on every page" in output(result)


def test_run_explains_a_missing_folder(tmp_path):
    result = invoke("run", tmp_path / "nope", "--backends", "olmocr,paddle")

    assert result.exit_code == 2
    assert "does not exist" in output(result)
    assert "Traceback" not in output(result)


def test_run_explains_a_folder_with_no_pages(tmp_path):
    (tmp_path / "readme.txt").write_text("no images here", encoding="utf-8")
    result = invoke("run", tmp_path, "--backends", "olmocr,paddle")

    assert result.exit_code == 2
    assert "holds no pages" in output(result)


def test_run_refuses_to_start_on_a_broken_glossary(page_folder):
    (page_folder / ".manyhands" / "glossary.json").write_text("{oops", encoding="utf-8")
    result = invoke("run", page_folder, "--backends", STUBS)

    assert result.exit_code == 2
    assert "cannot read glossary" in output(result)


def test_run_with_no_glossary_ignores_a_broken_one(page_folder):
    (page_folder / ".manyhands" / "glossary.json").write_text("{oops", encoding="utf-8")
    result = invoke("run", page_folder, "--backends", STUBS, "--no-glossary")

    assert result.exit_code == 0


def test_the_restricted_model_prints_its_licence_notice(page_folder):
    result = invoke("run", page_folder, "--backends", "stub-a,stub-b,churro", "--no-cache")

    assert "Qwen-research" in output(result)


def test_confirm_records_a_slot_into_the_glossary(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    payload = json.loads(
        (page_folder / "manyhands-out" / "p001" / "agreement.json").read_text(encoding="utf-8")
    )
    contested = next(entry for entry in payload["slots"] if entry["agreement"] < 1.0)

    result = invoke(
        "confirm",
        page_folder / "manyhands-out" / "p001",
        "--slot",
        contested["i"],
        "--reading",
        "Fenwick",
    )
    glossary = json.loads((page_folder / ".manyhands" / "glossary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "confirmed slot" in result.output
    assert [entry["reading"] for entry in glossary["entries"].values()] == ["Fenwick"]


def test_a_confirmed_reading_changes_the_next_run(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    payload = json.loads(
        (page_folder / "manyhands-out" / "p001" / "agreement.json").read_text(encoding="utf-8")
    )
    contested = next(entry for entry in payload["slots"] if entry["agreement"] < 1.0)
    report = page_folder / "manyhands-out" / "p001"
    invoke("confirm", report, "--slot", contested["i"], "--reading", "Fenwick")

    invoke("run", page_folder, "--backends", STUBS)
    again = json.loads(
        (page_folder / "manyhands-out" / "p001" / "agreement.json").read_text(encoding="utf-8")
    )
    slot = next(entry for entry in again["slots"] if entry["i"] == contested["i"])

    assert contested["consensus"] == "Renwick"
    assert slot["consensus"] == "Fenwick"
    assert slot["confirmed"] is True


def test_confirm_finds_the_report_from_the_page_image(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    result = invoke("confirm", page_folder / "p001.jpg", "--slot", 4, "--reading", "Fenwick")

    assert result.exit_code == 0
    assert (page_folder / ".manyhands" / "glossary.json").is_file()


def test_confirm_rejects_a_slot_the_backends_agreed_on(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    result = invoke("confirm", page_folder / "manyhands-out" / "p001", "--slot", 0, "--reading", "Buried")

    assert result.exit_code == 2
    assert "not a flagged slot" in output(result)


def test_a_reading_without_a_slot_is_refused(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    result = invoke("confirm", page_folder / "manyhands-out" / "p001", "--reading", "Fenwick")

    assert result.exit_code == 2
    assert "--reading needs --slot" in output(result)


def test_confirm_without_a_report_says_how_to_get_one(tmp_path):
    image = tmp_path / "p001.jpg"
    image.write_bytes(b"")
    result = invoke("confirm", image)

    assert result.exit_code == 2
    assert "Run `manyhands run`" in output(result)


def test_confirm_says_so_when_every_backend_agreed(tmp_path):
    folder = tmp_path / "pages"
    folder.mkdir()
    (folder / "p001.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    result = vote_page(
        "p001.jpg",
        folder / "p001.jpg",
        transcripts(("stub-a", FENWICK_LINE), ("stub-b", FENWICK_LINE)),
    )
    write_page(result, folder / "manyhands-out")
    write_index(folder / ".manyhands", [result])

    outcome = invoke("confirm", folder / "manyhands-out" / "p001")

    assert outcome.exit_code == 0
    assert "nothing to confirm" in outcome.output


def test_confirm_is_not_interactive_without_a_terminal(page_folder):
    invoke("run", page_folder, "--backends", STUBS)
    result = invoke("confirm", page_folder / "manyhands-out" / "p001")

    assert result.exit_code == 2
    assert "--slot N --reading TEXT" in output(result)


def test_run_json_summarises_the_folder_and_points_at_each_page(page_folder, tmp_path):
    """agreement.json is per page. Scripting over a folder needs the level above it."""
    summary = tmp_path / "run.json"
    result = invoke("run", page_folder, "--backends", STUBS, "--json", summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert f"summary: {summary}" in output(result)
    assert payload["slots"] == sum(page["slots"] for page in payload["pages"])
    assert payload["flagged"] == sum(page["flagged"] for page in payload["pages"])
    assert [page["page"] for page in payload["pages"]] == ["p001.jpg"]
    assert Path(payload["pages"][0]["out_dir"], "agreement.json").is_file()
    assert Path(payload["index"]).is_file()


def test_eval_scores_a_dataset_and_writes_json(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "alto-line.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    (dataset / "alto-line.xml").write_text(
        (PAGES.parent / "groundtruth" / "alto-line.xml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    truth = "Buried this day John Fenwick of the parish\nof Saint Mary, aged three score and ten"
    wrong = truth.replace("Fenwick", "Renwick")
    for transcript in transcripts(("stub-a", truth), ("stub-b", wrong), ("stub-c", truth)):
        cache_transcript(transcript, "alto-line", dataset / ".manyhands")

    result = invoke(
        "eval", dataset, "--backends", STUBS, "--per-page", "--json", tmp_path / "scores.json"
    )
    payload = json.loads((tmp_path / "scores.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "stub-a · alto-line.jpg:" in result.output
    assert "error-capture rate" in result.output
    assert "flag rate" in result.output
    assert payload["capture_rate"] == 1.0
    assert payload["slots"] == 16
    assert payload["pages"][0]["page"] == "alto-line.jpg"
    assert sum(page["error_chars"] for page in payload["pages"]) == payload["error_chars"]
    assert sum(page["captured_chars"] for page in payload["pages"]) == payload["captured_chars"]


def test_eval_writes_an_openable_index_with_the_page_reports(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "alto-line.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    (dataset / "alto-line.xml").write_text(
        (PAGES.parent / "groundtruth" / "alto-line.xml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    truth = "Buried this day John Fenwick of the parish"
    for transcript in transcripts(("stub-a", truth), ("stub-b", truth), ("stub-c", truth)):
        cache_transcript(transcript, "alto-line", dataset / ".manyhands")
    reports = tmp_path / "reports"

    result = invoke("eval", dataset, "--backends", STUBS, "--out", reports)

    assert result.exit_code == 0
    assert f"reports: {reports / 'index.html'}" in result.output
    assert 'href="alto-line/report.html"' in (reports / "index.html").read_text(encoding="utf-8")
    assert (reports / "alto-line" / "report.html").is_file()


def test_eval_explains_a_dataset_with_no_pairs(tmp_path):
    result = invoke("eval", tmp_path, "--backends", "olmocr,paddle")

    assert result.exit_code == 2
    assert "page001.jpg and page001.xml" in output(result)


def test_eval_skips_an_unreadable_ground_truth_file_and_says_why(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "p001.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    (dataset / "p001.xml").write_text("<TEI/>", encoding="utf-8")

    result = invoke("eval", dataset, "--backends", "olmocr,paddle")

    assert result.exit_code == 2
    assert "neither ALTO nor PAGE-XML" in output(result)
    assert "no pages could be scored" in output(result)


def test_a_reading_survives_a_redirected_stdout_on_a_legacy_codepage(tmp_path):
    """Windows encodes a redirected stream with the console codepage, which cannot hold
    the archaic glyphs this tool exists to show."""
    folder = tmp_path / "archaic"
    folder.mkdir()
    (folder / "p001.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    line = "Buried this day John Fenwick of the pariſh"
    for transcript in transcripts(
        ("stub-a", line), ("stub-b", line), ("stub-c", line.replace("pariſh", "poriſh"))
    ):
        cache_transcript(transcript, "p001", folder / ".manyhands")
    invoke("run", folder, "--backends", STUBS)

    finished = subprocess.run(
        [
            sys.executable, "-m", "manyhands", "confirm",
            str(folder / "manyhands-out" / "p001"), "--slot", "7", "--reading", "pariſh",
        ],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    printed = finished.stdout.decode("utf-8", "replace") + finished.stderr.decode("utf-8", "replace")

    assert finished.returncode == 0, printed
    assert "Traceback" not in printed
    assert "pariſh" in printed


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("1", "Fenwick"), ("2", "Renwick"), ("", None), ("s", None), ("Fenwicke", "Fenwicke")],
)
def test_the_review_prompt_reads_one_answer(answer, expected, monkeypatch):
    entry = {"i": 4, "agreement": 0.5, "votes": {"Fenwick": 1, "Renwick": 1}}
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: answer)

    assert cli._prompt_slot(entry) == expected


def test_quitting_the_review_stops_without_writing_the_rest(page_folder, monkeypatch):
    invoke("run", page_folder, "--backends", STUBS)
    answers = iter(["1", "q"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))
    monkeypatch.setattr(cli, "_interactive", lambda: True)

    result = invoke("confirm", page_folder / "manyhands-out" / "p001")
    glossary = json.loads((page_folder / ".manyhands" / "glossary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "1 reading written" in result.output
    assert len(glossary["entries"]) == 1


def test_one_model_named_twice_cannot_corroborate_itself(page_folder):
    """stub-a reads Fenwick alone. Listed twice it used to make the slot unanimous."""
    result = invoke("run", page_folder, "--backends", "stub-a,stub-a,stub-b")
    payload = json.loads(
        (page_folder / "manyhands-out" / "p001" / "agreement.json").read_text(encoding="utf-8")
    )
    surname = next(slot for slot in payload["slots"] if slot["consensus"] in {"Fenwick", "Renwick"})

    assert result.exit_code == 0
    assert payload["backends"] == ["stub-a", "stub-b"]
    assert surname["votes"] == {"Fenwick": 1, "Renwick": 1}
    assert surname["agreement"] == 0.5
    assert "already covers" in output(result)


def test_eval_takes_the_target_it_is_judged_against(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "alto-line.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    (dataset / "alto-line.xml").write_text(
        (PAGES.parent / "groundtruth" / "alto-line.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    truth = "Buried this day John Fenwick of the parish\nof Saint Mary, aged three score and ten"
    wrong = truth.replace("Fenwick", "Renwick")
    for transcript in transcripts(("stub-a", truth), ("stub-b", wrong), ("stub-c", truth)):
        cache_transcript(transcript, "alto-line", dataset / ".manyhands")

    lenient = invoke("eval", dataset, "--backends", STUBS, "--target-flag", "0.5")
    strict = invoke("eval", dataset, "--backends", STUBS, "--target-flag", "0.01")

    assert "capture >= 70%, flag rate < 50% [met]" in output(lenient)
    assert "capture >= 70%, flag rate < 1% [not met]" in output(strict)


def test_eval_warns_when_the_ground_truth_names_another_page(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "elsewhere.jpg").write_bytes((PAGES / "journal-p001.jpg").read_bytes())
    (dataset / "elsewhere.xml").write_text(
        (PAGES.parent / "groundtruth" / "alto-line.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    truth = "Buried this day John Fenwick of the parish"
    for transcript in transcripts(("stub-a", truth), ("stub-b", truth), ("stub-c", truth)):
        cache_transcript(transcript, "elsewhere", dataset / ".manyhands")

    result = invoke("eval", dataset, "--backends", STUBS)

    assert result.exit_code == 0
    assert "warning: elsewhere.jpg is paired with ground truth that names" in output(result)

