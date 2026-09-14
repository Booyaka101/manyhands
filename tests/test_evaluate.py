"""Ground-truth parsing and the error-capture scoring."""

from __future__ import annotations

import pytest
from PIL import Image

from conftest import DATA, stream, vote
from manyhands.evaluate import (
    DatasetScore,
    GroundTruth,
    GroundTruthError,
    consensus_spans,
    find_pairs,
    format_pages,
    format_report,
    load_ground_truth,
    score_page,
)

TRUTH = DATA / "groundtruth"
LINE_ONE = "Buried this day John Fenwick of the parish"
LINE_TWO = "of Saint Mary, aged three score and ten"


def test_alto_with_one_string_per_line():
    truth = load_ground_truth(TRUTH / "alto-line.xml")

    assert truth.schema == "alto"
    assert truth.image == "register-p014.jpg"
    assert truth.lines == [LINE_ONE, LINE_TWO]


def test_alto_with_one_string_per_word_uses_the_explicit_spaces():
    truth = load_ground_truth(TRUTH / "alto-words.xml")

    assert truth.lines == ["Buried this day John Fenwick", "of the pa-", "rish"]
    assert truth.image == "register-p015.tif"


def test_page_xml_takes_the_line_level_text_equiv():
    truth = load_ground_truth(TRUTH / "page.xml")

    assert truth.schema == "page"
    assert truth.image == "register-p016.png"
    assert truth.lines == [LINE_ONE, LINE_TWO]


def test_alto_and_page_transcribing_the_same_page_agree():
    assert load_ground_truth(TRUTH / "alto-line.xml").text == load_ground_truth(TRUTH / "page.xml").text


def test_a_schema_eval_cannot_read_is_named_in_the_error():
    with pytest.raises(GroundTruthError) as caught:
        load_ground_truth(TRUTH / "tei.xml")

    assert "neither ALTO nor PAGE-XML" in str(caught.value)
    assert "<tei>" in str(caught.value)


def test_unparsable_xml_is_reported_not_raised_raw(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("<alto><TextLine>", encoding="utf-8")

    with pytest.raises(GroundTruthError) as caught:
        load_ground_truth(broken)

    assert "cannot read ground truth broken.xml" in str(caught.value)


def test_a_missing_file_is_reported_not_raised_raw(tmp_path):
    with pytest.raises(GroundTruthError):
        load_ground_truth(tmp_path / "nope.xml")


def test_consensus_spans_map_every_word_to_its_characters():
    ballot = vote([stream(LINE_ONE, LINE_TWO)] * 2, ["a", "b"])
    text, spans = consensus_spans(ballot)

    assert text == f"{LINE_ONE}\n{LINE_TWO}"
    assert len(spans) == len(LINE_ONE.split()) + len(LINE_TWO.split())
    assert text[spans[4][0] : spans[4][1]] == "Fenwick"
    assert all(flagged is False for _, _, flagged in spans)


def test_a_flagged_error_is_captured():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    wrong = LINE_ONE.replace("Fenwick", "Renwick")
    ballot = vote([stream(wrong, LINE_TWO), stream(LINE_ONE, LINE_TWO)], ["a", "b"])
    score = score_page("p.jpg", ballot, truth)

    assert score.error_chars > 0
    assert score.capture_rate == 1.0
    assert score.flagged == 1
    assert score.flag_rate == pytest.approx(1 / score.slots)


def test_an_error_both_backends_share_is_not_captured():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    wrong = LINE_ONE.replace("Fenwick", "Renwick")
    ballot = vote([stream(wrong, LINE_TWO)] * 2, ["a", "b"])
    score = score_page("p.jpg", ballot, truth)

    assert score.flagged == 0
    assert score.error_chars > 0
    assert score.capture_rate == 0.0
    assert score.cer > 0


def test_a_perfect_transcription_scores_no_errors_and_no_flags():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    ballot = vote([stream(LINE_ONE, LINE_TWO)] * 2, ["a", "b"])
    score = score_page("p.jpg", ballot, truth)

    assert score.error_chars == 0
    assert score.cer == 0.0
    assert score.capture_rate == 1.0
    assert score.flag_rate == 0.0


def test_a_word_the_consensus_dropped_counts_against_the_neighbouring_slot():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    short = LINE_ONE.replace("John ", "")
    ballot = vote([stream(short, LINE_TWO), stream(short, LINE_TWO), stream(LINE_ONE, LINE_TWO)], "abc")
    score = score_page("p.jpg", ballot, truth)

    assert score.error_chars >= len("John")
    assert score.captured_chars > 0


def test_an_empty_consensus_still_scores():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    score = score_page("p.jpg", vote([[], []], ["a", "b"]), truth)

    assert score.slots == 0
    assert score.flag_rate == 0.0
    assert score.cer == 1.0
    assert score.error_chars == len(truth.text)


def test_dataset_totals_weight_cer_by_page_length():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    clean = score_page("a.jpg", vote([stream(LINE_ONE, LINE_TWO)] * 2, ["a", "b"]), truth)
    wrong = score_page(
        "b.jpg",
        vote([stream(LINE_ONE.replace("Fenwick", "Renwick"), LINE_TWO), stream(LINE_ONE, LINE_TWO)], "ab"),
        truth,
    )
    score = DatasetScore(pages=[clean, wrong])

    assert score.slots == clean.slots + wrong.slots
    assert score.flagged == 1
    assert score.error_chars == wrong.error_chars
    assert 0 < score.cer < wrong.cer


def test_an_empty_dataset_score_does_not_divide_by_zero():
    score = DatasetScore()

    assert score.cer == 0.0
    assert score.flag_rate == 0.0
    assert score.capture_rate == 1.0


def test_find_pairs_matches_by_stem_recursively(tmp_path):
    (tmp_path / "sub").mkdir()
    for stem in ("p001", "p002"):
        Image.new("L", (8, 8), 255).save(tmp_path / "sub" / f"{stem}.png")
        (tmp_path / "sub" / f"{stem}.xml").write_text("<alto/>", encoding="utf-8")
    Image.new("L", (8, 8), 255).save(tmp_path / "unpaired.png")

    pairs = find_pairs(tmp_path)

    assert [image.stem for image, _ in pairs] == ["p001", "p002"]
    assert all(xml.suffix == ".xml" for _, xml in pairs)


def test_find_pairs_explains_what_it_wanted_when_it_finds_nothing(tmp_path):
    with pytest.raises(GroundTruthError) as caught:
        find_pairs(tmp_path)

    assert "page001.jpg and page001.xml" in str(caught.value)


def test_find_pairs_rejects_a_path_that_is_not_a_folder(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(GroundTruthError) as caught:
        find_pairs(target)

    assert "not a folder" in str(caught.value)


def test_the_printed_report_states_both_headline_numbers():
    truth = load_ground_truth(TRUTH / "alto-line.xml")
    page = score_page(
        "p.jpg",
        vote([stream(LINE_ONE.replace("Fenwick", "Renwick"), LINE_TWO), stream(LINE_ONE, LINE_TWO)], "ab"),
        truth,
    )
    text = format_report(DatasetScore(pages=[page]))

    assert "error-capture rate" in text
    assert "flag rate" in text
    assert "consensus CER" in text
    assert "[met]" in text or "[not met]" in text


def test_the_per_page_table_has_a_row_per_page():
    page = GroundTruth(lines=[LINE_ONE], image=None, schema="alto")
    score = score_page("p.jpg", vote([stream(LINE_ONE)] * 2, ["a", "b"]), page)
    table = format_pages([score, score])

    assert table.splitlines()[0].startswith("page")
    assert len(table.splitlines()) == 4
