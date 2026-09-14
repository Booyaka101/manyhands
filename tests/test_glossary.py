"""Confirmed readings: storage, and how a confirmation reaches the next vote."""

from __future__ import annotations

import json

import pytest

from conftest import FENWICK_LINE, stream, vote
from manyhands.glossary import GLOSSARY_VERSION, Glossary, GlossaryError
from manyhands.vote import variant_key

KEY = "fenwick|renwick"


def glossary_at(tmp_path):
    return Glossary.load(tmp_path / "glossary.json")


def test_a_missing_file_loads_as_an_empty_glossary(tmp_path):
    glossary = glossary_at(tmp_path)

    assert glossary.entries == {}
    assert glossary.as_lookup() == {}


def test_a_confirmation_round_trips_through_the_file(tmp_path):
    glossary = glossary_at(tmp_path)
    glossary.confirm(KEY, "Fenwick", rejected=["Renwick", "Fenwick"], page="p001.jpg")
    glossary.save()

    payload = json.loads((tmp_path / "glossary.json").read_text(encoding="utf-8"))
    reloaded = glossary_at(tmp_path)

    assert payload["version"] == GLOSSARY_VERSION
    assert payload["entries"][KEY]["rejected"] == ["Renwick"]
    assert reloaded.entries[KEY].reading == "Fenwick"
    assert reloaded.entries[KEY].pages == ["p001.jpg"]
    assert reloaded.as_lookup() == {KEY: "Fenwick"}


def test_confirming_the_same_key_again_updates_it_and_records_both_pages(tmp_path):
    glossary = glossary_at(tmp_path)
    glossary.confirm(KEY, "Fenwick", rejected=["Renwick"], page="p001.jpg")
    glossary.confirm(KEY, "Renwick", rejected=["Fenwick"], page="p002.jpg")

    entry = glossary.entries[KEY]

    assert len(glossary.entries) == 1
    assert entry.reading == "Renwick"
    assert entry.rejected == ["Fenwick"]
    assert entry.pages == ["p001.jpg", "p002.jpg"]


def test_the_same_page_is_not_recorded_twice(tmp_path):
    glossary = glossary_at(tmp_path)
    glossary.confirm(KEY, "Fenwick", rejected=[], page="p001.jpg")
    glossary.confirm(KEY, "Fenwick", rejected=[], page="p001.jpg")

    assert glossary.entries[KEY].pages == ["p001.jpg"]


def test_an_entry_with_no_reading_stays_out_of_the_lookup(tmp_path):
    glossary = glossary_at(tmp_path)
    glossary.confirm(KEY, "", rejected=["Renwick"], page="p001.jpg")

    assert KEY in glossary.entries
    assert glossary.as_lookup() == {}


def test_corrupt_json_names_the_file(tmp_path):
    (tmp_path / "glossary.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(GlossaryError) as caught:
        glossary_at(tmp_path)

    assert "cannot read glossary" in str(caught.value)


def test_valid_json_of_the_wrong_shape_is_refused(tmp_path):
    (tmp_path / "glossary.json").write_text('{"version": 1}', encoding="utf-8")

    with pytest.raises(GlossaryError) as caught:
        glossary_at(tmp_path)

    assert "no 'entries' object" in str(caught.value)


def test_entries_that_are_not_objects_are_skipped_rather_than_crashing(tmp_path):
    (tmp_path / "glossary.json").write_text(
        json.dumps({"entries": {"good": {"reading": "Fenwick"}, "bad": "Fenwick"}}), encoding="utf-8"
    )

    assert glossary_at(tmp_path).as_lookup() == {"good": "Fenwick"}


def test_save_creates_the_workspace_folder(tmp_path):
    glossary = Glossary.load(tmp_path / "deep" / "nested" / "glossary.json")
    glossary.confirm(KEY, "Fenwick", rejected=[], page="p001.jpg")
    glossary.save()

    assert (tmp_path / "deep" / "nested" / "glossary.json").is_file()


def test_a_confirmation_decides_the_same_disagreement_on_a_later_page(tmp_path):
    wrong = FENWICK_LINE.replace("Fenwick", "Renwick")
    first = vote([stream(FENWICK_LINE), stream(wrong)], ["a", "b"])
    slot = next(slot for slot in first.slots if slot.flagged)

    glossary = glossary_at(tmp_path)
    glossary.confirm(variant_key(slot.votes), "Fenwick", rejected=list(slot.votes), page="p001.jpg")
    glossary.save()

    later = vote(
        [stream(f"{wrong} again"), stream(f"{FENWICK_LINE} again")],
        ["a", "b"],
        glossary=glossary_at(tmp_path).as_lookup(),
    )
    decided = next(slot for slot in later.slots if slot.flagged)

    assert decided.consensus == "Fenwick"
    assert decided.confirmed is True


def test_the_key_ignores_case_and_edge_punctuation():
    from collections import Counter

    assert variant_key(Counter({"Fenwick,": 2, "renwick": 1})) == KEY
    assert variant_key(Counter({"Renwick": 1, "Fenwick": 2})) == KEY
    assert variant_key(Counter({"": 3})) == ""
