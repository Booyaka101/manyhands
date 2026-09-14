"""Voting, banding, tie-breaking and the glossary override."""

from __future__ import annotations

from collections import Counter

import pytest

from conftest import FENWICK_LINE, stream, vote
from manyhands.align import align
from manyhands.text import Token
from manyhands.vote import Slot, render_consensus, tally, variant_key

FIVE = ["model-a", "model-b", "model-c", "model-d", "model-e"]


def surname(ballot) -> Slot:
    """Return the slot holding the contested surname of the worked example."""
    position = FENWICK_LINE.split().index("Fenwick")
    return ballot.slots[position]


def test_four_of_five_agree_gives_amber_at_zero_point_eight(fenwick):
    ballot = vote(fenwick, FIVE)
    slot = surname(ballot)

    assert slot.consensus == "Fenwick"
    assert slot.agreement == pytest.approx(0.8)
    assert slot.band == "amber"
    assert slot.votes == {"Fenwick": 4, "Renwick": 1}
    assert slot.contested is False
    assert len(slot.readings) == 5
    assert slot.readings["model-e"].text == "Renwick"


def test_the_rest_of_the_worked_example_is_unanimous(fenwick):
    ballot = vote(fenwick, FIVE)
    flagged = [slot for slot in ballot.slots if slot.flagged]

    assert len(flagged) == 1
    assert flagged[0].consensus == "Fenwick"
    assert all(slot.band == "ok" for slot in ballot.slots if not slot.flagged)


def test_two_two_one_split_is_red_and_contested(split_vote):
    ballot = vote(split_vote, FIVE)
    slot = surname(ballot)

    assert slot.agreement == pytest.approx(0.4)
    assert slot.band == "red"
    assert slot.contested is True
    assert slot.votes == {"Fenwick": 2, "Fenwich": 2, "Renwick": 1}
    assert slot.consensus == "Fenwick"


def test_a_tie_breaks_towards_the_higher_ranked_backend():
    streams = [stream("alpha"), stream("beta")]

    assert vote(streams, ["first", "second"]).slots[0].consensus == "alpha"
    assert vote(streams, ["second", "first"]).slots[0].consensus == "alpha"
    assert vote([stream("beta"), stream("alpha")], ["first", "second"]).slots[0].consensus == "beta"


def test_a_tie_at_equal_rank_prefers_a_glossary_confirmed_reading():
    columns = align([stream("alpha"), stream("beta")])
    column = columns[0]
    # Both readings come from the same rank position, so only the glossary separates them.
    plain = tally(columns, ["a", "b"]).slots[0].consensus
    confirmed = tally(columns, ["a", "b"], glossary={"alpha|beta": "beta"}).slots[0]

    assert plain == "alpha"
    assert confirmed.consensus == "beta"
    assert confirmed.confirmed is True
    assert column.entries[0].text == "alpha"


def test_a_glossary_confirmation_overrides_the_majority(fenwick):
    key = variant_key(Counter({"Fenwick": 4, "Renwick": 1}))
    ballot = vote(fenwick, FIVE, glossary={key: "Renwick"})
    slot = surname(ballot)

    assert key == "fenwick|renwick"
    assert slot.consensus == "Renwick"
    assert slot.confirmed is True
    assert slot.agreement == pytest.approx(0.8)
    assert slot.votes == {"Fenwick": 4, "Renwick": 1}


def test_a_unanimous_slot_is_never_marked_confirmed():
    key = variant_key(Counter({"alpha": 2}))
    ballot = vote([stream("alpha"), stream("alpha")], ["a", "b"], glossary={key: "beta"})

    assert ballot.slots[0].consensus == "alpha"
    assert ballot.slots[0].confirmed is False


def test_a_backend_reading_nothing_counts_as_a_null_vote():
    columns = align([stream("alpha beta"), stream("alpha beta"), stream("alpha")])
    ballot = tally(columns, ["a", "b", "c"])
    slot = ballot.slots[1]

    assert slot.consensus == "beta"
    assert slot.agreement == pytest.approx(2 / 3)
    assert slot.votes == {"beta": 2, "": 1}
    assert slot.readings["c"] is None


def test_a_null_can_win_the_vote_and_leaves_the_word_out_of_the_consensus():
    columns = align([stream("alpha beta"), stream("alpha"), stream("alpha")])
    ballot = tally(columns, ["a", "b", "c"])

    assert ballot.slots[1].consensus == ""
    assert ballot.slots[1].flagged is True
    assert render_consensus(ballot) == "alpha"


@pytest.mark.parametrize(
    ("agreement", "band"),
    [(1.0, "ok"), (0.99, "pale"), (0.81, "pale"), (0.8, "amber"), (0.5, "amber"), (0.49, "red")],
)
def test_band_boundaries(agreement, band):
    slot = Slot(index=0, consensus="x", agreement=agreement, votes={}, readings={}, line=0)
    assert slot.band == band
    assert slot.flagged is (agreement < 1.0)


def test_variant_key_ignores_case_punctuation_and_order():
    left = variant_key(Counter({"Fenwick,": 2, "renwick": 1, "": 1}))
    right = variant_key(Counter({"Renwick": 1, "'Fenwick'": 3}))

    assert left == right == "fenwick|renwick"


def test_render_consensus_keeps_the_line_structure():
    ballot = vote([stream("one two", "three four"), stream("one two", "three four")], ["a", "b"])
    assert render_consensus(ballot) == "one two\nthree four"


def test_tally_with_no_columns_is_a_blank_ballot():
    ballot = tally([], ["a", "b"])
    assert ballot.slots == []
    assert ballot.blank is True
    assert render_consensus(ballot) == ""


def test_the_consensus_line_follows_the_backends_that_agreed():
    columns = align([stream("alpha", "beta"), stream("alpha", "beta"), stream("alpha beta")])
    ballot = tally(columns, ["a", "b", "c"])

    assert [slot.line for slot in ballot.slots] == [0, 1]
    assert isinstance(ballot.slots[1].readings["c"], Token)


def test_a_difference_of_case_alone_is_not_a_disagreement():
    ballot = vote([stream("Journal de Celestine"), stream("JOURNAL DE CELESTINE")], ["a", "b"])

    assert [slot.agreement for slot in ballot.slots] == [1.0, 1.0, 1.0]
    assert render_consensus(ballot) == "Journal de Celestine"


def test_the_consensus_spells_a_word_the_way_the_top_ranked_backend_did():
    lower, upper = stream("the parish,"), stream("THE PARISH")

    assert render_consensus(vote([lower, upper], ["a", "b"])) == "the parish,"
    assert render_consensus(vote([upper, lower], ["a", "b"])) == "THE PARISH"


def test_readings_still_show_each_backends_own_spelling():
    slot = vote([stream("Fenwick"), stream("FENWICK,")], ["a", "b"]).slots[0]

    assert slot.votes == {"Fenwick": 2}
    assert [token.text for token in slot.readings.values()] == ["Fenwick", "FENWICK,"]


def test_a_punctuation_only_token_is_not_the_same_as_reading_nothing():
    slot = vote([stream("John - Fenwick"), stream("John Fenwick")], ["a", "b"]).slots[1]

    assert slot.agreement == pytest.approx(0.5)
    assert slot.votes == {"-": 1, "": 1}


def test_the_commonest_spelling_wins_inside_a_group():
    streams = [stream("THE PARISH"), stream("the parish"), stream("the parish")]

    assert render_consensus(vote(streams, ["a", "b", "c"])) == "the parish"


def test_the_winners_include_a_backend_that_spelled_the_winning_word_differently():
    streams = [stream("John Fenwick"), stream("John FENWICK,"), stream("John Renwick")]

    slot = vote(streams, ["a", "b", "c"]).slots[1]

    assert slot.winners == ["a", "b"]
    assert slot.agreement == pytest.approx(2 / 3)
