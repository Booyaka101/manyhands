"""The aligner, against hand-built token streams."""

from __future__ import annotations

from conftest import stream, texts
from manyhands.align import align

BASE = "the quick brown fox jumps over the lazy dog"


def test_identical_streams_produce_one_column_per_token():
    streams = [stream(BASE) for _ in range(3)]
    columns = align(streams)

    assert len(columns) == len(BASE.split())
    for index in range(3):
        assert texts(columns, index) == BASE.split()


def test_insertion_gets_its_own_column_with_nulls_for_the_others():
    words = BASE.split()
    inserted = [*words[:4], "swiftly", *words[4:]]
    columns = align([stream(BASE), stream(BASE), stream(" ".join(inserted))])

    assert texts(columns, 2) == inserted
    assert texts(columns, 0) == [*words[:4], None, *words[4:]]
    assert texts(columns, 1) == texts(columns, 0)


def test_deletion_leaves_a_null_in_the_short_stream():
    words = BASE.split()
    dropped = [word for word in words if word != "brown"]
    columns = align([stream(BASE), stream(BASE), stream(" ".join(dropped))])

    assert texts(columns, 0) == words
    assert texts(columns, 2) == [words[0], words[1], None, *words[3:]]


def test_substitution_shares_a_column_with_the_word_it_replaced():
    words = BASE.split()
    changed = [*words[:2], "browne", *words[3:]]
    columns = align([stream(BASE), stream(BASE), stream(" ".join(changed))])

    assert len(columns) == len(words)
    assert texts(columns, 2) == changed
    assert columns[2].entries[0].text == "brown"
    assert columns[2].entries[2].text == "browne"


def test_backend_dropping_a_whole_line_nulls_that_line_only():
    lines = ("first line of the page", "second line of the page", "third line of the page")
    full = stream(*lines)
    partial = stream(lines[0], lines[2])
    columns = align([full, full, partial])

    assert texts(columns, 0) == [token.text for token in full]
    missing = [index for index, column in enumerate(columns) if 2 not in column.entries]
    assert len(missing) == len(lines[1].split())
    assert [columns[index].entries[0].line for index in missing] == [1] * len(missing)


def test_a_stream_that_swapped_two_phrases_keeps_the_longest_run_of_anchors():
    """One backend reordering part of the page has to fall back to the quadratic chain
    search, because the anchors no longer advance together."""
    words = BASE.split()
    swapped = [*words[4:], *words[:4]]
    columns = align([stream(BASE), stream(BASE), stream(" ".join(swapped))])

    assert [word for word in texts(columns, 0) if word] == words
    assert [word for word in texts(columns, 2) if word] == swapped
    assert len(columns) > len(words)


def test_empty_streams_align_to_nothing():
    assert align([]) == []
    assert align([[], []]) == []


def test_one_stream_aligns_to_itself():
    columns = align([stream(BASE)])
    assert texts(columns, 0) == BASE.split()


def test_repeated_words_do_not_collapse_into_one_column():
    repeated = "no no no no no"
    columns = align([stream(repeated), stream(repeated)])

    assert len(columns) == 5
    assert texts(columns, 0) == ["no"] * 5
    assert texts(columns, 1) == ["no"] * 5


def test_completely_different_streams_still_pair_positionally():
    columns = align([stream("alpha beta gamma"), stream("delta epsilon zeta")])

    assert len(columns) == 3
    assert texts(columns, 0) == ["alpha", "beta", "gamma"]
    assert texts(columns, 1) == ["delta", "epsilon", "zeta"]


def test_two_backends_inserting_the_same_word_share_the_column():
    words = BASE.split()
    inserted = " ".join([*words[:4], "swiftly", *words[4:]])
    columns = align([stream(BASE), stream(inserted), stream(inserted)])

    extra = [column for column in columns if 0 not in column.entries]
    assert len(extra) == 1
    assert extra[0].entries[1].text == "swiftly"
    assert extra[0].entries[2].text == "swiftly"


def test_alignment_order_is_stable_regardless_of_stream_order():
    words = BASE.split()
    short = " ".join(word for word in words if word != "lazy")
    forward = align([stream(BASE), stream(short)])
    reverse = align([stream(short), stream(BASE)])

    assert texts(forward, 0) == texts(reverse, 1)
    assert texts(forward, 1) == texts(reverse, 0)
