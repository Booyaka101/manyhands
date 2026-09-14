"""Tokenization and the normalisation that decides what counts as a disagreement."""

from __future__ import annotations

import pytest

from manyhands.text import fold, levenshtein, normalise, split_lines, tokenize, tokenize_line


def test_a_token_carries_its_line_and_character_span():
    tokens = tokenize(["Buried this day", "John Fenwick"])

    assert [token.text for token in tokens] == ["Buried", "this", "day", "John", "Fenwick"]
    assert [token.line for token in tokens] == [0, 0, 0, 1, 1]
    assert (tokens[1].start, tokens[1].end) == (7, 11)
    assert "Buried this day"[tokens[0].start : tokens[0].end] == "Buried"


def test_runs_of_whitespace_do_not_produce_empty_tokens():
    tokens = tokenize_line("  Buried\t\tthis   day  ", 0)

    assert [token.text for token in tokens] == ["Buried", "this", "day"]


def test_blank_lines_are_dropped_and_the_rest_keep_their_order():
    assert split_lines("one\n\n  \n two \n") == ["one", "two"]


def test_case_and_edge_punctuation_do_not_count_as_a_disagreement():
    assert normalise("Fenwick,") == normalise("fenwick") == "fenwick"
    assert normalise('"parish."') == "parish"
    assert normalise("(1914-1918)") == "1914-1918"


def test_an_archaic_long_s_reads_as_the_same_word():
    assert normalise("pariſh") == "parish"
    assert fold("ﬁnal") == "final"


def test_curly_quotes_and_dashes_fold_to_ascii():
    assert fold("don’t") == "don't"
    assert fold("1914–1918") == "1914-1918"
    assert fold("soft­hyphen") == "softhyphen"


def test_letters_and_diacritics_are_kept_because_they_are_the_reading():
    assert normalise("début") != normalise("debut")
    assert normalise("Fenwick") != normalise("Renwick")


def test_a_token_that_is_only_punctuation_normalises_away():
    assert normalise("—") == ""
    assert normalise("") == ""


def test_folding_leaves_composed_characters_alone():
    assert fold("é") == "é"


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("Fenwick", "Fenwick", 0),
        ("Fenwick", "Renwick", 1),
        ("Fenwick", "Fenwich", 1),
        ("Fenwick", "Fenwic", 1),
        ("Fenwick", "", 7),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
    ],
)
def test_edit_distance(a, b, distance):
    assert levenshtein(a, b) == distance
    assert levenshtein(b, a) == distance
