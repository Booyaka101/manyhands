"""Shared fixtures and the helpers the unit tests build token streams with."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from manyhands.align import Column, align
from manyhands.backends import Transcript, build_transcript
from manyhands.text import Token, tokenize
from manyhands.vote import Ballot, tally

#: The parish-register line the worked example in the README is built on.
FENWICK_LINE = "Buried this day John Fenwick of the parish"

DATA = Path(__file__).parent / "data"
PAGES = DATA / "pages"


def stream(*lines: str) -> list[Token]:
    """Tokenize hand-written lines into one backend's token stream."""
    return tokenize(list(lines))


def readings(columns: Sequence[Column], index: int) -> list[str | None]:
    """Return what each stream contributed to one column, nulls included."""
    column = columns[index]
    streams = sorted({key for col in columns for key in col.entries})
    return [column.entries[s].text if s in column.entries else None for s in streams]


def texts(columns: Sequence[Column], stream_index: int) -> list[str | None]:
    """Return one stream's tokens in column order, nulls included."""
    return [
        column.entries[stream_index].text if stream_index in column.entries else None
        for column in columns
    ]


def vote(streams: Sequence[Sequence[Token]], backends: Sequence[str], **kwargs: object) -> Ballot:
    """Align and tally a set of hand-built streams in one step."""
    return tally(align(streams), backends, **kwargs)  # type: ignore[arg-type]


def transcripts(*pairs: tuple[str, str]) -> list[Transcript]:
    """Build transcripts from ``(backend name, raw text)`` pairs."""
    return [build_transcript(name, raw, seconds=1.0) for name, raw in pairs]


def surnames(*names: str) -> list[list[Token]]:
    """Five backends reading the worked example, differing only in the surname."""
    return [stream(FENWICK_LINE.replace("Fenwick", name)) for name in names]


@pytest.fixture
def fenwick() -> list[list[Token]]:
    """Five backends reading the worked example: four Fenwick, one Renwick."""
    return surnames("Fenwick", "Fenwick", "Fenwick", "Fenwick", "Renwick")


@pytest.fixture
def split_vote() -> list[list[Token]]:
    """Five backends split 2/2/1 on the surname."""
    return surnames("Fenwick", "Fenwick", "Fenwich", "Fenwich", "Renwick")
