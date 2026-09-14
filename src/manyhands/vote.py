"""Per-slot voting over an aligned lattice."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from manyhands.align import Column
from manyhands.text import Token, normalise

#: Agreement below this is drawn red in the report.
RED_BAND = 0.5

#: Agreement up to and including this is amber. Four backends out of five is 0.8 and is
#: meant to read as amber, so the top of the band is inclusive and the bottom is not.
AMBER_BAND = 0.8

#: The label a slot carries when a backend read nothing there.
NULL_READING = ""

#: Grouping key for a null reading. A real token is ``\S+``, so its key can never be
#: empty once punctuation-only tokens fall back to their surface form.
_NULL_KEY = ""


@dataclass(slots=True)
class Slot:
    """One voted slot of the consensus.

    :param index: Position of the slot in the consensus stream.
    :param consensus: The reading that wins the vote. Empty means "all backends skipped".
    :param agreement: Winning vote count divided by the number of voting backends.
    :param votes: Surface reading to the number of backends that produced it.
    :param readings: Backend name to the exact token it contributed, or ``None``.
    :param line: Consensus line index, taken from the backends that agreed.
    :param winners: Backends whose reading won the slot, in rank order. A backend can be
        a winner while spelling the word differently, since case and edge punctuation do
        not split a group.
    :param contested: True when the top reading did not win outright.
    :param confirmed: True when a human confirmation in the glossary decided this slot.
    """

    index: int
    consensus: str
    agreement: float
    votes: dict[str, int]
    readings: dict[str, Token | None]
    line: int
    winners: list[str] = field(default_factory=list)
    contested: bool = False
    confirmed: bool = False
    bbox: tuple[int, int, int, int] | None = None

    @property
    def flagged(self) -> bool:
        """Return True when the backends did not agree unanimously."""
        return self.agreement < 1.0

    @property
    def band(self) -> str:
        """Return the highlight band name for this slot."""
        if self.agreement >= 1.0:
            return "ok"
        if self.agreement < RED_BAND:
            return "red"
        if self.agreement <= AMBER_BAND:
            return "amber"
        return "pale"


@dataclass(slots=True)
class Ballot:
    """The result of voting a whole page.

    :param slots: Voted slots in reading order.
    :param backends: Backend names that took part in the vote, in rank order.
    :param blank: True when every backend read the page as (near) empty.
    """

    slots: list[Slot] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    blank: bool = False


def tally(
    columns: Sequence[Column],
    backends: Sequence[str],
    *,
    glossary: dict[str, str] | None = None,
) -> Ballot:
    """Vote every column of an aligned lattice.

    :param columns: Aligned columns produced by :func:`manyhands.align.align`.
    :param backends: Backend names in rank order, one per stream and all distinct;
        earlier names win ties.
    :param glossary: Normalised variant key to the reading a human accepted for it.
    :returns: The voted page.
    """
    accepted = glossary or {}
    slots: list[Slot] = []
    for index, column in enumerate(columns):
        slots.append(_vote_column(index, column, backends, accepted))
    return Ballot(slots=slots, backends=list(backends), blank=not slots)


def _vote_column(
    index: int,
    column: Column,
    backends: Sequence[str],
    accepted: dict[str, str],
) -> Slot:
    readings: dict[str, Token | None] = {
        name: column.token(stream) for stream, name in enumerate(backends)
    }
    groups = _group_readings(readings, backends)
    counts: Counter[str] = Counter({reading: len(group) for reading, group in groups.items()})
    top = max(counts.values())
    agreement = top / len(backends) if backends else 0.0
    leaders = sorted(reading for reading, count in counts.items() if count == top)
    consensus = _break_tie(leaders, groups, accepted) if len(leaders) > 1 else leaders[0]

    # A human confirmation outranks the vote, but only where the backends actually
    # disagreed; a unanimous slot was never in question.
    human = accepted.get(variant_key(counts)) if agreement < 1.0 else None
    winning = groups.get(consensus)
    if human is not None:
        consensus = human

    return Slot(
        index=index,
        consensus=consensus,
        agreement=agreement,
        votes=dict(counts.most_common()),
        readings=readings,
        line=_consensus_line(winning, readings),
        winners=[backends[rank] for rank, _ in winning or []],
        contested=len(leaders) > 1,
        confirmed=human is not None,
    )


def _group_readings(
    readings: dict[str, Token | None],
    backends: Sequence[str],
) -> dict[str, list[tuple[int, Token | None]]]:
    """Group a column's readings by normalised key, best-ranked surface form first.

    Backends that differ only in case or edge punctuation are reading the same word, so
    they must not be shown to a human as a disagreement. A token that normalises to
    nothing keeps its surface form as its key, so stray punctuation is not merged with a
    backend that read nothing at all.

    Each group is in backend rank order, which is what breaks ties inside and between
    groups.
    """
    grouped: dict[str, list[tuple[int, Token | None]]] = {}
    for rank, name in enumerate(backends):
        token = readings[name]
        key = _NULL_KEY if token is None else (token.key or token.text)
        grouped.setdefault(key, []).append((rank, token))
    return {_surface_form(group): group for group in grouped.values()}


def _surface_form(group: list[tuple[int, Token | None]]) -> str:
    """Return the spelling to print for a group: the commonest, then the best ranked."""
    spellings = Counter(token.text for _, token in group if token is not None)
    if not spellings:
        return NULL_READING
    return max(spellings, key=spellings.__getitem__)


def _break_tie(
    leaders: Sequence[str],
    groups: dict[str, list[tuple[int, Token | None]]],
    accepted: dict[str, str],
) -> str:
    """Break a tie by glossary hit, then by best backend rank.

    The brief asked for rank first. Rank cannot tie: a backend sits in exactly one group,
    so the best rank in each group is unique and deciding on it first makes the glossary
    term unreachable. A reading a human has already accepted somewhere in this corpus is
    better evidence than the order the models happen to be listed in, so it goes first.
    """
    confirmed = set(accepted.values())
    return min(leaders, key=lambda reading: (reading not in confirmed, groups[reading][0][0]))


def _consensus_line(
    winners: list[tuple[int, Token | None]] | None,
    readings: dict[str, Token | None],
) -> int:
    """Return the line index the winning backends read this slot on."""
    lines = [token.line for _, token in winners or [] if token is not None]
    if not lines:
        lines = [token.line for token in readings.values() if token is not None]
    if not lines:
        return 0
    return Counter(lines).most_common(1)[0][0]


def variant_key(counts: Counter[str]) -> str:
    """Return the stable key a glossary entry is filed under for this set of readings.

    Two slots anywhere in the corpus that produced the same set of competing readings
    share a key, so confirming ``Fenwick`` over ``Renwick`` once applies everywhere the
    same disagreement recurs.
    """
    return "|".join(sorted({normalise(reading) for reading in counts if reading}))


def group_by_line(slots: Sequence[Slot]) -> list[list[Slot]]:
    """Group slots into consensus lines, in line order.

    A slot's line index can step backwards from its neighbour's, because the line comes
    from whichever backends won the slot and they do not all break the page in the same
    place. The report, ``consensus.txt`` and the eval score have to resolve that the same
    way or they describe different text.
    """
    grouped: dict[int, list[Slot]] = {}
    for slot in slots:
        grouped.setdefault(slot.line, []).append(slot)
    return [grouped[line] for line in sorted(grouped)]


def render_consensus(ballot: Ballot) -> str:
    """Render the consensus text of a ballot, one line per consensus line index."""
    lines = [
        " ".join(slot.consensus for slot in line if slot.consensus)
        for line in group_by_line(ballot.slots)
    ]
    return "\n".join(line for line in lines if line)
