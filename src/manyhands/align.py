"""Anchor-based progressive multiple alignment of N token streams, ROVER style.

The lattice is built from token keys that occur exactly once in every stream. Those
anchors are certain, so they become single-token columns and cut the page into
independent gaps. Each gap is aligned recursively: a key that was ambiguous across the
whole page is often unique inside a twenty-token gap, so the anchor pass keeps paying
off as it descends. When a gap has no anchors left it falls back to progressive
pairwise alignment against the longest stream, which is where null tokens appear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from manyhands.text import Token

# Recursion is bounded by the anchor pass removing at least one token per stream, but a
# pathological page should still not blow the Python stack.
_MAX_DEPTH = 64


@dataclass(slots=True)
class Column:
    """One aligned slot: at most one token from each stream.

    :param entries: Stream index to the token that stream contributed, if any.
    """

    entries: dict[int, Token] = field(default_factory=dict)

    def token(self, stream: int) -> Token | None:
        """Return the token stream ``stream`` contributed, or ``None`` for a null."""
        return self.entries.get(stream)


def align(streams: Sequence[Sequence[Token]]) -> list[Column]:
    """Align N token streams into a single ordered list of columns.

    :param streams: One token stream per backend, in backend order.
    :returns: Columns in reading order; every stream appears at most once per column.
    """
    if not streams:
        return []
    return _align_region([list(stream) for stream in streams], depth=0)


def _align_region(streams: list[list[Token]], *, depth: int) -> list[Column]:
    if all(not stream for stream in streams):
        return []
    if depth < _MAX_DEPTH:
        chain = _anchor_chain(streams)
        if chain:
            return _split_on_anchors(streams, chain, depth=depth)
    return _align_gap(streams)


def _anchor_chain(streams: list[list[Token]]) -> list[tuple[int, ...]]:
    """Return anchor positions as a chain of per-stream index tuples, increasing in all streams."""
    positions = _unique_shared_positions(streams)
    if not positions:
        return []

    candidates = sorted(positions.values())
    if all(
        all(a < b for a, b in zip(earlier, later, strict=True))
        for earlier, later in zip(candidates, candidates[1:], strict=False)
    ):
        # Backends normally return the page in the same order, so every anchor already
        # advances in every stream and the quadratic search below would only confirm it.
        return candidates

    # Longest chain that advances in every stream at once: an N-dimensional LIS.
    best = [1] * len(candidates)
    previous: list[int | None] = [None] * len(candidates)
    for i, later in enumerate(candidates):
        for j, earlier in enumerate(candidates[:i]):
            if best[j] + 1 > best[i] and all(a < b for a, b in zip(earlier, later, strict=True)):
                best[i] = best[j] + 1
                previous[i] = j
    end = max(range(len(candidates)), key=lambda i: best[i])
    chain: list[tuple[int, ...]] = []
    cursor: int | None = end
    while cursor is not None:
        chain.append(candidates[cursor])
        cursor = previous[cursor]
    chain.reverse()
    return chain


def _unique_shared_positions(streams: list[list[Token]]) -> dict[str, tuple[int, ...]]:
    """Return keys that occur exactly once in every stream, mapped to their positions."""
    per_stream: list[dict[str, int]] = []
    for stream in streams:
        seen: dict[str, int] = {}
        duplicated: set[str] = set()
        for index, token in enumerate(stream):
            key = token.key
            if not key or key in seen:
                duplicated.add(key)
            else:
                seen[key] = index
        for key in duplicated:
            seen.pop(key, None)
        per_stream.append(seen)

    if not per_stream:
        return {}
    shared = set(per_stream[0])
    for seen in per_stream[1:]:
        shared &= set(seen)
    return {key: tuple(seen[key] for seen in per_stream) for key in shared}


def _split_on_anchors(
    streams: list[list[Token]],
    chain: list[tuple[int, ...]],
    *,
    depth: int,
) -> list[Column]:
    columns: list[Column] = []
    cursors = [0] * len(streams)
    for anchor in chain:
        gap = [streams[s][cursors[s] : anchor[s]] for s in range(len(streams))]
        columns.extend(_align_region(gap, depth=depth + 1))
        columns.append(Column(entries={s: streams[s][anchor[s]] for s in range(len(streams))}))
        cursors = [index + 1 for index in anchor]
    tail = [streams[s][cursors[s] :] for s in range(len(streams))]
    columns.extend(_align_region(tail, depth=depth + 1))
    return columns


def _align_gap(streams: list[list[Token]]) -> list[Column]:
    """Align an anchor-free region progressively against the longest stream."""
    pivot_index = max(range(len(streams)), key=lambda s: len(streams[s]))
    pivot = streams[pivot_index]
    if not pivot:
        return []

    pivot_keys = [token.key for token in pivot]
    anchored: list[Column] = [Column(entries={pivot_index: token}) for token in pivot]
    # gaps[i] holds the columns that sit before pivot token i; gaps[len(pivot)] is the tail.
    gaps: list[list[Column]] = [[] for _ in range(len(pivot) + 1)]

    for stream_index, stream in enumerate(streams):
        if stream_index == pivot_index:
            continue
        _merge_stream(stream_index, stream, pivot_keys, anchored, gaps)

    columns: list[Column] = []
    for i, column in enumerate(anchored):
        columns.extend(gaps[i])
        columns.append(column)
    columns.extend(gaps[len(pivot)])
    return [column for column in columns if column.entries]


def _merge_stream(
    stream_index: int,
    stream: list[Token],
    pivot_keys: list[str],
    anchored: list[Column],
    gaps: list[list[Column]],
) -> None:
    keys = [token.key for token in stream]
    matcher = SequenceMatcher(a=pivot_keys, b=keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                anchored[i1 + offset].entries[stream_index] = stream[j1 + offset]
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for offset in range(paired):
                anchored[i1 + offset].entries[stream_index] = stream[j1 + offset]
            if j2 - j1 > paired:
                _place_in_gap(gaps[i2], stream_index, stream[j1 + paired : j2])
        elif tag == "insert":
            _place_in_gap(gaps[i1], stream_index, stream[j1:j2])


def _place_in_gap(gap: list[Column], stream_index: int, tokens: list[Token]) -> None:
    """Add tokens to a gap, sharing columns with the other streams that landed there.

    Two streams inserting at the same site are reading the same piece of the page, so
    their tokens belong in the same column and become competing candidates. A matching
    key wins the column outright; otherwise the next free column in order takes it.
    """
    cursor = 0
    for token in tokens:
        free = [
            position
            for position in range(cursor, len(gap))
            if stream_index not in gap[position].entries
        ]
        slot = next(
            (
                position
                for position in free
                if any(other.key == token.key for other in gap[position].entries.values())
            ),
            free[0] if free else None,
        )
        if slot is None:
            gap.append(Column(entries={stream_index: token}))
            cursor = len(gap)
        else:
            gap[slot].entries[stream_index] = token
            cursor = slot + 1
