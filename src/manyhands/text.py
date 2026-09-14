"""Token model and the text normalisation shared by alignment, voting and evaluation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Long s, ligatures and the private-use characters that eScriptorium-era transcriptions
# carry are folded away before matching, so two backends that differ only in how they
# render an archaic glyph are not reported as a disagreement.
_FOLD_MAP = {
    "ſ": "s",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "‐": "-",
    "‑": "-",
    "­": "",
}

_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>«»‹›„“”‘’*_/\\|—–-"

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Token:
    """One whitespace-delimited token as one backend read it.

    :param text: Surface form exactly as the backend produced it.
    :param line: Zero-based index of the line this token was read on.
    :param start: Character offset of the token within its line.
    :param end: Character offset one past the end of the token within its line.
    """

    text: str
    line: int
    start: int
    end: int

    @property
    def key(self) -> str:
        """Return the normalised matching key for this token."""
        return normalise(self.text)


def fold(text: str) -> str:
    """Return ``text`` with archaic and typographic variants folded to ASCII-ish forms."""
    folded = "".join(_FOLD_MAP.get(char, char) for char in text)
    return unicodedata.normalize("NFC", folded)


def normalise(text: str) -> str:
    """Return the key two readings must share to count as the same word.

    Case, edge punctuation and typographic variants are ignored; letters and
    diacritics are not, because those are exactly the differences a palaeographer
    needs to see.
    """
    stripped = fold(text).strip().strip(_EDGE_PUNCT)
    return stripped.casefold()


def split_lines(text: str) -> list[str]:
    """Split OCR output into non-empty display lines, preserving order."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def tokenize_line(line: str, line_index: int) -> list[Token]:
    """Tokenize one line into whitespace-delimited tokens carrying their character span."""
    return [
        Token(text=match.group(), line=line_index, start=match.start(), end=match.end())
        for match in _TOKEN_RE.finditer(line)
    ]


def tokenize(lines: list[str]) -> list[Token]:
    """Tokenize a list of lines into a flat token stream."""
    tokens: list[Token] = []
    for index, line in enumerate(lines):
        tokens.extend(tokenize_line(line, index))
    return tokens


def plural(count: int, word: str) -> str:
    """Return ``count`` and ``word``, with an s when there is not exactly one."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]
