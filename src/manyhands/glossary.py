"""The per-corpus glossary of readings a human has confirmed.

A glossary entry is filed under the set of competing readings, not under a page and a
slot index, so confirming ``Fenwick`` over ``Renwick`` once settles every later page
where the same models produce the same disagreement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GLOSSARY_VERSION = 1


@dataclass(slots=True)
class Entry:
    """One confirmed reading.

    :param reading: The surface form the human accepted.
    :param rejected: The competing readings that were on offer.
    :param pages: Pages where this decision was made, most recent last.
    """

    reading: str
    rejected: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Glossary:
    """A loaded glossary, addressed by variant key."""

    path: Path
    entries: dict[str, Entry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Glossary:
        """Load a glossary, returning an empty one when the file does not exist yet.

        :param path: Path to ``glossary.json``.
        :raises GlossaryError: If the file exists but cannot be read or parsed.
        """
        if not path.is_file():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            message = f"cannot read glossary at {path}: {exc}"
            raise GlossaryError(message) from exc
        raw_entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(raw_entries, dict):
            message = f"glossary at {path} has no 'entries' object"
            raise GlossaryError(message)
        entries = {
            str(key): Entry(
                reading=str(value.get("reading", "")),
                rejected=[str(item) for item in value.get("rejected") or []],
                pages=[str(item) for item in value.get("pages") or []],
            )
            for key, value in raw_entries.items()
            if isinstance(value, dict)
        }
        return cls(path=path, entries=entries)

    def save(self) -> None:
        """Write the glossary to disk, creating the workspace folder if needed."""
        payload: dict[str, Any] = {
            "version": GLOSSARY_VERSION,
            "entries": {
                key: {"reading": entry.reading, "rejected": entry.rejected, "pages": entry.pages}
                for key, entry in sorted(self.entries.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def confirm(self, key: str, reading: str, *, rejected: list[str], page: str) -> None:
        """Record a human decision for one variant key."""
        entry = self.entries.setdefault(key, Entry(reading=reading))
        entry.reading = reading
        competing = {*entry.rejected, *rejected}
        entry.rejected = sorted(item for item in competing if item and item != reading)
        if page not in entry.pages:
            entry.pages.append(page)

    def as_lookup(self) -> dict[str, str]:
        """Return the mapping the vote consults: variant key to accepted reading."""
        return {key: entry.reading for key, entry in self.entries.items() if entry.reading}


class GlossaryError(RuntimeError):
    """Raised when a glossary file exists but is not usable."""
