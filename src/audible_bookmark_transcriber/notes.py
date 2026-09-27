"""Reading and writing the transcribed notes file."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from audible_bookmark_transcriber.annotations import Annotation, Kind


@dataclass(frozen=True)
class Note:
    id: str
    kind: Kind
    start_ms: int
    end_ms: int
    timestamp: str
    chapter: str | None
    note: str
    text: str

    @classmethod
    def from_annotation(
        cls, annotation: Annotation, text: str, chapter: str | None = None
    ) -> "Note":
        return cls(
            id=annotation.id,
            kind=annotation.kind,
            start_ms=annotation.start_ms,
            end_ms=annotation.end_ms,
            timestamp=format_time(annotation.start_ms),
            chapter=chapter,
            note=annotation.note,
            text=text,
        )


def format_time(milliseconds: int) -> str:
    """Format a position as ``HH:MM:SS.mmm``."""
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def _migrate_legacy(data: dict[str, str]) -> list[Note]:
    """Convert the old ``{startPosition: text}`` format written by earlier versions.

    Legacy entries have no annotation id; the pipeline matches them to clips by
    start position so they are not transcribed again.
    """
    return [
        Note(
            id=f"legacy-{start}",
            kind="clip",
            start_ms=int(start),
            end_ms=int(start),
            timestamp=format_time(int(start)),
            chapter=None,
            note="",
            text=text,
        )
        for start, text in data.items()
        if text  # earlier versions stored null when transcription failed
    ]


def load_notes(path: Path) -> list[Note]:
    if not path.exists():
        return []
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return _migrate_legacy(data)
    return [Note(**item) for item in data]


def save_notes(path: Path, notes: list[Note]) -> None:
    """Atomically write ``notes`` sorted by position."""
    ordered = sorted(notes, key=lambda n: (n.start_ms, n.kind))
    payload = json.dumps([asdict(n) for n in ordered], ensure_ascii=False, indent=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
