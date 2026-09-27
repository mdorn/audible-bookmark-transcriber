import json

import numpy as np

from audible_bookmark_transcriber.annotations import Annotation
from audible_bookmark_transcriber.notes import load_notes
from audible_bookmark_transcriber.pipeline import transcribe_annotations

CLIP = Annotation(id="c1", kind="clip", start_ms=1_000, end_ms=2_000)
BOOKMARK = Annotation(id="b1", kind="bookmark", start_ms=500, end_ms=1_500)


class FakeBook:
    def __init__(self) -> None:
        self.decoded: list[tuple[int, int]] = []

    def decode_range(self, start_ms: int, end_ms: int) -> np.ndarray:
        self.decoded.append((start_ms, end_ms))
        return np.zeros(16, dtype=np.float32)

    def chapter_at(self, position_ms: int) -> str | None:
        return "Chapter 1"


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, samples: np.ndarray) -> str:
        self.calls += 1
        return f"text {self.calls}"


def test_transcribes_and_saves_all(tmp_path):
    notes_file = tmp_path / "notes.json"
    transcriber = FakeTranscriber()

    notes = transcribe_annotations(
        FakeBook(), [CLIP, BOOKMARK], transcriber, notes_file
    )

    assert [(n.id, n.chapter, n.text) for n in notes] == [
        ("b1", "Chapter 1", "text 2"),
        ("c1", "Chapter 1", "text 1"),
    ]
    assert load_notes(notes_file) == notes


def test_resumes_without_retranscribing(tmp_path):
    notes_file = tmp_path / "notes.json"
    transcribe_annotations(FakeBook(), [CLIP], FakeTranscriber(), notes_file)

    book, transcriber = FakeBook(), FakeTranscriber()
    transcribe_annotations(book, [CLIP, BOOKMARK], transcriber, notes_file)

    assert book.decoded == [(500, 1_500)]
    assert transcriber.calls == 1


def test_force_retranscribes_everything(tmp_path):
    notes_file = tmp_path / "notes.json"
    transcribe_annotations(FakeBook(), [CLIP], FakeTranscriber(), notes_file)

    transcriber = FakeTranscriber()
    transcribe_annotations(FakeBook(), [CLIP], transcriber, notes_file, force=True)

    assert transcriber.calls == 1


def test_legacy_transcription_is_reused_for_matching_clip(tmp_path):
    notes_file = tmp_path / "notes.json"
    notes_file.write_text(json.dumps({"1000": "from the OpenAI days"}))
    transcriber = FakeTranscriber()

    [note] = transcribe_annotations(FakeBook(), [CLIP], transcriber, notes_file)

    assert transcriber.calls == 0
    assert (note.id, note.text) == ("c1", "from the OpenAI days")
    assert [n.id for n in load_notes(notes_file)] == ["c1"]


def test_saves_clips_when_requested(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(
        "audible_bookmark_transcriber.pipeline.save_clip",
        lambda samples, path: saved.append(path.name),
    )

    transcribe_annotations(
        FakeBook(),
        [CLIP],
        FakeTranscriber(),
        tmp_path / "notes.json",
        clips_dir=tmp_path / "clips",
    )

    assert saved == ["clip_1000.mp3"]
