from __future__ import annotations

from pathlib import Path

from stridedj.domain import DraftId, PlaylistDraft, RunType, TrackId
from stridedj.music.history import PlaylistHistory


def _draft(draft_id: str, *tracks: str) -> PlaylistDraft:
    return PlaylistDraft(
        draft_id=DraftId(draft_id),
        run_type=RunType.Z2,
        target_spm=165,
        tolerance_bpm=3,
        track_ids=tuple(TrackId(t) for t in tracks),
        total_min=4 * len(tracks),
        mean_abs_bpm_error=1,
        novelty_share=0.5,
    )


def test_recent_track_uses_counts_only_the_last_playlists() -> None:
    history = PlaylistHistory(Path(":memory:"))
    history.record_created(_draft("d1", "a", "b"))
    history.record_created(_draft("d2", "b", "c"))
    history.record_created(_draft("d3", "c", "d"))
    assert history.recent_track_uses(last_playlists=2) == {"b": 1, "c": 2, "d": 1}
    assert history.recent_track_uses(last_playlists=10) == {"a": 1, "b": 2, "c": 2, "d": 1}
    assert history.recent_track_uses(last_playlists=0) == {}


def test_recording_the_same_draft_twice_is_a_no_op() -> None:
    history = PlaylistHistory(Path(":memory:"))
    history.record_created(_draft("d1", "a", "b"))
    history.record_created(_draft("d1", "a", "b"))  # e.g. the executor retried
    assert history.recent_track_uses(last_playlists=5) == {"a": 1, "b": 1}


def test_history_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "music_history.db"
    first = PlaylistHistory(path)
    first.record_created(_draft("d1", "a"))
    first.close()
    assert PlaylistHistory(path).recent_track_uses(last_playlists=3) == {"a": 1}
