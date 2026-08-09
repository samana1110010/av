from pathlib import Path

import pytest

from inference.sync_video import sync_video


def test_sync_video_rejects_missing_inputs(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        sync_video(tmp_path / "missing.mp4", tmp_path / "missing.wav", tmp_path / "out.mp4")
