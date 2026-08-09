from pathlib import Path

import soundfile as sf
import torch


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load audio without TorchCodec, which is sensitive to system FFmpeg DLLs."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    return waveform, sample_rate
