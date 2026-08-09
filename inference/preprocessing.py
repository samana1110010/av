import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
import torchaudio
from PIL import Image
from torchvision import transforms

from datasets.audio_io import load_audio


# Deterministic transforms matching evaluation
DETERMINISTIC_VIDEO_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MEL_SPECTROGRAM_TRANSFORM = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128
    ),
    torchaudio.transforms.AmplitudeToDB()
])

TARGET_TIME = 800


def extract_video_frames(video_path: str | Path, num_frames: int = 8) -> torch.Tensor:
    """
    Extracts exactly num_frames (default 8) uniformly from an MP4 video file.
    Applies deterministic Resize(256) + CenterCrop(224) + ImageNet Normalization.
    Returns tensor of shape (1, 8, 3, 224, 224).
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Input video received:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open or read video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid or corrupted video file (0 frames): {video_path}")

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = DETERMINISTIC_VIDEO_TRANSFORM(pil_img)
            frames.append(tensor_img)
        else:
            if frames:
                frames.append(frames[-1])

    cap.release()

    if not frames:
        raise ValueError(f"Failed to read any valid RGB frames from video: {video_path}")

    # Pad if fewer than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = frames[:num_frames]
    frames_tensor = torch.stack(frames).unsqueeze(0)  # (1, 8, 3, 224, 224)
    print(f"Video frames extracted:\n{len(frames)}")
    return frames_tensor


def extract_audio_waveform(video_path: str | Path) -> tuple[torch.Tensor, int]:
    """
    Extracts 16 kHz mono WAV audio from MP4 using ffmpeg.
    Returns (waveform_tensor, sample_rate).
    Falls back to a silent zero waveform if no audio stream exists or ffmpeg fails.
    """
    video_path = Path(video_path).resolve()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_wav_path = tmp_file.name

    print(f"Audio extracted:\n{tmp_wav_path}")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        tmp_wav_path
    ]

    try:
        subprocess.run(cmd, check=True)
        waveform, sr = load_audio(tmp_wav_path)
    except Exception:
        # Fallback to 3 seconds of zero waveform at 16kHz
        waveform = torch.zeros(1, 16000 * 3)
        sr = 16000
    finally:
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000

    return waveform, sr


def preprocess_audio_mel(waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Converts 16kHz mono audio waveform to Mel spectrogram.
    Pads or truncates to TARGET_TIME (800).
    Returns tensor of shape (1, 1, 128, 800).
    """
    mel = MEL_SPECTROGRAM_TRANSFORM(waveform)  # (1, 128, T)

    if mel.shape[-1] > TARGET_TIME:
        mel = mel[..., :TARGET_TIME]
    elif mel.shape[-1] < TARGET_TIME:
        pad = TARGET_TIME - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad))

    mel_tensor = mel.unsqueeze(0)  # (1, 1, 128, 800)
    return mel_tensor


def preprocess_video_and_audio(video_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combined preprocessing returning (frames_tensor, mel_tensor).
    """
    frames_tensor = extract_video_frames(video_path, num_frames=8)
    waveform, sr = extract_audio_waveform(video_path)
    mel_tensor = preprocess_audio_mel(waveform, sample_rate=sr)
    return frames_tensor, mel_tensor
