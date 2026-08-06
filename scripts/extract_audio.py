from pathlib import Path
import subprocess
from tqdm import tqdm

VIDEO_ROOT = Path("data/vggsound_selected/video")
AUDIO_ROOT = Path("data/audio")

AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

videos = list(VIDEO_ROOT.glob("*/video.mp4"))

print(f"Found {len(videos)} videos")

for video in tqdm(videos):

    video_id = video.parent.name

    output = AUDIO_ROOT / f"{video_id}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-ac",
        "1",          # mono
        "-ar",
        "16000",      # 16 kHz
        str(output)
    ]

    subprocess.run(cmd)

print("Done!")
