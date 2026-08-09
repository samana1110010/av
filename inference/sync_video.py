import subprocess
from pathlib import Path


def sync_video(video_path, audio_path, output_path):
    """
    Replace the video's audio with a new audio file.
    """

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    command = [
        "ffmpeg",
        "-y",                     # overwrite output if it exists
        "-i", str(video_path),    # input video
        "-i", str(audio_path),    # input audio
        "-c:v", "copy",           # don't re-encode video
        "-c:a", "aac",            # encode audio
        "-shortest",              # stop when shortest stream ends
        str(output_path)
    ]

    subprocess.run(command, check=True)