import subprocess
from pathlib import Path


def sync_video(video_path, audio_path, output_path):
    """
    Replace the video's audio with a new audio file.
    """

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",                     # overwrite output if it exists
        "-loglevel", "error",
        "-i", str(video_path),    # input video
        "-stream_loop", "-1",     # repeat short audio to cover the whole video
        "-i", str(audio_path),    # input audio
        "-map", "0:v:0",          # video stream from the original video
        "-map", "1:a:0",          # audio stream from the replacement audio
        "-c:v", "copy",           # don't re-encode video
        "-c:a", "aac",            # encode audio
        "-af", "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000",            # browser-compatible AAC sample rate
        "-b:a", "128k",
        "-shortest",              # stop when shortest stream ends
        str(output_path)
    ]

    subprocess.run(command, check=True)
