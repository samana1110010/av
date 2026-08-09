import cv2
from pathlib import Path
import subprocess
from tqdm import tqdm

VIDEO_ROOT = Path("data/unseen_test/raw/video")
FRAME_ROOT = Path("data/unseen_test/frames")
AUDIO_ROOT = Path("data/unseen_test/audio")

FRAME_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 8
videos = list(VIDEO_ROOT.glob("*/video.mp4"))

print(f"Found {len(videos)} videos to preprocess")

for video_path in tqdm(videos):
    video_id = video_path.parent.name
    
    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        frame_ids = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
        out_dir = FRAME_ROOT / video_id
        out_dir.mkdir(exist_ok=True)
        for idx, frame_no in enumerate(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, frame = cap.read()
            if success:
                cv2.imwrite(str(out_dir / f"{idx}.jpg"), frame)
    cap.release()
    
    # Extract audio
    output = AUDIO_ROOT / f"{video_id}.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", str(output)
    ]
    subprocess.run(cmd)

print("Done preprocessing!")
