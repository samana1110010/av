import cv2
from pathlib import Path
from tqdm import tqdm

VIDEO_ROOT = Path("data/vggsound_selected/video")
FRAME_ROOT = Path("data/frames")

FRAME_ROOT.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 8

videos = list(VIDEO_ROOT.glob("*/video.mp4"))

print(f"Found {len(videos)} videos")

for video_path in tqdm(videos):

    cap = cv2.VideoCapture(str(video_path))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        continue

    frame_ids = [
        int(i * total / NUM_FRAMES)
        for i in range(NUM_FRAMES)
    ]

    out_dir = FRAME_ROOT / video_path.parent.name
    out_dir.mkdir(exist_ok=True)

    for idx, frame_no in enumerate(frame_ids):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        success, frame = cap.read()

        if success:
            cv2.imwrite(
                str(out_dir / f"{idx}.jpg"),
                frame
            )

    cap.release()

print("Done!")
