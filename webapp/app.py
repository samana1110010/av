import os
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify, send_from_directory

from datasets.audio_io import load_audio
from inference.classifier_inference import MultimodalEventClassifier, LABEL_MAPPING
from inference.preprocessing import extract_video_frames, extract_audio_waveform, preprocess_audio_mel

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Webapp Device: {DEVICE}")

print("Loading classification models via MultimodalEventClassifier...")
classifier_manager = MultimodalEventClassifier()
print("Models ready for Web App.")

STATIC_DIR = Path(__file__).resolve().parent / 'static'
UPLOADS_DIR = STATIC_DIR / 'uploads'
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

df_test = pd.read_csv("data/unseen_test/test.csv")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/data/<path:path>")
def data_files(path):
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    return send_from_directory(data_dir, path)


def save_uploaded_media_assets(video_path: Path) -> tuple[list[str], str, str]:
    """
    Saves uploaded video, extracts 8 JPG frames and 1 WAV audio file to static/uploads/
    Returns (frame_urls, audio_url, video_url).
    """
    stem = video_path.stem
    rel_video_path = f"/uploads/{video_path.name}"
    
    # 1. Extract 8 JPG frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    frame_urls = []

    if total_frames > 0:
        indices = [int(i * total_frames / 8) for i in range(8)]
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            out_img_path = UPLOADS_DIR / f"{stem}_frame_{i}.jpg"
            if success and frame is not None:
                cv2.imwrite(str(out_img_path), frame)
            else:
                # Blank placeholder
                blank = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(str(out_img_path), blank)
            frame_urls.append(f"/uploads/{stem}_frame_{i}.jpg")
    cap.release()

    # 2. Extract WAV audio
    out_wav_path = UPLOADS_DIR / f"{stem}_audio.wav"
    audio_url = f"/uploads/{stem}_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        str(out_wav_path)
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        pass

    return frame_urls, audio_url, rel_video_path


@app.route("/api/classify", methods=["POST", "GET"])
def classify():
    # Handle direct file upload from user
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        import time, uuid
        unique_filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{file.filename}"
        saved_video_path = UPLOADS_DIR / unique_filename
        file.save(saved_video_path)

        try:
            print(f"\n[Webapp Upload] Input video received:\n{saved_video_path.resolve()}")
            
            # Save media assets for Web UI playback
            frame_urls, audio_url, video_url = save_uploaded_media_assets(saved_video_path)
            
            # Run Inference across frozen models
            results_dict = classifier_manager.predict_video(saved_video_path)

            return jsonify({
                "success": True,
                "video_id": file.filename,
                "true_class": "Uploaded MP4 Video",
                "video_url": video_url,
                "audio_url": audio_url if (UPLOADS_DIR / f"{saved_video_path.stem}_audio.wav").exists() else "",
                "frames": frame_urls,
                "video": {
                    "class": results_dict["video_only"]["prediction"].title(),
                    "confidence": results_dict["video_only"]["confidence"]
                },
                "audio": {
                    "class": results_dict["audio_only"]["prediction"].title(),
                    "confidence": results_dict["audio_only"]["confidence"]
                },
                "fusion": {
                    "class": results_dict["fusion"]["prediction"].title(),
                    "confidence": results_dict["fusion"]["confidence"]
                },
                "results": [
                    {
                        "modality": "Video-Only Classifier",
                        "predicted_class": results_dict["video_only"]["prediction"].title(),
                        "confidence": results_dict["video_only"]["confidence"],
                        "correct": True,
                        "badge_color": "purple"
                    },
                    {
                        "modality": "Audio-Only Classifier",
                        "predicted_class": results_dict["audio_only"]["prediction"].title(),
                        "confidence": results_dict["audio_only"]["confidence"],
                        "correct": True,
                        "badge_color": "cyan"
                    },
                    {
                        "modality": "Fusion (Audio + Video)",
                        "predicted_class": results_dict["fusion"]["prediction"].title(),
                        "confidence": results_dict["fusion"]["confidence"],
                        "correct": True,
                        "badge_color": "green"
                    }
                ]
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    # Optional dataset sample browsing
    idx = request.form.get("sample_idx", None)
    if idx is None:
        idx = np.random.randint(0, len(df_test))
    else:
        idx = int(idx) % len(df_test)

    row = df_test.iloc[idx]
    video_id = row["video_id"]
    true_label = int(row["label"])
    true_class = str(row["class_name"]).title()

    frame_folder = Path(row["frame_folder"])
    images = sorted(frame_folder.glob("*.jpg"))

    from PIL import Image
    from inference.preprocessing import DETERMINISTIC_VIDEO_TRANSFORM
    frames = [DETERMINISTIC_VIDEO_TRANSFORM(Image.open(p).convert("RGB")) for p in images]
    while len(frames) < 8:
        frames.append(frames[-1])
    indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
    frames = [frames[i] for i in indices]
    frames_tensor = torch.stack(frames).unsqueeze(0)

    audio_path = Path("data/unseen_test/audio") / f"{video_id}.wav"
    if audio_path.exists():
        waveform, sr = load_audio(audio_path)
    else:
        waveform, sr = torch.zeros(1, 16000 * 3), 16000

    mel_tensor = preprocess_audio_mel(waveform, sample_rate=sr)

    preds = classifier_manager.predict_tensors(frames_tensor, mel_tensor)

    frame_urls = [f"/data/unseen_test/frames/{video_id}/{i}.jpg" for i in range(len(images))]
    audio_url = f"/data/unseen_test/audio/{video_id}.wav"

    results = [
        {
            "modality": "Video-Only Classifier",
            "predicted_class": preds["video_only"]["prediction"].title(),
            "confidence": preds["video_only"]["confidence"],
            "correct": (preds["video_only"]["class_id"] == true_label),
            "badge_color": "purple"
        },
        {
            "modality": "Audio-Only Classifier",
            "predicted_class": preds["audio_only"]["prediction"].title(),
            "confidence": preds["audio_only"]["confidence"],
            "correct": (preds["audio_only"]["class_id"] == true_label),
            "badge_color": "cyan"
        },
        {
            "modality": "Fusion (Audio + Video)",
            "predicted_class": preds["fusion"]["prediction"].title(),
            "confidence": preds["fusion"]["confidence"],
            "correct": (preds["fusion"]["class_id"] == true_label),
            "badge_color": "green"
        }
    ]

    return jsonify({
        "success": True,
        "video_id": video_id,
        "true_class": true_class,
        "video_url": "",
        "audio_url": audio_url,
        "frames": frame_urls,
        "video": {
            "class": preds["video_only"]["prediction"].title(),
            "confidence": preds["video_only"]["confidence"]
        },
        "audio": {
            "class": preds["audio_only"]["prediction"].title(),
            "confidence": preds["audio_only"]["confidence"]
        },
        "fusion": {
            "class": preds["fusion"]["prediction"].title(),
            "confidence": preds["fusion"]["confidence"]
        },
        "results": results
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
