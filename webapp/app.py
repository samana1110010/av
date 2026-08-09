import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from flask import Flask, jsonify, request, send_from_directory
from torchvision import transforms
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException

from datasets.audio_io import load_audio
from models.audio_encoder import AudioEncoder
from models.video_encoder import VideoEncoder
from inference.sync_video import sync_video
from inference.video_query import load_video_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
GENERATED_DIR = PROJECT_ROOT / "generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
TARGET_MEL_FRAMES = 800
APP_VERSION = "2026.08.09-5"
OUTPUT_TTL_SECONDS = 60 * 60

mel_transform = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.AmplitudeToDB(),
])


def _load_model(model, checkpoint_name):
    checkpoint = CHECKPOINT_DIR / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint}")
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    return model.to(DEVICE).eval()


def _load_gallery():
    metadata = pd.read_csv(DATA_DIR / "train.csv")
    video_embeddings = np.load(DATA_DIR / "video_embeddings.npy").astype("float32")
    audio_embeddings = np.load(EMBEDDING_DIR / "audio_embeddings.npy").astype("float32")
    video_ids = np.load(DATA_DIR / "video_ids.npy").astype(str)
    audio_ids = np.load(EMBEDDING_DIR / "audio_ids.npy").astype(str)

    expected = len(metadata)
    sizes = {expected, len(video_embeddings), len(audio_embeddings), len(video_ids), len(audio_ids)}
    if len(sizes) != 1:
        raise RuntimeError("Metadata, IDs, and embedding galleries have different lengths")
    if not np.array_equal(video_ids, audio_ids):
        raise RuntimeError("Video and audio embedding IDs are not aligned")
    if not np.array_equal(metadata["video_id"].astype(str).to_numpy(), video_ids):
        raise RuntimeError("Embedding IDs do not match data/train.csv order")

    return metadata, video_embeddings, audio_embeddings


video_encoder = _load_model(
    VideoEncoder(embedding_dim=128, weights=None), "video_encoder_final.pt"
)
audio_encoder = _load_model(
    AudioEncoder(embedding_dim=128, weights=None), "audio_encoder_final.pt"
)
df, gallery_video_embs, gallery_audio_embs = _load_gallery()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_generated_outputs():
    cutoff = time.time() - OUTPUT_TTL_SECONDS
    for output in GENERATED_DIR.glob("sync-*.mp4"):
        try:
            if output.stat().st_mtime < cutoff:
                output.unlink()
        except FileNotFoundError:
            continue


def process_video_query(file_path: Path) -> np.ndarray:
    batch = load_video_tensor(file_path).to(DEVICE)
    with torch.inference_mode():
        return video_encoder(batch).cpu().numpy()


def process_audio_query(file_path: Path, work_dir: Path) -> np.ndarray:
    normalized_audio = work_dir / "query.wav"
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(file_path),
        "-ac", "1", "-ar", "16000", str(normalized_audio),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg is required to process audio uploads") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip().splitlines()[-1] if error.stderr else "unknown error"
        raise ValueError(f"The uploaded audio could not be decoded: {detail}") from error

    waveform, _ = load_audio(normalized_audio)
    mel = mel_transform(waveform)
    if mel.shape[-1] > TARGET_MEL_FRAMES:
        mel = mel[..., :TARGET_MEL_FRAMES]
    elif mel.shape[-1] < TARGET_MEL_FRAMES:
        mel = torch.nn.functional.pad(mel, (0, TARGET_MEL_FRAMES - mel.shape[-1]))

    with torch.inference_mode():
        return audio_encoder(mel.unsqueeze(0).to(DEVICE)).cpu().numpy()


def _result_payload(index: int, query_type: str, score: float, rank: int):
    row = df.iloc[index]
    video_id = str(row["video_id"])
    result_type = "audio" if query_type == "video" else "video"
    return {
        "rank": rank,
        "id": video_id,
        "title": str(row["class_name"]).title(),
        "label": int(row["label"]),
        "score": round(float(score) * 100, 1),
        "type": result_type,
        "audio_url": f"/data/audio/{video_id}.wav",
        "video_url": f"/data/vggsound_selected/video/{video_id}/video.mp4",
        "frames": [f"/data/frames/{video_id}/{i}.jpg" for i in range(8)],
    }


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=100 * 1024 * 1024)
    if test_config:
        app.config.update(test_config)

    @app.after_request
    def disable_local_asset_cache(response):
        if request.path == "/" or request.path.endswith((".css", ".js", ".html")):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ready",
            "version": APP_VERSION,
            "device": str(DEVICE),
            "gallery_size": len(df),
            "embedding_dim": int(gallery_video_embs.shape[1]),
        })

    @app.post("/api/retrieve")
    def retrieve():
        query_type = request.form.get("type", "").lower()
        if query_type not in {"video", "audio"}:
            return jsonify({"success": False, "error": "Type must be video or audio"}), 400

        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"success": False, "error": "Choose a media file first"}), 400

        filename = secure_filename(upload.filename)
        extension = Path(filename).suffix.lower()
        allowed = VIDEO_EXTENSIONS if query_type == "video" else AUDIO_EXTENSIONS
        if extension not in allowed:
            supported = ", ".join(sorted(allowed))
            return jsonify({
                "success": False,
                "error": f"Unsupported {query_type} format. Use: {supported}",
            }), 415

        try:
            with tempfile.TemporaryDirectory(prefix="syncsense-") as temp_dir:
                work_dir = Path(temp_dir)
                upload_path = work_dir / f"upload{extension}"
                upload.save(upload_path)

                if query_type == "video":
                    query_embedding = process_video_query(upload_path)
                    target_embeddings = gallery_audio_embs
                else:
                    query_embedding = process_audio_query(upload_path, work_dir)
                    target_embeddings = gallery_video_embs

                similarities = (query_embedding @ target_embeddings.T)[0]
                ranking = np.argsort(similarities)[::-1][:5]
                results = [
                    _result_payload(int(index), query_type, similarities[index], rank)
                    for rank, index in enumerate(ranking, start=1)
                ]

                output = None
                if query_type == "video":
                    cleanup_generated_outputs()
                    best_match = results[0]
                    matched_audio = DATA_DIR / "audio" / f"{best_match['id']}.wav"
                    output_name = f"sync-{uuid.uuid4().hex}.mp4"
                    output_path = GENERATED_DIR / output_name
                    try:
                        sync_video(upload_path, matched_audio, output_path)
                    except (OSError, subprocess.CalledProcessError) as error:
                        output_path.unlink(missing_ok=True)
                        raise RuntimeError("The final video could not be encoded with FFmpeg") from error
                    output = {
                        "url": f"/api/output/{output_name}",
                        "download_url": f"/api/output/{output_name}?download=1",
                        "filename": f"syncsense-{Path(filename).stem}.mp4",
                        "audio": {
                            "id": best_match["id"],
                            "title": best_match["title"],
                            "score": best_match["score"],
                        },
                    }
        except (ValueError, RuntimeError) as error:
            return jsonify({"success": False, "error": str(error)}), 422

        return jsonify({
            "success": True,
            "query": {"name": filename, "type": query_type},
            "results": results,
            "output": output,
        })

    @app.get("/api/output/<filename>")
    def generated_output(filename):
        if not filename.startswith("sync-") or not filename.endswith(".mp4"):
            return jsonify({"success": False, "error": "Invalid output name"}), 404
        return send_from_directory(
            GENERATED_DIR,
            filename,
            as_attachment=request.args.get("download") == "1",
            download_name=request.args.get("name") or filename,
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"success": False, "error": "File is larger than 100 MB"}), 413

    @app.errorhandler(Exception)
    def unexpected_error(error):
        if isinstance(error, HTTPException) or not request.path.startswith("/api/"):
            return error
        if app.testing:
            raise error
        app.logger.exception("Unhandled API error")
        return jsonify({
            "success": False,
            "error": "The server could not process this request. Restart the app and try again.",
        }), 500

    @app.get("/data/<path:path>")
    def data_files(path):
        return send_from_directory(DATA_DIR, path)

    @app.get("/<path:path>")
    def static_files(path):
        return send_from_directory(STATIC_DIR, path)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
