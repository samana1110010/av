# SyncSense

SyncSense adds a contextually matched soundtrack to a silent video. It searches
500 audio candidates using trained cross-modal PyTorch encoders, selects the
best match, loops or trims the audio to the video duration, and returns a
playable and downloadable synchronized MP4.

## Run the web app

Requirements: Python 3.12, `uv`, and FFmpeg available on `PATH`.

```powershell
uv sync
uv run python main.py
```

Open <http://127.0.0.1:5000>. Uploads are processed in a temporary directory
and removed after inference. Generated MP4 files remain available for one hour.
The maximum upload size is 100 MB.

## Test the project

```powershell
uv run pytest -q
uv run python -m evaluation.evaluate
uv run python -m inference.index
uv run python -m inference.retrieve
```

The tests cover datasets, model output, media synchronization, Flask routes,
and a real audio-to-video retrieval request. The web app health endpoint is
available at <http://127.0.0.1:5000/api/health>.

## Command-line retrieval

The CLI encodes the supplied video at request time; it does not select a stored
training-video embedding. It searches the current audio FAISS gallery and
creates a synchronized MP4 from the top match:

```powershell
uv run python -m inference.retrieve `
  --video "path/to/silent-video.mp4" `
  --output "synced-output.mp4"
```

Use `--no-sync` to print the top matches without generating a video. If the
audio embeddings are newer than `embeddings/audio.index`, the CLI rebuilds the
index automatically.

## Required artifacts

- `checkpoints/audio_encoder_final.pt`
- `checkpoints/video_encoder_final.pt`
- `data/video_embeddings.npy` and `data/video_ids.npy`
- `embeddings/audio_embeddings.npy` and `embeddings/audio_ids.npy`
- `data/train.csv`, extracted frames, audio files, and source videos

If models are retrained, regenerate both embedding galleries before running the
web app so their order remains aligned with `data/train.csv`.
