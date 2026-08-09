import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from inference.sync_video import sync_video
from inference.video_query import load_video_tensor
from models.video_encoder import VideoEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "video_encoder_final.pt"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "embeddings" / "audio_embeddings.npy"
DEFAULT_AUDIO_IDS = PROJECT_ROOT / "embeddings" / "audio_ids.npy"
DEFAULT_INDEX = PROJECT_ROOT / "embeddings" / "audio.index"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "train.csv"


def encode_video(video_path: str | Path, checkpoint: str | Path = DEFAULT_CHECKPOINT):
    """Encode an arbitrary video as one normalized 128-D query vector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Video checkpoint not found: {checkpoint}")

    model = VideoEncoder(embedding_dim=128, weights=None).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    frames = load_video_tensor(video_path).to(device)
    with torch.inference_mode():
        return model(frames).cpu().numpy().astype("float32")


def load_audio_index(
    index_path: str | Path = DEFAULT_INDEX,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS,
):
    """Load the FAISS gallery, rebuilding it when embeddings are newer."""
    index_path = Path(index_path)
    embeddings_path = Path(embeddings_path)
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Audio embeddings not found: {embeddings_path}")

    stale = (
        not index_path.is_file()
        or index_path.stat().st_mtime < embeddings_path.stat().st_mtime
    )
    if stale:
        embeddings = np.load(embeddings_path).astype("float32")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2-D audio gallery, got {embeddings.shape}")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
    else:
        index = faiss.read_index(str(index_path))
    return index


def retrieve_audio(
    video_path: str | Path,
    top_k: int = 5,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    index_path: str | Path = DEFAULT_INDEX,
    embeddings_path: str | Path = DEFAULT_EMBEDDINGS,
    audio_ids_path: str | Path = DEFAULT_AUDIO_IDS,
    metadata_path: str | Path = DEFAULT_METADATA,
):
    query = encode_video(video_path, checkpoint)
    index = load_audio_index(index_path, embeddings_path)
    audio_ids = np.load(audio_ids_path).astype(str)
    if index.ntotal != len(audio_ids):
        raise RuntimeError(
            f"FAISS contains {index.ntotal} vectors but audio_ids contains {len(audio_ids)} IDs"
        )
    if query.shape[1] != index.d:
        raise RuntimeError(
            f"Query dimension {query.shape[1]} does not match FAISS dimension {index.d}"
        )

    count = min(max(1, top_k), index.ntotal)
    scores, indices = index.search(query, count)
    metadata = pd.read_csv(metadata_path).set_index("video_id")
    matches = []
    for rank, (score, index_position) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        audio_id = str(audio_ids[index_position])
        row = metadata.loc[audio_id] if audio_id in metadata.index else None
        matches.append({
            "rank": rank,
            "audio_id": audio_id,
            "class_name": str(row["class_name"]) if row is not None else "unknown",
            "score": float(score),
        })
    return matches


def build_parser():
    parser = argparse.ArgumentParser(
        description="Match an arbitrary video to gallery audio and create a synced MP4."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-sync", action="store_true", help="Print matches without creating an MP4"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    matches = retrieve_audio(args.video, top_k=args.top_k)

    print(f"Query video: {args.video}")
    print("Top audio matches:")
    for match in matches:
        print(
            f"{match['rank']}. {match['class_name']} / {match['audio_id']} "
            f"(similarity={match['score']:.4f})"
        )

    if not args.no_sync:
        best = matches[0]
        audio_path = DEFAULT_AUDIO_DIR / f"{best['audio_id']}.wav"
        sync_video(args.video, audio_path, args.output)
        print(f"Synced output: {args.output.resolve()}")
    return matches


if __name__ == "__main__":
    main()
