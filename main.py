import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from inference.classifier_inference import MultimodalEventClassifier


def main():
    parser = argparse.ArgumentParser(description="SyncSense Video Event Classification CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video file")
    args = parser.parse_args()

    raw_path = Path(args.video)
    if not raw_path.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    video_path = raw_path.resolve()

    try:
        classifier = MultimodalEventClassifier()
        results = classifier.predict_video(video_path)

        print("\n========================================")
        print("SyncSense Video Event Classification")
        print("========================================")
        print(f"\nInput video: {video_path}\n")
        print(f"Video-only prediction: {results['video_only']['prediction']} ({results['video_only']['confidence']:.2f}%)")
        print(f"Audio-only prediction: {results['audio_only']['prediction']} ({results['audio_only']['confidence']:.2f}%)")
        print(f"Fusion prediction:     {results['fusion']['prediction']} ({results['fusion']['confidence']:.2f}%)\n")
        print("========================================\n")

    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
