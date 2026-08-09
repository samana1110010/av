import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from models.classifiers import VideoClassifier, AudioClassifier, FusionClassifier
from inference.preprocessing import preprocess_video_and_audio


LABEL_MAPPING = {
    0: "basketball bounce",
    1: "car passing by",
    2: "dog barking",
    3: "door slamming",
    4: "fire crackling",
    5: "footsteps on snow",
    6: "hammering nails",
    7: "raining",
    8: "ripping paper",
    9: "typing on computer keyboard"
}


class MultimodalEventClassifier:
    """
    Inference manager that loads frozen checkpoints and runs predictions
    for Video-only, Audio-only, and Fusion classifiers.
    """

    def __init__(
        self,
        video_ckpt: str = "checkpoints/video_classifier_best.pt",
        audio_ckpt: str = "checkpoints/audio_classifier_best.pt",
        fusion_ckpt: str = "checkpoints/fusion_classifier_best.pt",
        device: str | torch.device | None = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Instantiate Models
        self.video_model = VideoClassifier().to(self.device)
        self.audio_model = AudioClassifier().to(self.device)
        self.fusion_model = FusionClassifier().to(self.device)

        # Load Checkpoints
        if os.path.exists(video_ckpt):
            self.video_model.load_state_dict(
                torch.load(video_ckpt, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Video classifier checkpoint missing: {video_ckpt}")

        if os.path.exists(audio_ckpt):
            self.audio_model.load_state_dict(
                torch.load(audio_ckpt, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Audio classifier checkpoint missing: {audio_ckpt}")

        if os.path.exists(fusion_ckpt):
            self.fusion_model.load_state_dict(
                torch.load(fusion_ckpt, map_location=self.device, weights_only=True)
            )
        else:
            raise FileNotFoundError(f"Fusion classifier checkpoint missing: {fusion_ckpt}")

        # Set to eval mode
        self.video_model.eval()
        self.audio_model.eval()
        self.fusion_model.eval()

    def predict_tensors(
        self,
        frames: torch.Tensor,
        mel: torch.Tensor
    ) -> dict:
        """
        Runs inference on pre-extracted video frames and mel spectrogram tensors.
        """
        frames = frames.to(self.device)
        mel = mel.to(self.device)

        with torch.no_grad():
            v_logits = self.video_model(frames)
            a_logits = self.audio_model(mel)
            f_logits = self.fusion_model(frames, mel)

            v_probs = F.softmax(v_logits, dim=1)[0]
            a_probs = F.softmax(a_logits, dim=1)[0]
            f_probs = F.softmax(f_logits, dim=1)[0]

            v_pred = v_probs.argmax().item()
            a_pred = a_probs.argmax().item()
            f_pred = f_probs.argmax().item()

        return {
            "video_only": {
                "prediction": LABEL_MAPPING[v_pred],
                "confidence": round(float(v_probs[v_pred]) * 100, 2),
                "class_id": v_pred
            },
            "audio_only": {
                "prediction": LABEL_MAPPING[a_pred],
                "confidence": round(float(a_probs[a_pred]) * 100, 2),
                "class_id": a_pred
            },
            "fusion": {
                "prediction": LABEL_MAPPING[f_pred],
                "confidence": round(float(f_probs[f_pred]) * 100, 2),
                "class_id": f_pred
            }
        }

    def predict_video(self, video_path: str | Path) -> dict:
        """
        Full end-to-end inference from an MP4 video file path.
        """
        frames, mel = preprocess_video_and_audio(video_path)
        return self.predict_tensors(frames, mel)
