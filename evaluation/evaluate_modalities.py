import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

from datasets.multimodal_dataset import MultimodalDataset
from models.classifiers import VideoClassifier, AudioClassifier, FusionClassifier


def evaluate_all(test_csv="data/unseen_test/test.csv", audio_dir="data/unseen_test/audio"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")
    print(f"Loading test set from: {test_csv}")

    test_dataset = MultimodalDataset(test_csv, audio_dir, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Class names mapping
    df = pd.read_csv(test_csv)
    label_to_name = df[["label", "class_name"]].drop_duplicates().sort_values("label").set_index("label")["class_name"].to_dict()
    class_names = [label_to_name[i] for i in range(len(label_to_name))]

    # Load Models
    video_model = VideoClassifier().to(device)
    audio_model = AudioClassifier().to(device)
    fusion_model = FusionClassifier().to(device)

    video_ckpt = "checkpoints/video_classifier_best.pt"
    audio_ckpt = "checkpoints/audio_classifier_best.pt"
    fusion_ckpt = "checkpoints/fusion_classifier_best.pt"

    if os.path.exists(video_ckpt):
        video_model.load_state_dict(torch.load(video_ckpt, map_location=device, weights_only=True))
        print(f"Loaded Video Classifier from {video_ckpt}")
    else:
        print(f"Warning: {video_ckpt} not found!")

    if os.path.exists(audio_ckpt):
        audio_model.load_state_dict(torch.load(audio_ckpt, map_location=device, weights_only=True))
        print(f"Loaded Audio Classifier from {audio_ckpt}")
    else:
        print(f"Warning: {audio_ckpt} not found!")

    if os.path.exists(fusion_ckpt):
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device, weights_only=True))
        print(f"Loaded Fusion Classifier from {fusion_ckpt}")
    else:
        print(f"Warning: {fusion_ckpt} not found!")

    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    all_labels = []
    all_vids = []
    video_preds, video_probs = [], []
    audio_preds, audio_probs = [], []
    fusion_preds, fusion_probs = [], []

    with torch.no_grad():
        for frames, mel, labels, vids in test_loader:
            frames = frames.to(device)
            mel = mel.to(device)
            all_labels.extend(labels.numpy())
            all_vids.extend(vids)

            v_logits = video_model(frames)
            a_logits = audio_model(mel)
            f_logits = fusion_model(frames, mel)

            video_preds.extend(v_logits.argmax(dim=1).cpu().numpy())
            audio_preds.extend(a_logits.argmax(dim=1).cpu().numpy())
            fusion_preds.extend(f_logits.argmax(dim=1).cpu().numpy())

            video_probs.extend(torch.softmax(v_logits, dim=1).cpu().numpy())
            audio_probs.extend(torch.softmax(a_logits, dim=1).cpu().numpy())
            fusion_probs.extend(torch.softmax(f_logits, dim=1).cpu().numpy())

    all_labels = np.array(all_labels)

    def calc_metrics(preds):
        acc = accuracy_score(all_labels, preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, preds, labels=list(range(len(class_names))))
        return acc, prec * 100.0, rec * 100.0, f1 * 100.0, cm

    v_acc, v_prec, v_rec, v_f1, v_cm = calc_metrics(video_preds)
    a_acc, a_prec, a_rec, a_f1, a_cm = calc_metrics(audio_preds)
    f_acc, f_prec, f_rec, f_f1, f_cm = calc_metrics(fusion_preds)

    print("\n========================================")
    print("MULTIMODAL CLASSIFICATION RESULTS")
    print("========================================")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 72)
    print(f"{'Video-only':<20} | {v_acc:6.2f}%    | {v_prec:6.2f}%    | {v_rec:6.2f}%    | {v_f1:6.2f}%")
    print(f"{'Audio-only':<20} | {a_acc:6.2f}%    | {a_prec:6.2f}%    | {a_rec:6.2f}%    | {a_f1:6.2f}%")
    print(f"{'Fusion (Audio+Video)':<20} | {f_acc:6.2f}%    | {f_prec:6.2f}%    | {f_rec:6.2f}%    | {f_f1:6.2f}%")

    print("\n--- RESEARCH COMPARISON SUMMARY ---")
    if f_acc > v_acc and f_acc > a_acc:
        print(f"SUCCESS: Combining Audio and Video information (Fusion Acc: {f_acc:.2f}%) outperforms both Video-only ({v_acc:.2f}%) and Audio-only ({a_acc:.2f}%).")
    else:
        print(f"OBSERVATION: Fusion Acc ({f_acc:.2f}%) vs Video-only ({v_acc:.2f}%) vs Audio-only ({a_acc:.2f}%).")

    print("\n--- CONFUSION MATRICES ---")
    print("\nVideo-Only Confusion Matrix:")
    print(v_cm)
    print("\nAudio-Only Confusion Matrix:")
    print(a_cm)
    print("\nFusion Confusion Matrix:")
    print(f_cm)

    # Save results to json and csv
    results_dict = {
        "video_only": {"accuracy": v_acc, "precision": v_prec, "recall": v_rec, "f1": v_f1, "confusion_matrix": v_cm.tolist()},
        "audio_only": {"accuracy": a_acc, "precision": a_prec, "recall": a_rec, "f1": a_f1, "confusion_matrix": a_cm.tolist()},
        "fusion": {"accuracy": f_acc, "precision": f_prec, "recall": f_rec, "f1": f_f1, "confusion_matrix": f_cm.tolist()},
        "class_names": class_names
    }

    os.makedirs("evaluation", exist_ok=True)
    json_path = "evaluation/results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nSaved detailed evaluation metrics to {json_path}")

    csv_data = [
        {"Model": "Video-only", "Accuracy": v_acc, "Precision": v_prec, "Recall": v_rec, "F1": v_f1},
        {"Model": "Audio-only", "Accuracy": a_acc, "Precision": a_prec, "Recall": a_rec, "F1": a_f1},
        {"Model": "Fusion", "Accuracy": f_acc, "Precision": f_prec, "Recall": f_rec, "F1": f_f1}
    ]
    df_results = pd.DataFrame(csv_data)
    csv_path = "evaluation/results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved summary metrics table to {csv_path}")

    # Per-sample predictions saving for failure analysis
    sample_records = []
    for i in range(len(all_vids)):
        sample_records.append({
            "video_id": all_vids[i],
            "ground_truth_label": int(all_labels[i]),
            "ground_truth_class": class_names[all_labels[i]],
            "video_pred_class": class_names[video_preds[i]],
            "video_pred_conf": round(float(video_probs[i][video_preds[i]]) * 100, 2),
            "video_correct": bool(video_preds[i] == all_labels[i]),
            "audio_pred_class": class_names[audio_preds[i]],
            "audio_pred_conf": round(float(audio_probs[i][audio_preds[i]]) * 100, 2),
            "audio_correct": bool(audio_preds[i] == all_labels[i]),
            "fusion_pred_class": class_names[fusion_preds[i]],
            "fusion_pred_conf": round(float(fusion_probs[i][fusion_preds[i]]) * 100, 2),
            "fusion_correct": bool(fusion_preds[i] == all_labels[i]),
        })

    with open("evaluation/unseen_predictions.json", "w") as f:
        json.dump(sample_records, f, indent=2)

    pd.DataFrame(sample_records).to_csv("evaluation/unseen_predictions.csv", index=False)
    print("Saved individual sample predictions to evaluation/unseen_predictions.json and evaluation/unseen_predictions.csv")

    return results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/unseen_test/test.csv")
    parser.add_argument("--audio_dir", type=str, default="data/unseen_test/audio")
    args = parser.parse_args()

    evaluate_all(args.test_csv, args.audio_dir)
