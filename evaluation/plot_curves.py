import os
import json
import matplotlib.pyplot as plt

def plot_learning_curves():
    metrics_files = {
        "Video-only": "checkpoints/video_classifier_metrics.json",
        "Audio-only": "checkpoints/audio_classifier_metrics.json",
        "Fusion": "checkpoints/fusion_classifier_metrics.json"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multimodal Classification Training & Validation Learning Curves", fontsize=16)

    colors = {
        "Video-only": "#2b5c8f",
        "Audio-only": "#d95f02",
        "Fusion": "#2ca02c"
    }

    for label, filepath in metrics_files.items():
        if not os.path.exists(filepath):
            print(f"Metrics file {filepath} not found. Skipping {label}.")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        epochs = range(1, len(data["train_loss"]) + 1)
        c = colors[label]

        # Train Loss
        axes[0, 0].plot(epochs, data["train_loss"], label=label, color=c, linewidth=2)
        # Val Loss
        axes[0, 1].plot(epochs, data["val_loss"], label=label, color=c, linewidth=2, linestyle="--")
        # Train Accuracy
        axes[1, 0].plot(epochs, data["train_acc"], label=label, color=c, linewidth=2)
        # Val Accuracy
        axes[1, 1].plot(epochs, data["val_acc"], label=label, color=c, linewidth=2, linestyle="--")

    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title("Training Accuracy (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_title("Validation Accuracy (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs("evaluation", exist_ok=True)
    out_path = "evaluation/learning_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f"Learning curves plot saved to {out_path}")

if __name__ == "__main__":
    plot_learning_curves()
