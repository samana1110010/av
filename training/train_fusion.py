import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultimodalDataset
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.classifiers import FusionClassifier


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, mel, labels, _ in loader:
            frames = frames.to(device)
            mel = mel.to(device)
            labels = labels.to(device)

            logits = model(frames, mel)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


def train_fusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = MultimodalDataset("data/train_split.csv", "data/audio", is_training=True)
    val_dataset = MultimodalDataset("data/val_split.csv", "data/audio", is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Encoders
    video_encoder = VideoEncoder()
    audio_encoder = AudioEncoder()

    if args.use_retrieval_ckpt:
        if os.path.exists("checkpoints/video_encoder_final.pt"):
            print("Loading video encoder retrieval weights from checkpoints/video_encoder_final.pt...")
            video_encoder.load_state_dict(torch.load("checkpoints/video_encoder_final.pt", map_location=device, weights_only=True))

        if os.path.exists("checkpoints/audio_encoder_final.pt"):
            print("Loading audio encoder retrieval weights from checkpoints/audio_encoder_final.pt...")
            audio_encoder.load_state_dict(torch.load("checkpoints/audio_encoder_final.pt", map_location=device, weights_only=True))

    model = FusionClassifier(video_encoder=video_encoder, audio_encoder=audio_encoder, num_classes=10, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    # Freeze entire encoders (backbones + unused projectors)
    for param in model.video_encoder.parameters():
        param.requires_grad = False
    for param in model.audio_encoder.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Trainable Parameters (Fusion Head): {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/fusion_classifier_best.pt"
    metrics_path = "checkpoints/fusion_classifier_metrics.json"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"\nStarting Fusion (Audio + Video) Classifier Training (hidden_dim={args.hidden_dim}, dropout={args.dropout}, lr={args.lr}, weight_decay={args.weight_decay})...")

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            print(f"\n--> Epoch {epoch}: Unfreezing layer4 of both backbones for fine-tuning!")
            for name, param in model.video_encoder.backbone.named_parameters():
                if "7." in name or name.startswith("7"):
                    param.requires_grad = True
            for name, param in model.audio_encoder.backbone.named_parameters():
                if "7." in name or name.startswith("7"):
                    param.requires_grad = True

            optimizer = torch.optim.AdamW([
                {"params": filter(lambda p: p.requires_grad, model.video_encoder.backbone.parameters()), "lr": args.lr * 0.1},
                {"params": filter(lambda p: p.requires_grad, model.audio_encoder.backbone.parameters()), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr}
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for frames, mel, labels, _ in train_loader:
            frames = frames.to(device)
            mel = mel.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(frames, mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()

        train_loss = train_loss / total
        train_acc = correct / total * 100.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model with Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\nEarly stopping triggered! No improvement for {args.patience} epochs.")
                break

    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%. Model saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio + Video Fusion Classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--unfreeze_epoch", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--use_retrieval_ckpt", action="store_true", default=False)

    args = parser.parse_args()
    train_fusion(args)
