"""Train ViT-Small on MNIST and save training history."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import load_mnist
from train import train
from vit import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-Small on MNIST")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading MNIST...")
    X_train, y_train, X_val, y_val = load_mnist()
    print(f"  train: {X_train.shape}  val: {X_val.shape}")

    model = VisionTransformer(
        img_size=96,
        patch_size=16,
        in_channels=1,
        embed_dim=192,
        num_layers=6,
        num_heads=3,
        mlp_dim=768,
        num_classes=10,
        dropout=0.1,
    )

    history = train(
        model,
        X_train, y_train,
        X_val,   y_val,
        epochs=args.epochs,
        lr=1e-3,
        batch_size=args.batch_size,
        warmup_epochs=5,
    )

    final_val_acc = history["val_acc"][-1]
    best_val_acc  = max(history["val_acc"])
    print(f"\nFinal val accuracy: {final_val_acc:.4f}")
    print(f"Best  val accuracy: {best_val_acc:.4f}")

    os.makedirs("weights", exist_ok=True)
    np.save("weights/mnist_history.npy", history, allow_pickle=True)
    print("History saved to weights/mnist_history.npy")


if __name__ == "__main__":
    main()
