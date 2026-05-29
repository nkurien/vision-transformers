"""Fine-tune ViT-Base on Stanford Dogs using pre-trained ImageNet weights.

Usage:
    python scripts/train_dogs.py \\
        --config configs/vit_base.yaml \\
        --pretrained weights/vit_base_patch16_224_pretrained.npy

    # Smoke test with a small slice of data:
    python scripts/train_dogs.py \\
        --config configs/vit_base.yaml \\
        --pretrained weights/vit_base_patch16_224_pretrained.npy \\
        --epochs 2 --batch-size 8 --max-per-class 10
"""

import argparse
import os
import sys

import numpy as np
import yaml
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import augment_batch, load_dogs
from train import load_weights, train
from vit import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on Stanford Dogs")
    parser.add_argument("--config",        required=True,  help="Path to YAML config")
    parser.add_argument("--pretrained",    required=True,  help="Path to pretrained .npy weights")
    parser.add_argument("--data-dir",      default="data/dogs/Images")
    parser.add_argument("--epochs",        type=int,   default=None, help="Override config epochs")
    parser.add_argument("--batch-size",    type=int,   default=None, help="Override config batch_size")
    parser.add_argument("--max-per-class", type=int,   default=None,
                        help="Cap images per class (useful for smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    epochs     = args.epochs     if args.epochs     is not None else tcfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else tcfg["batch_size"]

    print("Loading Stanford Dogs...")
    X_train, y_train, X_val, y_val, class_names = load_dogs(
        data_dir=args.data_dir,
        img_size=mcfg["img_size"],
        max_per_class=args.max_per_class,
    )
    num_classes = len(class_names)
    print(f"  {num_classes} classes | train: {len(X_train)}  val: {len(X_val)}")

    model = VisionTransformer(
        img_size=mcfg["img_size"],
        patch_size=mcfg["patch_size"],
        embed_dim=mcfg["embed_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        mlp_dim=mcfg["mlp_dim"],
        num_classes=num_classes,
        dropout=mcfg["dropout"],
    )

    print(f"Loading pretrained weights from {args.pretrained} (head skipped)...")
    load_weights(model, args.pretrained, skip_head=True)

    freeze_until = tcfg["freeze_until"]
    model.freeze_layers(until=freeze_until)
    n_frozen = len(model.encoder.blocks) - abs(freeze_until)
    print(f"Frozen {n_frozen}/{len(model.encoder.blocks)} transformer blocks")

    print(f"Training for {epochs} epochs  |  batch {batch_size}  |  lr {tcfg['lr']:.1e}")
    history = train(
        model,
        X_train, y_train,
        X_val,   y_val,
        epochs=epochs,
        lr=tcfg["lr"],
        batch_size=batch_size,
        warmup_epochs=tcfg["warmup_epochs"],
        weight_decay=tcfg["weight_decay"],
        checkpoint_prefix="dogs_checkpoint",
        augment_fn=partial(augment_batch, crop_size=mcfg["img_size"], pad=16),
    )

    best_val_acc = max(history["val_acc"])
    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    os.makedirs("weights", exist_ok=True)
    np.save("weights/dogs_history.npy",
            {"history": history, "class_names": class_names},
            allow_pickle=True)
    print("History saved to weights/dogs_history.npy")


if __name__ == "__main__":
    main()
