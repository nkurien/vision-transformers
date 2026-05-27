"""Smoke tests for the Dogs training pipeline.

Uses a tiny synthetic dataset and a temp YAML config — no real data or
pretrained weights required.
"""

import numpy as np
import pytest
import os
import sys
import tempfile

import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from vit import VisionTransformer
from train import load_weights, train, _save_checkpoint


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = dict(
    img_size=32, patch_size=8, embed_dim=16,
    num_layers=2, num_heads=2, mlp_dim=32,
    num_classes=3, dropout=0.0,
)

TRAINING_CFG = dict(
    epochs=1, batch_size=2, lr=1e-3,
    warmup_epochs=1, weight_decay=0.01, freeze_until=-1,
)


def _make_dogs_dir(root, classes, n_per_class=4, img_size=32, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            pixels = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(os.path.join(cls_dir, f"{i}.jpg"))


def _make_config(tmp_path, overrides=None):
    cfg = {
        "model": {
            "img_size":    TINY_CFG["img_size"],
            "patch_size":  TINY_CFG["patch_size"],
            "embed_dim":   TINY_CFG["embed_dim"],
            "num_layers":  TINY_CFG["num_layers"],
            "num_heads":   TINY_CFG["num_heads"],
            "mlp_dim":     TINY_CFG["mlp_dim"],
            "dropout":     TINY_CFG["dropout"],
        },
        "training": dict(TRAINING_CFG),
    }
    if overrides:
        for k, v in overrides.items():
            section, key = k.split(".")
            cfg[section][key] = v
    path = str(tmp_path / "config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path, cfg


def _make_pretrained(tmp_path, model):
    """Save current model weights as a fake pretrained checkpoint."""
    path = str(tmp_path / "pretrained.npy")
    _save_checkpoint.__wrapped__ = None
    state = {
        "head_W": model.head_W, "head_b": model.head_b,
        "patch_embed": model.patch_embed,
        "cls_token": model.cls_token, "pos_embed": model.pos_embed,
    }
    for i, block in enumerate(model.encoder.blocks):
        state.update({
            f"b{i}_norm1_gamma": block.norm1.gamma, f"b{i}_norm1_beta": block.norm1.beta,
            f"b{i}_W_q": block.attn.W_q, f"b{i}_W_k": block.attn.W_k,
            f"b{i}_W_v": block.attn.W_v, f"b{i}_W_o": block.attn.W_o,
            f"b{i}_norm2_gamma": block.norm2.gamma, f"b{i}_norm2_beta": block.norm2.beta,
            f"b{i}_W1": block.W1, f"b{i}_b1": block.b1,
            f"b{i}_W2": block.W2, f"b{i}_b2": block.b2,
        })
    np.save(path, state, allow_pickle=True)
    return path


# ---------------------------------------------------------------------------
# load_weights skip_head
# ---------------------------------------------------------------------------

SKIP_HEAD_CONFIGS = [
    (3, 5),   # pretrained on 3 classes, fine-tune on 5
    (10, 2),  # pretrained on 10, fine-tune on 2
]

@pytest.mark.parametrize("pretrained_classes,finetune_classes", SKIP_HEAD_CONFIGS)
def test_skip_head_leaves_head_unchanged(tmp_path, pretrained_classes, finetune_classes):
    # Arrange — save weights from a model with pretrained_classes
    src = VisionTransformer(**{**TINY_CFG, "num_classes": pretrained_classes})
    pretrained_path = _make_pretrained(tmp_path, src)

    dst = VisionTransformer(**{**TINY_CFG, "num_classes": finetune_classes})
    original_head_W = dst.head_W.copy()
    original_head_b = dst.head_b.copy()

    # Act
    load_weights(dst, pretrained_path, skip_head=True)

    # Assert — head unchanged, backbone loaded
    np.testing.assert_array_equal(dst.head_W, original_head_W)
    np.testing.assert_array_equal(dst.head_b, original_head_b)
    np.testing.assert_array_equal(dst.patch_embed, src.patch_embed)


@pytest.mark.parametrize("pretrained_classes,finetune_classes", SKIP_HEAD_CONFIGS)
def test_skip_head_false_loads_head(tmp_path, pretrained_classes, finetune_classes):
    # Arrange — same class count so shapes match
    src = VisionTransformer(**{**TINY_CFG, "num_classes": pretrained_classes})
    dst = VisionTransformer(**{**TINY_CFG, "num_classes": pretrained_classes})
    pretrained_path = _make_pretrained(tmp_path, src)

    # Act
    load_weights(dst, pretrained_path, skip_head=False)

    # Assert — head is loaded
    np.testing.assert_array_equal(dst.head_W, src.head_W)
    np.testing.assert_array_equal(dst.head_b, src.head_b)


# ---------------------------------------------------------------------------
# checkpoint_prefix
# ---------------------------------------------------------------------------

PREFIX_CONFIGS = [
    ("checkpoint",      "checkpoint_epoch_1.npy"),
    ("dogs_checkpoint", "dogs_checkpoint_epoch_1.npy"),
    ("mnist",           "mnist_epoch_1.npy"),
]

@pytest.mark.parametrize("prefix,expected_filename", PREFIX_CONFIGS)
def test_checkpoint_prefix(tmp_path, monkeypatch, prefix, expected_filename):
    # Arrange
    monkeypatch.chdir(tmp_path)
    os.makedirs("weights", exist_ok=True)
    model = VisionTransformer(**TINY_CFG)

    # Act
    _save_checkpoint(model, epoch=1, prefix=prefix)

    # Assert
    assert os.path.exists(f"weights/{expected_filename}")


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

def test_train_dogs_script_runs(tmp_path, monkeypatch):
    """Full pipeline: config → data → model → load weights → freeze → train."""
    monkeypatch.chdir(tmp_path)

    # Arrange — synthetic dataset and config
    classes = ["breed_a", "breed_b", "breed_c"]
    data_dir = str(tmp_path / "Images")
    _make_dogs_dir(data_dir, classes, n_per_class=4, img_size=TINY_CFG["img_size"])
    config_path, cfg = _make_config(tmp_path)

    src_model = VisionTransformer(**{**TINY_CFG, "num_classes": 5})
    pretrained_path = _make_pretrained(tmp_path, src_model)

    # Act — replicate what train_dogs.py main() does
    import yaml
    from preprocessing import load_dogs

    with open(config_path) as f:
        loaded_cfg = yaml.safe_load(f)

    mcfg = loaded_cfg["model"]
    tcfg = loaded_cfg["training"]

    X_train, y_train, X_val, y_val, class_names = load_dogs(
        data_dir=data_dir, img_size=mcfg["img_size"]
    )

    model = VisionTransformer(
        img_size=mcfg["img_size"], patch_size=mcfg["patch_size"],
        embed_dim=mcfg["embed_dim"], num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"], mlp_dim=mcfg["mlp_dim"],
        num_classes=len(class_names), dropout=mcfg["dropout"],
    )
    load_weights(model, pretrained_path, skip_head=True)
    model.freeze_layers(until=tcfg["freeze_until"])

    history = train(
        model, X_train, y_train, X_val, y_val,
        epochs=tcfg["epochs"], lr=tcfg["lr"],
        batch_size=tcfg["batch_size"],
        warmup_epochs=tcfg["warmup_epochs"],
        checkpoint_prefix="dogs_checkpoint",
    )

    # Assert
    assert set(history.keys()) == {"train_loss", "train_acc", "val_loss", "val_acc", "lr"}
    assert len(history["train_loss"]) == tcfg["epochs"]
    assert os.path.exists("weights/dogs_checkpoint_epoch_1.npy")
