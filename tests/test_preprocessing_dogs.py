"""Tests for load_dogs — uses a synthetic on-disk fixture, no real dataset required."""

import numpy as np
import pytest
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import load_dogs, _IMAGENET_MEAN, _IMAGENET_STD


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_dataset(root, classes, images_per_class, img_h=32, img_w=32, seed=0):
    """Write random JPEG images into root/{class}/{i}.jpg."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(images_per_class):
            pixels = rng.integers(0, 256, (img_h, img_w, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(os.path.join(cls_dir, f"{i}.jpg"))


CONFIGS = [
    (["breed_a", "breed_b"],           5,  0.2),
    (["breed_x", "breed_y", "breed_z"],8,  0.25),
]


# ---------------------------------------------------------------------------
# Basic shape / split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("classes,n_per_class,val_frac", CONFIGS)
def test_output_shapes(tmp_path, classes, n_per_class, val_frac):
    # Arrange
    root = str(tmp_path / "Images")
    _make_dataset(root, classes, n_per_class)
    n_val_per   = max(1, int(n_per_class * val_frac))
    n_train_per = n_per_class - n_val_per

    # Act
    X_train, y_train, X_val, y_val, class_names = load_dogs(
        root, img_size=16, val_fraction=val_frac
    )

    # Assert
    assert X_train.shape == (n_train_per * len(classes), 3, 16, 16)
    assert X_val.shape   == (n_val_per   * len(classes), 3, 16, 16)
    assert y_train.shape == (n_train_per * len(classes),)
    assert y_val.shape   == (n_val_per   * len(classes),)


@pytest.mark.parametrize("classes,n_per_class,val_frac", CONFIGS)
def test_class_names_sorted(tmp_path, classes, n_per_class, val_frac):
    # Arrange
    root = str(tmp_path / "Images")
    _make_dataset(root, classes, n_per_class)

    # Act
    _, _, _, _, class_names = load_dogs(root, img_size=16, val_fraction=val_frac)

    # Assert
    assert class_names == sorted(classes)


@pytest.mark.parametrize("classes,n_per_class,val_frac", CONFIGS)
def test_labels_in_range(tmp_path, classes, n_per_class, val_frac):
    # Arrange
    root = str(tmp_path / "Images")
    _make_dataset(root, classes, n_per_class)

    # Act
    _, y_train, _, y_val, class_names = load_dogs(root, img_size=16, val_fraction=val_frac)

    # Assert
    assert y_train.min() >= 0 and y_train.max() < len(class_names)
    assert y_val.min()   >= 0 and y_val.max()   < len(class_names)


@pytest.mark.parametrize("classes,n_per_class,val_frac", CONFIGS)
def test_dtypes(tmp_path, classes, n_per_class, val_frac):
    # Arrange
    root = str(tmp_path / "Images")
    _make_dataset(root, classes, n_per_class)

    # Act
    X_train, y_train, X_val, y_val, _ = load_dogs(root, img_size=16, val_fraction=val_frac)

    # Assert
    assert X_train.dtype == np.float32
    assert X_val.dtype   == np.float32
    assert y_train.dtype == np.int32
    assert y_val.dtype   == np.int32


# ---------------------------------------------------------------------------
# ImageNet normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("img_size", [16, 32])
def test_imagenet_normalisation(tmp_path, img_size):
    """A solid-colour image (one channel fills entirely) should normalise correctly."""
    # Arrange — single class, one image: solid 128,128,128
    root = str(tmp_path / "Images")
    cls_dir = os.path.join(root, "breed_a")
    os.makedirs(cls_dir, exist_ok=True)
    Image.fromarray(
        np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    ).save(os.path.join(cls_dir, "0.png"))

    # Act
    X_train, _, X_val, _, _ = load_dogs(root, img_size=img_size, val_fraction=0.0)
    # val_fraction=0 → all in train
    x = X_train[0]   # (3, H, W)

    # Assert — each channel should be (128/255 - mean) / std
    for c in range(3):
        expected = (128.0 / 255.0 - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        np.testing.assert_allclose(x[c], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# max_per_class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_per_class", [1, 2, 3])
def test_max_per_class_caps_samples(tmp_path, max_per_class):
    # Arrange — 3 classes, 10 images each
    classes = ["a", "b", "c"]
    root = str(tmp_path / "Images")
    _make_dataset(root, classes, images_per_class=10)

    # Act
    X_train, _, X_val, _, _ = load_dogs(
        root, img_size=8, val_fraction=0.2, max_per_class=max_per_class
    )

    # Assert — total samples <= max_per_class * num_classes
    assert len(X_train) + len(X_val) <= max_per_class * len(classes)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_data_dir_raises(tmp_path):
    # Arrange
    bad_path = str(tmp_path / "does_not_exist")

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_dogs(bad_path)
