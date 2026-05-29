"""Tests for load_dogs — uses a synthetic on-disk fixture, no real dataset required."""

import numpy as np
import pytest
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import augment_batch, center_crop, load_dogs, _IMAGENET_MEAN, _IMAGENET_STD


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


# ---------------------------------------------------------------------------
# center_crop
# ---------------------------------------------------------------------------

CROP_CONFIGS = [
    (2, 3,  8,  8, 6),
    (1, 1, 10, 12, 8),
    (4, 3, 16, 16, 14),
]


@pytest.mark.parametrize("B,C,H,W,crop_size", CROP_CONFIGS)
def test_center_crop_shape(B, C, H, W, crop_size):
    # Arrange
    x = np.random.rand(B, C, H, W).astype(np.float32)

    # Act
    out = center_crop(x, crop_size)

    # Assert
    assert out.shape == (B, C, crop_size, crop_size)


@pytest.mark.parametrize("B,C,H,W,crop_size", CROP_CONFIGS)
def test_center_crop_values(B, C, H, W, crop_size):
    # Arrange
    rng = np.random.default_rng(0)
    x = rng.random((B, C, H, W)).astype(np.float32)
    top  = (H - crop_size) // 2
    left = (W - crop_size) // 2

    # Act
    out = center_crop(x, crop_size)

    # Assert
    np.testing.assert_array_equal(out, x[:, :, top:top + crop_size, left:left + crop_size])


# ---------------------------------------------------------------------------
# augment_batch
# ---------------------------------------------------------------------------

AUGMENT_CONFIGS = [
    (4, 3,  8,  8, 6),
    (2, 1, 10, 10, 8),
    (1, 3, 16, 16, 14),
]


@pytest.mark.parametrize("B,C,H,W,crop_size", AUGMENT_CONFIGS)
def test_augment_batch_shape(B, C, H, W, crop_size):
    # Arrange
    x = np.random.rand(B, C, H, W).astype(np.float32)

    # Act
    out = augment_batch(x, crop_size)

    # Assert
    assert out.shape == (B, C, crop_size, crop_size)


@pytest.mark.parametrize("B,C,H,W,crop_size", AUGMENT_CONFIGS)
def test_augment_batch_dtype_preserved(B, C, H, W, crop_size):
    # Arrange
    x = np.random.rand(B, C, H, W).astype(np.float32)

    # Act
    out = augment_batch(x, crop_size)

    # Assert
    assert out.dtype == x.dtype


def test_augment_batch_output_is_valid_transform():
    """Each output must equal either the input crop or its horizontal flip."""
    # Arrange — crop_size == H == W eliminates crop-position randomness (randint(0,1)==0)
    B, C, size = 3, 2, 6
    x = np.arange(B * C * size * size, dtype=np.float32).reshape(B, C, size, size)

    for seed in range(20):
        np.random.seed(seed)

        # Act
        out = augment_batch(x, crop_size=size)

        # Assert — every image is either the original or h-flipped
        for i in range(B):
            is_normal  = np.allclose(out[i], x[i])
            is_flipped = np.allclose(out[i], x[i, :, :, ::-1])
            assert is_normal or is_flipped, f"seed={seed} image={i}: not a valid transform"


def test_augment_batch_crop_position_varies():
    """Different crop positions must be selected across calls when H > crop_size."""
    # Arrange — unique values so any positional difference shows up in the output
    x = np.arange(16 * 16, dtype=np.float32).reshape(1, 1, 16, 16)

    outputs = []
    for seed in range(30):
        np.random.seed(seed)
        outputs.append(augment_batch(x, crop_size=8)[0].copy())

    # Assert — not all crops are identical (crop position varies)
    assert not all(np.array_equal(outputs[0], o) for o in outputs[1:])


def test_augment_batch_flip_both_outcomes_occur():
    """Horizontal flip must fire sometimes and not fire other times."""
    # Arrange — single image, crop_size == image_size to isolate flip randomness
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    x_flip = x[0, :, :, ::-1]

    saw_normal = saw_flipped = False
    for seed in range(30):
        np.random.seed(seed)
        out = augment_batch(x, crop_size=4)
        if np.allclose(out[0], x[0]):
            saw_normal = True
        elif np.allclose(out[0], x_flip):
            saw_flipped = True
        if saw_normal and saw_flipped:
            break

    # Assert
    assert saw_normal,  "flip never skipped across 30 seeds"
    assert saw_flipped, "flip never fired across 30 seeds"
