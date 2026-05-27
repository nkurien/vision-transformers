import gzip
import os
import struct
import urllib.request

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

_MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = {
    "train-images": "train-images-idx3-ubyte.gz",
    "train-labels": "train-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: str) -> None:
    print(f"Downloading {os.path.basename(dest)} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


def _parse_images(data: bytes) -> np.ndarray:
    """Parse IDX3 image bytes → uint8 array (N, H, W)."""
    magic, n, h, w = struct.unpack_from(">IIII", data, 0)
    if magic != 0x00000803:
        raise ValueError(f"Bad IDX3 magic: {magic:#010x}")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, h, w)


def _parse_labels(data: bytes) -> np.ndarray:
    """Parse IDX1 label bytes → uint8 array (N,)."""
    magic, n = struct.unpack_from(">II", data, 0)
    if magic != 0x00000801:
        raise ValueError(f"Bad IDX1 magic: {magic:#010x}")
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def _resize_nearest(images: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Nearest-neighbour resize (N, H, W) → (N, th, tw) in one vectorised op."""
    _, h, w = images.shape
    row_idx = (np.arange(th) * h / th).astype(np.intp)
    col_idx = (np.arange(tw) * w / tw).astype(np.intp)
    return images[:, row_idx[:, None], col_idx[None, :]]


def load_mnist(
    data_dir: str = "data/mnist",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (and download if necessary) MNIST, resized to 96×96.

    Returns:
        X_train: (50000, 1, 96, 96) float32 in [-1, 1]
        y_train: (50000,)           int32
        X_val:   (10000, 1, 96, 96) float32 in [-1, 1]
        y_val:   (10000,)           int32

    The last 10k samples of the 60k training set are used as validation.
    """
    os.makedirs(data_dir, exist_ok=True)

    paths = {}
    for key, fname in _MNIST_FILES.items():
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest):
            _download(_MNIST_BASE + fname, dest)
        paths[key] = dest

    with gzip.open(paths["train-images"], "rb") as f:
        images = _parse_images(f.read())   # (60000, 28, 28) uint8

    with gzip.open(paths["train-labels"], "rb") as f:
        labels = _parse_labels(f.read())   # (60000,) uint8

    # Normalise to [-1, 1]
    images = images.astype(np.float32) / 127.5 - 1.0

    # Nearest-neighbour resize 28×28 → 96×96
    images = _resize_nearest(images, 96, 96)   # (60000, 96, 96)

    # Add channel dim → (N, 1, 96, 96)
    images = images[:, np.newaxis, :, :]

    # Train / val split — last 10k as validation
    X_train = images[:50_000]
    y_train = labels[:50_000].astype(np.int32)
    X_val   = images[50_000:]
    y_val   = labels[50_000:].astype(np.int32)

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Stanford Dogs
# ---------------------------------------------------------------------------

# ImageNet channel statistics — applied after scaling pixels to [0, 1]
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: str, img_size: int) -> np.ndarray:
    """Load a JPEG/PNG, resize to (img_size, img_size), return (3, H, W) float32 normalised."""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    x   = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3) in [0, 1]
    x   = (x - _IMAGENET_MEAN) / _IMAGENET_STD               # ImageNet normalise
    return x.transpose(2, 0, 1)                               # (3, H, W)


def load_dogs(
    data_dir: str = "data/dogs/Images",
    img_size: int = 224,
    val_fraction: float = 0.2,
    seed: int = 0,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load Stanford Dogs from a folder-per-breed directory tree.

    Expected layout: data_dir/{breed_folder}/{image}.jpg

    Args:
        data_dir:       Root of the Images directory.
        img_size:       Resize target (square). Default 224 for ViT-Base.
        val_fraction:   Fraction of each class held out for validation.
        seed:           RNG seed for the per-class shuffle.
        max_per_class:  Cap images per class. Useful during development to
                        avoid loading all ~11 GB into memory at once.

    Returns:
        X_train: (N_train, 3, img_size, img_size) float32 — ImageNet normalised
        y_train: (N_train,) int32
        X_val:   (N_val,   3, img_size, img_size) float32
        y_val:   (N_val,)   int32
        class_names: sorted list of breed folder names (index == label int)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dogs data not found at '{data_dir}'. "
            "Download from http://vision.stanford.edu/adit/ImageSets/ and extract there."
        )

    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_names:
        raise ValueError(f"No subdirectories found in '{data_dir}'.")

    rng        = np.random.default_rng(seed)
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        paths   = sorted(
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not paths:
            continue

        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]

        n_val  = int(len(paths) * val_fraction)
        val_p  = paths[:n_val]
        train_p = paths[n_val:]

        for p in train_p:
            X_train_list.append(_load_image(p, img_size))
            y_train_list.append(label)

        for p in val_p:
            X_val_list.append(_load_image(p, img_size))
            y_val_list.append(label)

    X_train = np.stack(X_train_list).astype(np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    if X_val_list:
        X_val = np.stack(X_val_list).astype(np.float32)
        y_val = np.array(y_val_list, dtype=np.int32)
    else:
        X_val = np.empty((0, 3, img_size, img_size), dtype=np.float32)
        y_val = np.empty((0,), dtype=np.int32)

    return X_train, y_train, X_val, y_val, class_names


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(x: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract non-overlapping patches from a batch of images.

    Args:
        x:          (N, C, H, W) — H and W must be divisible by patch_size.
        patch_size: Side length of each square patch.

    Returns:
        patches: (N, num_patches, patch_dim)
                 where num_patches = (H/P)*(W/P) and patch_dim = P*P*C,
                 ordered in raster scan (row-major, left-to-right, top-to-bottom).
    """
    N, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H and W must be divisible by patch_size"
    h, w = H // P, W // P
    x = x.reshape(N, C, h, P, w, P)
    x = x.transpose(0, 2, 4, 3, 5, 1)   # (N, h, w, P, P, C)
    return x.reshape(N, h * w, P * P * C)
