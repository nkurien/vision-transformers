import gzip
import os
import struct
import urllib.request

import numpy as np


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
