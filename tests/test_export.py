"""Tests for export_timm_weights weight-mapping logic.

Uses a tiny synthetic state dict so no timm/torch download is required.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class _FakeTensor:
    """Minimal stand-in for a torch tensor — just wraps a numpy array."""
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def _make_state(D: int = 8, mlp_dim: int = 16, num_patches: int = 4,
                num_classes: int = 3, num_layers: int = 2) -> dict:
    rng = np.random.default_rng(0)
    P   = 2      # patch_size — 2×2×1 = patch_dim 4; not needed directly here
    state = {}

    def t(arr):
        return _FakeTensor(arr)

    state["cls_token"]  = t(rng.standard_normal((1, 1, D)))
    state["pos_embed"]  = t(rng.standard_normal((1, num_patches + 1, D)))
    # patch_embed.proj.weight: timm (D, in_ch, P, P)
    state["patch_embed.proj.weight"] = t(rng.standard_normal((D, 1, 2, 2)))
    state["patch_embed.proj.bias"]   = t(rng.standard_normal((D,)))   # should be skipped
    state["head.weight"] = t(rng.standard_normal((num_classes, D)))
    state["head.bias"]   = t(rng.standard_normal((num_classes,)))
    state["norm.weight"] = t(rng.standard_normal((D,)))               # should be skipped
    state["norm.bias"]   = t(rng.standard_normal((D,)))               # should be skipped

    for i in range(num_layers):
        state[f"blocks.{i}.norm1.weight"]      = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.norm1.bias"]        = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.attn.qkv.weight"]   = t(rng.standard_normal((3 * D, D)))
        state[f"blocks.{i}.attn.qkv.bias"]     = t(rng.standard_normal((3 * D,)))  # skip
        state[f"blocks.{i}.attn.proj.weight"]  = t(rng.standard_normal((D, D)))
        state[f"blocks.{i}.attn.proj.bias"]    = t(rng.standard_normal((D,)))      # skip
        state[f"blocks.{i}.norm2.weight"]      = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.norm2.bias"]        = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.mlp.fc1.weight"]    = t(rng.standard_normal((mlp_dim, D)))
        state[f"blocks.{i}.mlp.fc1.bias"]      = t(rng.standard_normal((mlp_dim,)))
        state[f"blocks.{i}.mlp.fc2.weight"]    = t(rng.standard_normal((D, mlp_dim)))
        state[f"blocks.{i}.mlp.fc2.bias"]      = t(rng.standard_normal((D,)))

    return state


D, MLP, NP, NC, NL = 8, 16, 4, 3, 2
STATE = _make_state(D=D, mlp_dim=MLP, num_patches=NP, num_classes=NC, num_layers=NL)


@pytest.fixture(scope="module")
def mapped():
    from export_timm_weights import _map_weights
    return _map_weights(STATE, num_layers=NL)


# ---------------------------------------------------------------------------
# Keys present / absent
# ---------------------------------------------------------------------------

EXPECTED_KEYS = (
    ["patch_embed", "cls_token", "pos_embed", "head_W", "head_b"]
    + [f"b{i}_{k}" for i in range(NL)
       for k in ("norm1_gamma", "norm1_beta",
                 "W_q", "W_k", "W_v", "W_o",
                 "norm2_gamma", "norm2_beta",
                 "W1", "b1", "W2", "b2")]
)

DROPPED_KEYS = [
    "patch_embed.proj.bias", "norm.weight", "norm.bias",
    "b0_attn_qkv_bias", "b0_attn_proj_bias",
]


@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_expected_key_present(mapped, key):
    assert key in mapped, f"missing key: {key}"


def test_attn_biases_not_exported(mapped):
    # Arrange / Act: mapped fixture already ran
    # Assert
    for key in mapped:
        assert "qkv_bias" not in key and "proj_bias" not in key


def test_final_norm_not_exported(mapped):
    assert "norm_gamma" not in mapped and "norm_beta" not in mapped


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    ("patch_embed",   (4, D)),        # patch_dim=2*2*1=4, embed_dim=D
    ("cls_token",     (1, 1, D)),
    ("pos_embed",     (1, NP + 1, D)),
    ("head_W",        (D, NC)),
    ("head_b",        (NC,)),
    ("b0_W_q",        (D, D)),
    ("b0_W_k",        (D, D)),
    ("b0_W_v",        (D, D)),
    ("b0_W_o",        (D, D)),
    ("b0_W1",         (D, MLP)),
    ("b0_b1",         (MLP,)),
    ("b0_W2",         (MLP, D)),
    ("b0_b2",         (D,)),
    ("b0_norm1_gamma",(D,)),
    ("b0_norm1_beta", (D,)),
]


@pytest.mark.parametrize("key,expected_shape", SHAPE_CASES)
def test_shape(mapped, key, expected_shape):
    # Arrange
    arr = mapped[key]

    # Act / Assert
    assert arr.shape == expected_shape, (
        f"{key}: expected {expected_shape}, got {arr.shape}"
    )


# ---------------------------------------------------------------------------
# Transpose correctness — linear weight matrices must be (in, out)
# ---------------------------------------------------------------------------

TRANSPOSE_CASES = [
    # (our_key, timm_key_prefix, timm_shape)
    ("head_W",  "head.weight",              (NC, D)),
    ("b0_W_q",  "blocks.0.attn.qkv.weight", None),   # derived from QKV split
    ("b0_W1",   "blocks.0.mlp.fc1.weight",  (MLP, D)),
    ("b0_W2",   "blocks.0.mlp.fc2.weight",  (D, MLP)),
    ("b0_W_o",  "blocks.0.attn.proj.weight",(D, D)),
]


@pytest.mark.parametrize("our_key,timm_key,timm_shape", TRANSPOSE_CASES)
def test_weight_is_transposed(mapped, our_key, timm_key, timm_shape):
    # Arrange
    our_w = mapped[our_key]

    if timm_key == "blocks.0.attn.qkv.weight":
        # Reconstruct what W_q should be from the raw QKV weight
        raw = STATE[timm_key].numpy()                     # (3D, D)
        expected = raw.reshape(3, D, D)[0].T              # W_q: (D, D) transposed
    else:
        expected = STATE[timm_key].numpy().T

    # Act / Assert
    np.testing.assert_array_equal(our_w, expected)
