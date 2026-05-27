"""Tests for export_timm_weights weight-mapping logic.

Uses a tiny synthetic state dict so no timm/torch download is required.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def _make_state(D: int = 8, mlp_dim: int = 16, num_patches: int = 4,
                num_classes: int = 3, num_layers: int = 2) -> dict:
    rng = np.random.default_rng(0)

    def t(arr):
        return _FakeTensor(arr)

    state = {}
    state["cls_token"]  = t(rng.standard_normal((1, 1, D)))
    state["pos_embed"]  = t(rng.standard_normal((1, num_patches + 1, D)))
    state["patch_embed.proj.weight"] = t(rng.standard_normal((D, 1, 2, 2)))
    state["patch_embed.proj.bias"]   = t(rng.standard_normal((D,)))   # unsupported
    state["head.weight"] = t(rng.standard_normal((num_classes, D)))
    state["head.bias"]   = t(rng.standard_normal((num_classes,)))
    state["norm.weight"] = t(rng.standard_normal((D,)))
    state["norm.bias"]   = t(rng.standard_normal((D,)))

    for i in range(num_layers):
        state[f"blocks.{i}.norm1.weight"]      = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.norm1.bias"]        = t(rng.standard_normal((D,)))
        state[f"blocks.{i}.attn.qkv.weight"]   = t(rng.standard_normal((3 * D, D)))
        state[f"blocks.{i}.attn.qkv.bias"]     = t(rng.standard_normal((3 * D,)))
        state[f"blocks.{i}.attn.proj.weight"]  = t(rng.standard_normal((D, D)))
        state[f"blocks.{i}.attn.proj.bias"]    = t(rng.standard_normal((D,)))
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
    ["patch_embed", "cls_token", "pos_embed", "head_W", "head_b",
     "norm_gamma", "norm_beta"]
    + [f"b{i}_{k}" for i in range(NL)
       for k in ("norm1_gamma", "norm1_beta",
                 "W_q", "b_q", "W_k", "b_k", "W_v", "b_v", "W_o", "b_o",
                 "norm2_gamma", "norm2_beta",
                 "W1", "b1", "W2", "b2")]
)

@pytest.mark.parametrize("key", EXPECTED_KEYS)
def test_expected_key_present(mapped, key):
    # Arrange / Act: mapped fixture
    # Assert
    assert key in mapped, f"missing key: {key}"


def test_patch_embed_bias_not_exported(mapped):
    # patch_embed.proj.bias is the only key we drop
    assert "patch_embed_bias" not in mapped


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------

SHAPE_CASES = [
    ("patch_embed",    (4, D)),        # patch_dim=2*2*1=4
    ("cls_token",      (1, 1, D)),
    ("pos_embed",      (1, NP + 1, D)),
    ("head_W",         (D, NC)),
    ("head_b",         (NC,)),
    ("norm_gamma",     (D,)),
    ("norm_beta",      (D,)),
    ("b0_W_q",         (D, D)),
    ("b0_b_q",         (D,)),
    ("b0_W_k",         (D, D)),
    ("b0_b_k",         (D,)),
    ("b0_W_v",         (D, D)),
    ("b0_b_v",         (D,)),
    ("b0_W_o",         (D, D)),
    ("b0_b_o",         (D,)),
    ("b0_W1",          (D, MLP)),
    ("b0_b1",          (MLP,)),
    ("b0_W2",          (MLP, D)),
    ("b0_b2",          (D,)),
    ("b0_norm1_gamma", (D,)),
    ("b0_norm1_beta",  (D,)),
]

@pytest.mark.parametrize("key,expected_shape", SHAPE_CASES)
def test_shape(mapped, key, expected_shape):
    # Arrange
    arr = mapped[key]
    # Act / Assert
    assert arr.shape == expected_shape, f"{key}: expected {expected_shape}, got {arr.shape}"


# ---------------------------------------------------------------------------
# Transpose correctness — linear weight matrices must be (in, out)
# ---------------------------------------------------------------------------

TRANSPOSE_CASES = [
    ("head_W",  "head.weight",               (NC, D)),
    ("b0_W_q",  "blocks.0.attn.qkv.weight",  None),   # derived from QKV split
    ("b0_W1",   "blocks.0.mlp.fc1.weight",   (MLP, D)),
    ("b0_W2",   "blocks.0.mlp.fc2.weight",   (D, MLP)),
    ("b0_W_o",  "blocks.0.attn.proj.weight", (D, D)),
]

@pytest.mark.parametrize("our_key,timm_key,timm_shape", TRANSPOSE_CASES)
def test_weight_is_transposed(mapped, our_key, timm_key, timm_shape):
    # Arrange
    our_w = mapped[our_key]
    if timm_key == "blocks.0.attn.qkv.weight":
        raw      = STATE[timm_key].numpy()
        expected = raw.reshape(3, D, D)[0].T
    else:
        expected = STATE[timm_key].numpy().T
    # Act / Assert
    np.testing.assert_array_equal(our_w, expected)


# ---------------------------------------------------------------------------
# Bias split correctness — QKV bias (3D,) → b_q, b_k, b_v each (D,)
# ---------------------------------------------------------------------------

BIAS_SPLIT_CASES = [
    ("b0_b_q", 0),
    ("b0_b_k", 1),
    ("b0_b_v", 2),
]

@pytest.mark.parametrize("our_key,idx", BIAS_SPLIT_CASES)
def test_qkv_bias_split(mapped, our_key, idx):
    # Arrange
    raw      = STATE["blocks.0.attn.qkv.bias"].numpy()   # (3D,)
    expected = raw.reshape(3, D)[idx]
    # Act / Assert
    np.testing.assert_array_equal(mapped[our_key], expected)


def test_proj_bias_exported_directly(mapped):
    # Arrange
    expected = STATE["blocks.0.attn.proj.bias"].numpy()
    # Act / Assert
    np.testing.assert_array_equal(mapped["b0_b_o"], expected)


def test_norm_gamma_exported_directly(mapped):
    # Arrange
    expected = STATE["norm.weight"].numpy()
    # Act / Assert
    np.testing.assert_array_equal(mapped["norm_gamma"], expected)
