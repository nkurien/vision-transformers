"""Round-trip tests for load_weights / _save_checkpoint."""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vit import VisionTransformer
from train import load_weights


VIT_CONFIGS = [
    dict(img_size=32, patch_size=8,  embed_dim=16, num_layers=2, num_heads=2, mlp_dim=32,  num_classes=5),
    dict(img_size=48, patch_size=16, embed_dim=32, num_layers=3, num_heads=4, mlp_dim=64,  num_classes=10),
]


def _make_model(cfg):
    return VisionTransformer(**cfg)


def _randomise(model, rng):
    """Fill every weight with distinct random values so default-zero init can't mask errors."""
    model.patch_embed[:] = rng.standard_normal(model.patch_embed.shape)
    model.cls_token[:]   = rng.standard_normal(model.cls_token.shape)
    model.pos_embed[:]   = rng.standard_normal(model.pos_embed.shape)
    model.head_W[:]      = rng.standard_normal(model.head_W.shape)
    model.head_b[:]      = rng.standard_normal(model.head_b.shape)
    model.norm.gamma[:]  = rng.standard_normal(model.norm.gamma.shape)
    model.norm.beta[:]   = rng.standard_normal(model.norm.beta.shape)
    for block in model.encoder.blocks:
        block.norm1.gamma[:] = rng.standard_normal(block.norm1.gamma.shape)
        block.norm1.beta[:]  = rng.standard_normal(block.norm1.beta.shape)
        block.attn.W_q[:]    = rng.standard_normal(block.attn.W_q.shape)
        block.attn.b_q[:]    = rng.standard_normal(block.attn.b_q.shape)
        block.attn.W_k[:]    = rng.standard_normal(block.attn.W_k.shape)
        block.attn.b_k[:]    = rng.standard_normal(block.attn.b_k.shape)
        block.attn.W_v[:]    = rng.standard_normal(block.attn.W_v.shape)
        block.attn.b_v[:]    = rng.standard_normal(block.attn.b_v.shape)
        block.attn.W_o[:]    = rng.standard_normal(block.attn.W_o.shape)
        block.attn.b_o[:]    = rng.standard_normal(block.attn.b_o.shape)
        block.norm2.gamma[:] = rng.standard_normal(block.norm2.gamma.shape)
        block.norm2.beta[:]  = rng.standard_normal(block.norm2.beta.shape)
        block.W1[:]          = rng.standard_normal(block.W1.shape)
        block.b1[:]          = rng.standard_normal(block.b1.shape)
        block.W2[:]          = rng.standard_normal(block.W2.shape)
        block.b2[:]          = rng.standard_normal(block.b2.shape)


def _state_dict(model):
    state = {
        "head_W":     model.head_W,  "head_b":     model.head_b,
        "norm_gamma": model.norm.gamma, "norm_beta": model.norm.beta,
        "patch_embed": model.patch_embed,
        "cls_token":  model.cls_token, "pos_embed": model.pos_embed,
    }
    for i, block in enumerate(model.encoder.blocks):
        state.update({
            f"b{i}_norm1_gamma": block.norm1.gamma, f"b{i}_norm1_beta": block.norm1.beta,
            f"b{i}_W_q": block.attn.W_q, f"b{i}_b_q": block.attn.b_q,
            f"b{i}_W_k": block.attn.W_k, f"b{i}_b_k": block.attn.b_k,
            f"b{i}_W_v": block.attn.W_v, f"b{i}_b_v": block.attn.b_v,
            f"b{i}_W_o": block.attn.W_o, f"b{i}_b_o": block.attn.b_o,
            f"b{i}_norm2_gamma": block.norm2.gamma, f"b{i}_norm2_beta": block.norm2.beta,
            f"b{i}_W1": block.W1, f"b{i}_b1": block.b1,
            f"b{i}_W2": block.W2, f"b{i}_b2": block.b2,
        })
    return state


# ---------------------------------------------------------------------------
# load_weights round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", VIT_CONFIGS)
def test_load_weights_round_trip(cfg, tmp_path):
    # Arrange
    rng = np.random.default_rng(7)
    src = _make_model(cfg)
    _randomise(src, rng)
    dst = _make_model(cfg)

    path = str(tmp_path / "ckpt.npy")
    np.save(path, _state_dict(src), allow_pickle=True)

    # Act
    load_weights(dst, path)

    # Assert
    np.testing.assert_array_equal(dst.patch_embed,  src.patch_embed)
    np.testing.assert_array_equal(dst.cls_token,    src.cls_token)
    np.testing.assert_array_equal(dst.pos_embed,    src.pos_embed)
    np.testing.assert_array_equal(dst.head_W,       src.head_W)
    np.testing.assert_array_equal(dst.head_b,       src.head_b)
    np.testing.assert_array_equal(dst.norm.gamma,   src.norm.gamma)
    np.testing.assert_array_equal(dst.norm.beta,    src.norm.beta)
    for i, (sb, db) in enumerate(zip(src.encoder.blocks, dst.encoder.blocks)):
        np.testing.assert_array_equal(db.norm1.gamma, sb.norm1.gamma, err_msg=f"block {i} norm1.gamma")
        np.testing.assert_array_equal(db.attn.W_q,    sb.attn.W_q,    err_msg=f"block {i} W_q")
        np.testing.assert_array_equal(db.attn.b_q,    sb.attn.b_q,    err_msg=f"block {i} b_q")
        np.testing.assert_array_equal(db.attn.b_o,    sb.attn.b_o,    err_msg=f"block {i} b_o")
        np.testing.assert_array_equal(db.W1,          sb.W1,          err_msg=f"block {i} W1")
        np.testing.assert_array_equal(db.b2,          sb.b2,          err_msg=f"block {i} b2")


@pytest.mark.parametrize("cfg", VIT_CONFIGS)
def test_load_weights_forward_matches(cfg, tmp_path):
    """After loading, both models must produce identical logits on the same input."""
    # Arrange
    rng = np.random.default_rng(42)
    src = _make_model(cfg)
    _randomise(src, rng)
    dst = _make_model(cfg)

    path = str(tmp_path / "ckpt.npy")
    np.save(path, _state_dict(src), allow_pickle=True)

    # Act
    load_weights(dst, path)
    x = rng.standard_normal((2, 3, cfg["img_size"], cfg["img_size"])).astype(np.float32)
    logits_src = src.forward(x, training=False)
    logits_dst = dst.forward(x, training=False)

    # Assert
    np.testing.assert_array_equal(logits_src, logits_dst)
