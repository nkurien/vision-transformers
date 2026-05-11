import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import MultiHeadAttention


FORWARD_CONFIGS = [
    (2, 36, 192, 3),   # ViT-Small patch sequence
    (1, 16,  64, 4),   # small square sequence
    (4,  8,  32, 2),   # large batch, short sequence
]

GRAD_CONFIGS = [
    (1, 4,  8, 2),
    (1, 4, 16, 4),
]


@pytest.mark.parametrize("B,N,D,h", FORWARD_CONFIGS)
def test_attention_shape(B, N, D, h):
    # Arrange
    rng = np.random.default_rng(0)
    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    X = rng.standard_normal((B, N, D))

    # Act
    output, attn_weights = mha.forward(X, X, X)

    # Assert
    assert output.shape == (B, N, D)
    assert attn_weights.shape == (B, h, N, N)


@pytest.mark.parametrize("B,N,D,h", FORWARD_CONFIGS)
def test_attention_weights_normalised(B, N, D, h):
    # Arrange
    rng = np.random.default_rng(1)
    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    X = rng.standard_normal((B, N, D))

    # Act
    _, attn_weights = mha.forward(X, X, X)

    # Assert
    sums = attn_weights.sum(axis=-1)  # (B, h, N)
    assert np.allclose(sums, np.ones_like(sums))


@pytest.mark.slow
@pytest.mark.parametrize("B,N,D,h", GRAD_CONFIGS)
def test_attention_gradients(B, N, D, h):
    """Numerical gradient check for W_q using finite differences (~30s)."""
    # Arrange
    rng = np.random.default_rng(2)
    eps = 1e-5
    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    Q = rng.standard_normal((B, N, D))
    K = rng.standard_normal((B, N, D))
    V = rng.standard_normal((B, N, D))

    # Act — analytical gradient
    output, _ = mha.forward(Q, K, V)
    weight_grads, _ = mha.backward(np.ones_like(output))
    dW_q_analytical = weight_grads["dW_q"]

    # Act — numerical gradient via finite differences
    dW_q_numerical = np.zeros_like(mha.W_q)
    for i in range(D):
        for j in range(D):
            mha.W_q[i, j] += eps
            loss_pos = mha.forward(Q, K, V)[0].sum()
            mha.W_q[i, j] -= 2 * eps
            loss_neg = mha.forward(Q, K, V)[0].sum()
            mha.W_q[i, j] += eps  # restore
            dW_q_numerical[i, j] = (loss_pos - loss_neg) / (2 * eps)

    # Assert
    abs_diff = np.abs(dW_q_numerical - dW_q_analytical)
    scale = np.abs(dW_q_numerical) + np.abs(dW_q_analytical) + 1e-8
    max_rel_error = (abs_diff / scale).max()
    assert max_rel_error < 1e-4, f"max relative error {max_rel_error:.2e} exceeds 1e-4"
