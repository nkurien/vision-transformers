import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import MultiHeadAttention


def test_attention_shape():
    rng = np.random.default_rng(0)
    B, N, D, h = 2, 36, 192, 3
    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    X = rng.standard_normal((B, N, D))
    output, attn_weights = mha.forward(X, X, X)
    assert output.shape == (B, N, D)
    assert attn_weights.shape == (B, h, N, N)


def test_attention_weights_normalised():
    rng = np.random.default_rng(1)
    B, N, D, h = 2, 36, 192, 3
    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    X = rng.standard_normal((B, N, D))
    _, attn_weights = mha.forward(X, X, X)
    sums = attn_weights.sum(axis=-1)  # (B, h, N)
    assert np.allclose(sums, np.ones_like(sums))


@pytest.mark.slow
def test_attention_gradients():
    """Numerical gradient check for W_q using finite differences (~30s)."""
    rng = np.random.default_rng(2)
    B, N, D, h = 1, 4, 8, 2
    eps = 1e-5

    mha = MultiHeadAttention(embed_dim=D, num_heads=h)
    Q = rng.standard_normal((B, N, D))
    K = rng.standard_normal((B, N, D))
    V = rng.standard_normal((B, N, D))

    # Analytical gradient via backward (loss = sum of output)
    output, _ = mha.forward(Q, K, V)
    grad_output = np.ones_like(output)
    weight_grads, _ = mha.backward(grad_output)
    dW_q_analytical = weight_grads["dW_q"]

    # Numerical gradient via finite differences
    dW_q_numerical = np.zeros_like(mha.W_q)
    for i in range(D):
        for j in range(D):
            mha.W_q[i, j] += eps
            loss_pos = mha.forward(Q, K, V)[0].sum()
            mha.W_q[i, j] -= 2 * eps
            loss_neg = mha.forward(Q, K, V)[0].sum()
            mha.W_q[i, j] += eps  # restore
            dW_q_numerical[i, j] = (loss_pos - loss_neg) / (2 * eps)

    # Max relative error
    abs_diff = np.abs(dW_q_numerical - dW_q_analytical)
    scale = np.abs(dW_q_numerical) + np.abs(dW_q_analytical) + 1e-8
    max_rel_error = (abs_diff / scale).max()

    assert max_rel_error < 1e-4, f"max relative error {max_rel_error:.2e} exceeds 1e-4"
