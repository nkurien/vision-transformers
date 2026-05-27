import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import Adam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_param(w_val, g_val, **kw):
    w = np.atleast_1d(np.array(w_val, dtype=float))
    g = np.atleast_1d(np.array(g_val, dtype=float))
    return Adam([(w, g)], **kw), w, g


# ---------------------------------------------------------------------------
# Shape / basic mechanics
# ---------------------------------------------------------------------------

STEP_CONFIGS = [
    (1e-3, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
    (1e-2, [0.5],            [1.0]),
    (1e-4, [-1.0, 0.0, 1.0], [0.5, 0.5, 0.5]),
]

@pytest.mark.parametrize("lr,w_init,g_init", STEP_CONFIGS)
def test_step_updates_weights_in_place(lr, w_init, g_init):
    # Arrange
    opt, w, _ = _single_param(w_init, g_init, lr=lr, weight_decay=0.0)
    w_before = w.copy()

    # Act
    opt.step(t=1)

    # Assert
    assert not np.allclose(w, w_before)


ZERO_GRAD_CONFIGS = [
    (1e-3, [1.0, -1.0, 2.0]),
    (1e-2, [0.0, 0.0]),
    (1e-4, [5.0]),
]

@pytest.mark.parametrize("lr,w_init", ZERO_GRAD_CONFIGS)
def test_zero_grad_no_update_without_decay(lr, w_init):
    # Arrange
    opt, w, _ = _single_param(w_init, [0.0] * len(w_init), lr=lr, weight_decay=0.0)
    w_before = w.copy()

    # Act
    opt.step(t=1)

    # Assert
    np.testing.assert_array_equal(w, w_before)


# ---------------------------------------------------------------------------
# Decoupled weight decay (AdamW correctness)
# ---------------------------------------------------------------------------

DECAY_CONFIGS = [
    ([1.0, -1.0, 2.0], 0.1),
    ([3.0],             0.01),
    ([-2.0, 4.0],       0.5),
]

@pytest.mark.parametrize("w_init,wd", DECAY_CONFIGS)
def test_weight_decay_shrinks_weights(w_init, wd):
    # Arrange — zero gradient so only decay acts
    opt, w, _ = _single_param(w_init, [0.0] * len(w_init), lr=1e-3, weight_decay=wd)
    w_before = np.abs(w.copy())

    # Act
    opt.step(t=1)

    # Assert
    np.testing.assert_array_less(np.abs(w), w_before + 1e-12)


@pytest.mark.parametrize("wd,lr", [(0.1, 1e-2), (0.01, 5e-3), (0.5, 1e-3)])
def test_weight_decay_uses_raw_lr_not_bias_corrected(wd, lr):
    """Decay must scale with lr, not lr_t (bias-corrected lr).

    At t=1, beta2 bias correction makes lr_t << lr for beta2 close to 1.
    The AdamW decay term uses raw lr — so with zero gradient the full
    decay is exactly lr * wd * w regardless of t.
    """
    # Arrange
    opt, w, _ = _single_param([1.0], [0.0], lr=lr, weight_decay=wd)

    # Act
    opt.step(t=1)

    # Assert
    expected = 1.0 - lr * wd * 1.0
    np.testing.assert_allclose(w[0], expected, rtol=1e-6)


@pytest.mark.parametrize("w_init,g_init,wd", [
    ([10.0], [0.01], 1.0),
    ([5.0],  [0.1],  0.5),
    ([1.0],  [1.0],  2.0),
])
def test_moments_use_raw_gradient_not_decayed(w_init, g_init, wd):
    # Arrange
    opt, _, g = _single_param(w_init, g_init, lr=1e-3, weight_decay=wd)

    # Act
    opt.step(t=1)

    # Assert — first moment must track raw g, not g + wd*w
    expected_m = (1 - opt.beta1) * g[0]
    np.testing.assert_allclose(opt.m[0][0], expected_m, rtol=1e-6)


# ---------------------------------------------------------------------------
# Multiple parameters
# ---------------------------------------------------------------------------

MULTI_PARAM_SHAPES = [
    [(3,), (2, 4), (5,)],
    [(1,), (10,)],
    [(4, 4)],
]

@pytest.mark.parametrize("shapes", MULTI_PARAM_SHAPES)
def test_multiple_params_all_updated(shapes):
    # Arrange
    rng = np.random.default_rng(42)
    params = [(rng.standard_normal(s), rng.standard_normal(s)) for s in shapes]
    originals = [w.copy() for w, _ in params]
    opt = Adam(params, lr=1e-3)

    # Act
    opt.step(t=1)

    # Assert
    for (w, _), orig in zip(params, originals):
        assert not np.allclose(w, orig)
