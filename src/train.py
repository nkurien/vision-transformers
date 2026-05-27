import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Softmax cross-entropy loss and its gradient w.r.t. logits."""
    B     = logits.shape[0]
    probs = _softmax(logits)
    loss  = -np.log(probs[np.arange(B), y] + 1e-10).mean()
    grad  = probs.copy()
    grad[np.arange(B), y] -= 1.0
    grad /= B
    return loss, grad


def _get_params(model) -> list:
    """Return (weight, grad) pairs for all trainable (non-frozen) parameters.

    Safe to call before the first backward pass — missing grad attributes fall
    back to zero arrays so Adam can be initialised from this list.
    """
    def p(w, attr, obj=None):
        obj = model if obj is None else obj
        return (w, getattr(obj, attr, np.zeros_like(w)))

    pairs = [
        p(model.head_W,      "grad_head_W"),
        p(model.head_b,      "grad_head_b"),
        p(model.patch_embed, "grad_patch_embed"),
        p(model.cls_token,   "grad_cls_token"),
        p(model.pos_embed,   "grad_pos_encoding"),
    ]

    for block in model.encoder.blocks:
        if getattr(block, "frozen", False):
            continue
        a = block.attn
        pairs += [
            p(block.norm1.gamma, "grad_gamma", block.norm1),
            p(block.norm1.beta,  "grad_beta",  block.norm1),
            p(a.W_q, "grad_W_q", a),
            p(a.W_k, "grad_W_k", a),
            p(a.W_v, "grad_W_v", a),
            p(a.W_o, "grad_W_o", a),
            p(block.norm2.gamma, "grad_gamma", block.norm2),
            p(block.norm2.beta,  "grad_beta",  block.norm2),
            p(block.W1, "grad_W1", block),
            p(block.b1, "grad_b1", block),
            p(block.W2, "grad_W2", block),
            p(block.b2, "grad_b2", block),
        ]

    return pairs


def _save_checkpoint(model, epoch: int) -> None:
    os.makedirs("weights", exist_ok=True)
    state = {
        "head_W":      model.head_W,
        "head_b":      model.head_b,
        "patch_embed": model.patch_embed,
        "cls_token":   model.cls_token,
        "pos_embed":   model.pos_embed,
    }
    for i, block in enumerate(model.encoder.blocks):
        state[f"b{i}_norm1_gamma"] = block.norm1.gamma
        state[f"b{i}_norm1_beta"]  = block.norm1.beta
        state[f"b{i}_W_q"] = block.attn.W_q
        state[f"b{i}_W_k"] = block.attn.W_k
        state[f"b{i}_W_v"] = block.attn.W_v
        state[f"b{i}_W_o"] = block.attn.W_o
        state[f"b{i}_norm2_gamma"] = block.norm2.gamma
        state[f"b{i}_norm2_beta"]  = block.norm2.beta
        state[f"b{i}_W1"] = block.W1
        state[f"b{i}_b1"] = block.b1
        state[f"b{i}_W2"] = block.W2
        state[f"b{i}_b2"] = block.b2
    np.save(f"weights/checkpoint_epoch_{epoch}.npy", state, allow_pickle=True)


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def cosine_schedule(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> float:
    """Linear warmup from lr_max/warmup_epochs to lr_max, then cosine decay to lr_min.

    Warmup starts at a nonzero value (lr_max / warmup_epochs) so the first
    epoch produces visible learning. Starting from ~0 (lr_min) made the first
    epoch a no-op with Adam's bias correction amplifying the tiny LR.
    """
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))


# ---------------------------------------------------------------------------
# Adam optimiser
# ---------------------------------------------------------------------------

class Adam:
    """Adam optimiser with L2 weight decay added to the gradient before the
    moment update (i.e. not decoupled).

    Args:
        params:       list of (weight, grad) tuples; update .params with fresh
                      grad references after each backward before calling step().
        lr:           Learning rate; can be changed between steps via .lr.
        beta1, beta2: Exponential decay rates for first and second moments.
        eps:          Denominator stability constant.
        weight_decay: L2 coefficient applied to weight before moment update.
    """

    def __init__(
        self,
        params: list,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-5,
    ):
        self.params       = params
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(w) for w, _ in params]
        self.v = [np.zeros_like(w) for w, _ in params]

    def step(self, t: int) -> None:
        """Update all weights in-place.

        Args:
            t: Current step (1-indexed); used for bias correction.
        """
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** t) / (1.0 - self.beta1 ** t)
        for i, (w, g) in enumerate(self.params):
            self.m[i]  = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i]  = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2
            # decoupled weight decay (AdamW): applied directly to weights, not via gradient
            w         -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps) + self.lr * self.weight_decay * w


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, X: np.ndarray, y: np.ndarray, batch_size: int, optimizer: Adam, t: int
) -> tuple[float, float]:
    """One full pass over the training data.

    Args:
        model:      VisionTransformer.
        X:          (N, C, H, W) training images.
        y:          (N,) integer labels.
        batch_size: Mini-batch size.
        optimizer:  Adam instance; .params refreshed after each backward.
        t:          Global step count at epoch start (1-indexed).

    Returns:
        (mean_loss, accuracy) over the epoch.
    """
    N       = X.shape[0]
    indices = np.random.permutation(N)
    total_loss    = 0.0
    total_correct = 0
    num_batches   = 0

    num_steps = (N - batch_size) // batch_size + 1
    pbar = tqdm(range(0, N - batch_size + 1, batch_size), total=num_steps,
                unit="batch", leave=False)

    for start in pbar:
        idx      = indices[start : start + batch_size]
        xb, yb   = X[idx], y[idx]

        logits            = model.forward(xb, training=True)
        loss, grad_logits = _cross_entropy(logits, yb)
        model.backward(grad_logits)

        optimizer.params = _get_params(model)
        optimizer.step(t + num_batches + 1)

        total_loss    += loss
        total_correct += (logits.argmax(axis=-1) == yb).sum()
        num_batches   += 1

        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}",
                         acc=f"{total_correct / (num_batches * batch_size):.4f}")

    return total_loss / max(num_batches, 1), total_correct / N


def validate(
    model, X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[float, float]:
    """Evaluate the model without weight updates.

    Args:
        model:      VisionTransformer.
        X:          (N, C, H, W) images.
        y:          (N,) integer labels.
        batch_size: Batch size for inference.

    Returns:
        (mean_loss, accuracy).
    """
    N             = X.shape[0]
    total_loss    = 0.0
    total_correct = 0
    num_batches   = 0

    for start in range(0, N - batch_size + 1, batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]

        logits   = model.forward(xb, training=False)
        loss, _  = _cross_entropy(logits, yb)

        total_loss    += loss
        total_correct += (logits.argmax(axis=-1) == yb).sum()
        num_batches   += 1

    return total_loss / max(num_batches, 1), total_correct / N


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:  int,
    lr:      float,
    batch_size: int,
    warmup_epochs: int = 5,
) -> dict:
    """Train model with cosine LR schedule and early stopping (patience=10).

    Saves a checkpoint after every epoch to weights/checkpoint_epoch_{n}.npy.

    Args:
        model:         VisionTransformer.
        X_train/y_train: Training images (N, C, H, W) and integer labels (N,).
        X_val/y_val:   Validation images and labels.
        epochs:        Maximum epochs.
        lr:            Peak learning rate (end of warmup).
        batch_size:    Mini-batch size.
        warmup_epochs: Epochs for linear LR warmup.

    Returns:
        history: dict of lists — train_loss, train_acc, val_loss, val_acc, lr.
    """
    optimizer = Adam(_get_params(model), lr=lr)
    history   = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], lr=[])

    best_val_loss     = float("inf")
    patience          = 0
    t                 = 0
    batches_per_epoch = max(len(X_train) // batch_size, 1)

    for epoch in range(epochs):
        current_lr   = cosine_schedule(epoch, epochs, warmup_epochs, lr)
        optimizer.lr = current_lr

        train_loss, train_acc = train_one_epoch(
            model, X_train, y_train, batch_size, optimizer, t
        )
        t += batches_per_epoch

        val_loss, val_acc = validate(model, X_val, y_val, batch_size)

        _save_checkpoint(model, epoch + 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:3d} | "
            f"loss {train_loss:.4f} | acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience      = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"Early stopping at epoch {epoch + 1} (patience=10).")
                break

    return history
