import numpy as np
from attention import MultiHeadAttention


class LayerNorm:
    """Layer normalisation over the last axis.

    gamma and beta are learnable; initialised to 1 and 0.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim)
        self.beta  = np.zeros(dim)
        self.eps   = eps
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean  = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1,  keepdims=True)
        std   = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        self._cache = dict(x_hat=x_hat, std=std)
        return self.gamma * x_hat + self.beta

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        """Return grad_x; also stores grad_gamma and grad_beta as attributes."""
        x_hat = self._cache["x_hat"]
        std   = self._cache["std"]

        leading = tuple(range(grad_y.ndim - 1))
        self.grad_gamma = (grad_y * x_hat).sum(axis=leading)  # (D,)
        self.grad_beta  = grad_y.sum(axis=leading)             # (D,)

        # Closed-form grad through normalisation (see Ba et al., 2016 appendix)
        grad_x_hat = grad_y * self.gamma
        grad_x = (1.0 / std) * (
            grad_x_hat
            - grad_x_hat.mean(axis=-1, keepdims=True)
            - x_hat * (grad_x_hat * x_hat).mean(axis=-1, keepdims=True)
        )
        return grad_x


def gelu(x: np.ndarray) -> np.ndarray:
    # Tanh approximation (Hendrycks & Gimpel, 2016); matches transformers default
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x**3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * c * (1.0 + 3 * 0.044715 * x**2)


class ViTBlock:
    """Single Vision Transformer block (pre-norm).

    Forward: LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual
    MLP:     Linear(embed_dim → mlp_dim) → GELU → Dropout → Linear(mlp_dim → embed_dim)

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads.
        mlp_dim:   Hidden dimension of the MLP (typically 4 × embed_dim).
        dropout:   Dropout probability applied after attention and between MLP layers.

    forward(x, training=True) → x   shape unchanged: (B, N, D)
    backward(grad_output)     → grad_input (B, N, D)
        Weight gradients stored as grad_W1, grad_b1, grad_W2, grad_b2;
        LayerNorm param grads on norm1/norm2 as grad_gamma, grad_beta.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        self.dropout   = dropout
        self.embed_dim = embed_dim
        self.mlp_dim   = mlp_dim

        self.norm1 = LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = LayerNorm(embed_dim)

        # MLP weights — scaled Xavier
        s1 = np.sqrt(2.0 / (embed_dim + mlp_dim))
        s2 = np.sqrt(2.0 / (mlp_dim   + embed_dim))
        self.W1 = np.random.randn(embed_dim, mlp_dim)  * s1
        self.b1 = np.zeros(mlp_dim)
        self.W2 = np.random.randn(mlp_dim,  embed_dim) * s2
        self.b2 = np.zeros(embed_dim)

        self.cache: dict = {}

    def _drop(
        self, x: np.ndarray, training: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply inverted dropout; return (dropped_x, mask) or (x, None)."""
        if not training or self.dropout == 0.0:
            return x, None
        mask = (np.random.rand(*x.shape) >= self.dropout)
        return x * mask / (1.0 - self.dropout), mask

    def _drop_backward(
        self, grad: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        if mask is None:
            return grad
        return grad * mask / (1.0 - self.dropout)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # --- Attention sub-layer ---
        x_in    = x
        x_norm1 = self.norm1.forward(x)
        attn_out, attn_weights = self.attn.forward(x_norm1, x_norm1, x_norm1)
        attn_out, drop_mask1 = self._drop(attn_out, training)
        x = x_in + attn_out

        # --- MLP sub-layer ---
        x_after_attn = x
        x_norm2      = self.norm2.forward(x)
        h            = x_norm2 @ self.W1 + self.b1   # (B, N, mlp_dim)
        h_act        = gelu(h)
        h_drop, drop_mask2 = self._drop(h_act, training)
        mlp_out      = h_drop @ self.W2 + self.b2    # (B, N, embed_dim)
        mlp_out, drop_mask3 = self._drop(mlp_out, training)
        x = x_after_attn + mlp_out

        self.cache = dict(
            x_in=x_in,
            x_norm1=x_norm1,
            attn_weights=attn_weights,
            drop_mask1=drop_mask1,
            x_after_attn=x_after_attn,
            x_norm2=x_norm2,
            h=h,
            h_act=h_act,
            h_drop=h_drop,
            drop_mask2=drop_mask2,
            drop_mask3=drop_mask3,
        )
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate through the block.

        Args:
            grad_output: (B, N, D) — gradient w.r.t. this block's output.

        Returns:
            grad_input: (B, N, D)

        Side effects:
            Stores grad_W1, grad_b1, grad_W2, grad_b2 as attributes.
            Stores grad_gamma / grad_beta on self.norm1 and self.norm2.
            Stores weight gradients on self.attn via its own backward().
        """
        c = self.cache
        B, N, D = grad_output.shape

        # 1. Second residual: x_out = x_after_attn + mlp_dropped
        d_x_after_attn = grad_output          # identity branch
        d_mlp_dropped  = grad_output

        # 2. MLP path (reverse)

        # dropout3
        d_mlp_out = self._drop_backward(d_mlp_dropped, c["drop_mask3"])

        # W2 + b2: mlp_out = h_drop @ W2 + b2
        self.grad_W2 = c["h_drop"].reshape(-1, self.mlp_dim).T @ d_mlp_out.reshape(-1, D)
        self.grad_b2 = d_mlp_out.sum(axis=(0, 1))
        d_h_drop     = d_mlp_out @ self.W2.T

        # dropout2
        d_h_act = self._drop_backward(d_h_drop, c["drop_mask2"])

        # GELU: h_act = gelu(h)
        d_h = d_h_act * _gelu_grad(c["h"])

        # W1 + b1: h = x_norm2 @ W1 + b1
        self.grad_W1 = c["x_norm2"].reshape(-1, D).T @ d_h.reshape(-1, self.mlp_dim)
        self.grad_b1 = d_h.sum(axis=(0, 1))
        d_x_norm2    = d_h @ self.W1.T

        # LayerNorm2
        d_x_after_attn += self.norm2.backward(d_x_norm2)

        # 3. First residual: x_after_attn = x_in + attn_dropped
        d_x_in       = d_x_after_attn   # identity branch
        d_attn_dropped = d_x_after_attn

        # 4. Attention path (reverse)

        # dropout1
        d_attn_out = self._drop_backward(d_attn_dropped, c["drop_mask1"])

        # MultiHeadAttention backward (Q = K = V = x_norm1 in self-attention)
        attn_weight_grads, attn_input_grads = self.attn.backward(d_attn_out)
        self.attn.grad_W_q = attn_weight_grads["dW_q"]
        self.attn.grad_W_k = attn_weight_grads["dW_k"]
        self.attn.grad_W_v = attn_weight_grads["dW_v"]
        self.attn.grad_W_o = attn_weight_grads["dW_o"]
        d_x_norm1 = attn_input_grads["dQ"] + attn_input_grads["dK"] + attn_input_grads["dV"]

        # LayerNorm1
        d_x_in += self.norm1.backward(d_x_norm1)

        return d_x_in


class TransformerEncoder:
    """Stack of num_layers ViTBlocks.

    forward(x, training=True)  → x            shape unchanged: (B, N, D)
    backward(grad_output)      → grad_input    shape unchanged: (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for block in self.blocks:
            x = block.forward(x, training=training)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        for block in reversed(self.blocks):
            grad_output = block.backward(grad_output)
        return grad_output
