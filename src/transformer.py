import numpy as np
from attention import MultiHeadAttention


class LayerNorm:
    """Layer normalisation over the last axis.

    gamma and beta are learnable; initialised to 1 and 0.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


def gelu(x: np.ndarray) -> np.ndarray:
    # Tanh approximation (Hendrycks & Gimpel, 2016); matches transformers default
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


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
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        self.dropout = dropout

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


class TransformerEncoder:
    """Stack of num_layers ViTBlocks.

    forward(x, training=True) → x   shape unchanged: (B, N, D)
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
