import numpy as np


class MultiHeadAttention:
    """Multi-head self-attention (forward pass only).

    Args:
        embed_dim: Total embedding dimension D.
        num_heads: Number of attention heads h. embed_dim must be divisible by num_heads.

    Forward:
        Q, K, V: (B, N, D)
        mask:    (B, 1, 1, N) or (B, 1, N, N) boolean array; True positions are masked out.
        returns: output (B, N, D), attn_weights (B, h, N, N)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Scaled Xavier (Glorot) initialisation
        scale = np.sqrt(2.0 / (embed_dim + embed_dim))
        self.W_q = np.random.randn(embed_dim, embed_dim) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim) * scale

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute multi-head attention.

        Args:
            Q: Query tensor (B, N, D)
            K: Key tensor   (B, N, D)
            V: Value tensor (B, N, D)
            mask: Optional boolean mask (B, 1, 1, N) or (B, 1, N, N).
                  Positions where mask is True are set to -inf before softmax.

        Returns:
            output:       (B, N, D)
            attn_weights: (B, num_heads, N, N)  — weights after softmax, summing to 1 per query
        """
        B, N, D = Q.shape
        h, d_k = self.num_heads, self.head_dim

        # Linear projections: (B, N, D)
        q = Q @ self.W_q
        k = K @ self.W_k
        v = V @ self.W_v

        # Split into heads and move head axis forward: (B, h, N, d_k)
        q = q.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product scores: (B, h, N, N)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax over key dimension (axis=-1)
        scores -= scores.max(axis=-1, keepdims=True)  # numerical stability
        attn_weights = np.exp(scores)
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

        # Weighted sum over values: (B, h, N, d_k)
        out = attn_weights @ v

        # Merge heads: (B, N, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

        # Output projection: (B, N, D)
        output = out @ self.W_o

        return output, attn_weights
