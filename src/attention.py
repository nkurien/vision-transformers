import numpy as np


class MultiHeadAttention:
    """Multi-head self-attention.

    Args:
        embed_dim: Total embedding dimension D.
        num_heads: Number of attention heads h. embed_dim must be divisible by num_heads.

    Forward:
        Q, K, V: (B, N, D)
        mask:    (B, 1, 1, N) or (B, 1, N, N) boolean array; True positions are masked out.
        returns: output (B, N, D), attn_weights (B, h, N, N)

    Backward:
        grad_output: (B, N, D)
        returns: weight_grads {dW_q, dW_k, dW_v, dW_o}, input_grads {dQ, dK, dV}
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

        self.b_q = np.zeros(embed_dim)
        self.b_k = np.zeros(embed_dim)
        self.b_v = np.zeros(embed_dim)
        self.b_o = np.zeros(embed_dim)

        self._cache: dict | None = None

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
        q = Q @ self.W_q + self.b_q
        k = K @ self.W_k + self.b_k
        v = V @ self.W_v + self.b_v

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
        out_heads = attn_weights @ v

        # Merge heads: (B, N, D)
        out = out_heads.transpose(0, 2, 1, 3).reshape(B, N, D)

        # Output projection: (B, N, D)
        output = out @ self.W_o + self.b_o

        self._cache = dict(Q=Q, K=K, V=V, q=q, k=k, v=v, attn_weights=attn_weights, out=out)

        return output, attn_weights

    def backward(
        self, grad_output: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Backpropagate through multi-head attention.

        Args:
            grad_output: Gradient w.r.t. forward output, shape (B, N, D).

        Returns:
            weight_grads: {dW_q, dW_k, dW_v, dW_o} — each (D, D), summed over batch.
            input_grads:  {dQ, dK, dV}              — each (B, N, D).
        """
        assert self._cache is not None, "forward() must be called before backward()"
        Q   = self._cache["Q"]            # (B, N, D)
        K   = self._cache["K"]
        V   = self._cache["V"]
        q   = self._cache["q"]            # (B, h, N, d_k)
        k   = self._cache["k"]
        v   = self._cache["v"]
        A   = self._cache["attn_weights"] # (B, h, N, N)
        out = self._cache["out"]          # (B, N, D)  pre-W_o

        B, N, D = grad_output.shape
        h, d_k = self.num_heads, self.head_dim

        # 1. Output projection: output = out @ W_o + b_o
        dW_o  = out.reshape(-1, D).T @ grad_output.reshape(-1, D)  # (D, D)
        db_o  = grad_output.reshape(-1, D).sum(axis=0)             # (D,)
        d_out = grad_output @ self.W_o.T                            # (B, N, D)

        # 2. Reverse merge-heads reshape: (B, N, D) → (B, h, N, d_k)
        d_out_heads = d_out.reshape(B, N, h, d_k).transpose(0, 2, 1, 3)

        # 3. Attention output: out_heads = A @ v
        d_A = d_out_heads @ v.transpose(0, 1, 3, 2)   # (B, h, N, N)
        dv  = A.transpose(0, 1, 3, 2) @ d_out_heads   # (B, h, N, d_k)

        # 4. Softmax: A = softmax(scores)
        #    d_scores_i = A_i * (d_A_i - sum_j A_j * d_A_j)
        d_scores = A * (d_A - (d_A * A).sum(axis=-1, keepdims=True))  # (B, h, N, N)

        # 5. Scaled dot-product: scores = q @ k^T / sqrt(d_k)
        d_scores /= np.sqrt(d_k)
        dq = d_scores @ k                           # (B, h, N, d_k)
        dk = d_scores.transpose(0, 1, 3, 2) @ q    # (B, h, N, d_k)

        # 6. Reverse head split: (B, h, N, d_k) → (B, N, D)
        dq = dq.transpose(0, 2, 1, 3).reshape(B, N, D)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, N, D)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, N, D)

        # 7. Linear projections — weight grads summed over batch via reshape
        dW_q = Q.reshape(-1, D).T @ dq.reshape(-1, D)  # (D, D)
        dW_k = K.reshape(-1, D).T @ dk.reshape(-1, D)
        dW_v = V.reshape(-1, D).T @ dv.reshape(-1, D)
        db_q = dq.reshape(-1, D).sum(axis=0)           # (D,)
        db_k = dk.reshape(-1, D).sum(axis=0)
        db_v = dv.reshape(-1, D).sum(axis=0)

        dQ = dq @ self.W_q.T  # (B, N, D)
        dK = dk @ self.W_k.T
        dV = dv @ self.W_v.T

        weight_grads = dict(dW_q=dW_q, dW_k=dW_k, dW_v=dW_v, dW_o=dW_o,
                            db_q=db_q, db_k=db_k, db_v=db_v, db_o=db_o)
        input_grads  = dict(dQ=dQ, dK=dK, dV=dV)
        return weight_grads, input_grads
