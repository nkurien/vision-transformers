import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from transformer import TransformerEncoder


class VisionTransformer:
    """Vision Transformer (ViT) for image classification.

    Args:
        img_size:    Input image height/width (square assumed).
        patch_size:  Size of each square patch.
        in_channels: Number of input channels (default 3 for RGB).
        embed_dim:   Token embedding dimension D.
        num_layers:  Number of TransformerEncoder blocks.
        num_heads:   Attention heads per block.
        mlp_dim:     MLP hidden dimension inside each block.
        num_classes: Number of output classes.
        dropout:     Dropout probability.

    forward(x, training=False) → logits (B, num_classes)
        x: (B, C, H, W) float array, pixel values normalised by the caller.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.0,
    ):
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.in_channels = in_channels

        num_patches = (img_size // patch_size) ** 2
        patch_dim   = patch_size * patch_size * in_channels

        # Patch embedding projection
        scale = np.sqrt(2.0 / (patch_dim + embed_dim))
        self.patch_embed = np.random.randn(patch_dim, embed_dim) * scale

        # Learnable [CLS] token and positional encoding
        self.cls_token = np.zeros((1, 1, embed_dim))
        self.pos_embed = np.random.randn(1, num_patches + 1, embed_dim) * 0.02

        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Classification head (linear, on [CLS] output only)
        head_scale    = np.sqrt(2.0 / (embed_dim + num_classes))
        self.head_W   = np.random.randn(embed_dim, num_classes) * head_scale
        self.head_b   = np.zeros(num_classes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Run the full ViT forward pass.

        Args:
            x: Input images (B, C, H, W).
            training: Enables dropout when True.

        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]

        # 1. Extract and flatten patches: (B, num_patches, patch_dim)
        patches = self._extract_patches(x)

        # 2. Linear patch embedding: (B, num_patches, embed_dim)
        tokens = patches @ self.patch_embed

        # 3. Prepend [CLS] token: (B, num_patches+1, embed_dim)
        cls = np.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        tokens = np.concatenate([cls, tokens], axis=1)

        # 4. Add positional encoding
        tokens = tokens + self.pos_embed

        # 5. Transformer encoder
        tokens = self.encoder.forward(tokens, training=training)

        # 6. Extract [CLS] output and classify
        cls_out = tokens[:, 0]                       # (B, embed_dim)
        logits  = cls_out @ self.head_W + self.head_b  # (B, num_classes)
        return logits

    def freeze_layers(self, until: int) -> None:
        """Freeze all transformer blocks except the last abs(until).

        Example: freeze_layers(until=-4) freezes all but the last 4 blocks
        and leaves the classification head always trainable.
        """
        n     = abs(until)
        total = len(self.encoder.blocks)
        for i, block in enumerate(self.encoder.blocks):
            block.frozen = i < (total - n)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Rearrange image pixels into a sequence of flattened patches.

        Args:
            x: (B, C, H, W)

        Returns:
            patches: (B, num_patches, patch_size*patch_size*C)
        """
        B, C, H, W = x.shape
        P  = self.patch_size
        h  = H // P
        w  = W // P
        x  = x.reshape(B, C, h, P, w, P)
        x  = x.transpose(0, 2, 4, 3, 5, 1)   # (B, h, w, P, P, C)
        return x.reshape(B, h * w, P * P * C)
