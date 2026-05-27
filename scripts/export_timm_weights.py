"""Export pre-trained ViT weights from timm to NumPy .npy format.

Usage:
    python scripts/export_timm_weights.py --model vit_base_patch16_224 --output weights/
    python scripts/export_timm_weights.py --model vit_small_patch16_224 --output weights/ --validate

The only timm key not exported is patch_embed.proj.bias (our patch embedding
has no bias). All attention biases and the final LayerNorm are now fully
exported. Logits should match timm closely when --validate is used.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train import load_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSUPPORTED_KEYS: set[str] = set()  # keys to silently drop; extend if model variants add unsupported params


def _map_weights(state_dict, num_layers: int) -> dict:
    """Convert a timm state dict to our checkpoint schema.

    timm linear weights are (out, in); ours are (in, out), so all
    weight matrices are transposed.
    """
    out = {}
    skipped = []

    for key, val in state_dict.items():
        v = val.numpy()

        # --- dropped keys ---
        if key in _UNSUPPORTED_KEYS:
            skipped.append(key)
            continue

        # --- patch embedding ---
        if key == "patch_embed.proj.weight":
            # timm: (embed_dim, C, P, P) → flatten C-first to (patch_dim, embed_dim)
            D = v.shape[0]
            out["patch_embed"] = v.reshape(D, -1).T
            continue

        if key == "patch_embed.proj.bias":
            out["patch_embed_bias"] = v  # (embed_dim,)
            continue

        if key == "cls_token":
            out["cls_token"] = v          # (1, 1, D) — same shape
            continue

        if key == "pos_embed":
            out["pos_embed"] = v          # (1, N+1, D) — same shape
            continue

        # --- classification head ---
        if key == "head.weight":
            out["head_W"] = v.T           # (num_classes, D) → (D, num_classes)
            continue

        if key == "head.bias":
            out["head_b"] = v
            continue

        # --- final LayerNorm ---
        if key == "norm.weight":
            out["norm_gamma"] = v
            continue

        if key == "norm.bias":
            out["norm_beta"] = v
            continue

        # --- per-block weights ---
        # Expected prefixes: "blocks.{i}.*"
        if not key.startswith("blocks."):
            skipped.append(key)
            continue

        parts = key.split(".", 2)         # ["blocks", "{i}", "rest"]
        i     = int(parts[1])
        rest  = parts[2]

        # LayerNorms
        if rest == "norm1.weight":
            out[f"b{i}_norm1_gamma"] = v
        elif rest == "norm1.bias":
            out[f"b{i}_norm1_beta"] = v
        elif rest == "norm2.weight":
            out[f"b{i}_norm2_gamma"] = v
        elif rest == "norm2.bias":
            out[f"b{i}_norm2_beta"] = v

        # Fused QKV weight: (3D, D) → three (D, D) matrices in (in, out) layout
        elif rest == "attn.qkv.weight":
            D = v.shape[1]
            W = v.reshape(3, D, D)        # (3, out=D, in=D)
            out[f"b{i}_W_q"] = W[0].T    # (D, D) in (in, out)
            out[f"b{i}_W_k"] = W[1].T
            out[f"b{i}_W_v"] = W[2].T

        # Fused QKV bias: (3D,) → three (D,) vectors
        elif rest == "attn.qkv.bias":
            D = v.shape[0] // 3
            b = v.reshape(3, D)
            out[f"b{i}_b_q"] = b[0]
            out[f"b{i}_b_k"] = b[1]
            out[f"b{i}_b_v"] = b[2]

        # Output projection weight: (D, D) in (out, in) → (in, out)
        elif rest == "attn.proj.weight":
            out[f"b{i}_W_o"] = v.T

        # Output projection bias: (D,) — no transpose needed
        elif rest == "attn.proj.bias":
            out[f"b{i}_b_o"] = v

        # MLP fc1: (mlp_dim, D) → (D, mlp_dim)
        elif rest == "mlp.fc1.weight":
            out[f"b{i}_W1"] = v.T
        elif rest == "mlp.fc1.bias":
            out[f"b{i}_b1"] = v

        # MLP fc2: (D, mlp_dim) → (mlp_dim, D)
        elif rest == "mlp.fc2.weight":
            out[f"b{i}_W2"] = v.T
        elif rest == "mlp.fc2.bias":
            out[f"b{i}_b2"] = v

        else:
            skipped.append(key)

    if skipped:
        print(f"  Skipped {len(skipped)} unsupported key(s):")
        for k in skipped:
            print(f"    {k}")

    return out


def _validate(timm_model, our_model, model_name: str) -> None:
    """Run both models on the same random input and report similarity.

    Full logit match is not expected (missing biases + final norm), but the
    CLS representations should be highly correlated after weight transfer.
    """
    import torch

    rng  = np.random.default_rng(0)
    img_size = 224 if "base" in model_name or "large" in model_name else 224
    x_np  = rng.standard_normal((1, 3, img_size, img_size)).astype(np.float32)
    x_th  = torch.from_numpy(x_np)

    timm_model.eval()
    with torch.no_grad():
        timm_logits = timm_model(x_th).numpy()   # (1, num_classes)

    our_logits = our_model.forward(x_np, training=False)  # (1, num_classes)

    # Cosine similarity between logit vectors
    def cosine(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    sim = cosine(timm_logits, our_logits)
    mae = float(np.abs(timm_logits - our_logits).mean())
    print(f"\n  Logit cosine similarity : {sim:.4f}  (1.0 = perfect)")
    print(f"  Logit mean abs error    : {mae:.4f}")
    print("  Note: small gap expected — patch_embed has no bias in our model,")
    print("        and float32 accumulation differs between NumPy and PyTorch.")
    if sim > 0.99:
        print("  [PASS] similarity looks good for fine-tuning")
    else:
        print("  [WARN] low similarity — check weight mapping (expected >0.99)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "vit_small_patch16_224": dict(
        img_size=224, patch_size=16, embed_dim=384,
        num_layers=12, num_heads=6, mlp_dim=1536, num_classes=1000,
    ),
    "vit_base_patch16_224": dict(
        img_size=224, patch_size=16, embed_dim=768,
        num_layers=12, num_heads=12, mlp_dim=3072, num_classes=1000,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export timm ViT weights to NumPy")
    parser.add_argument(
        "--model",    default="vit_base_patch16_224",
        choices=list(_MODEL_CONFIGS),
        help="timm model name to export",
    )
    parser.add_argument(
        "--output",   default="weights/",
        help="directory to write the .npy file",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="run a forward pass to check similarity after export",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import timm
        import torch
    except ImportError:
        sys.exit("timm and torch are required for export — pip install timm torch")

    print(f"Loading timm model: {args.model}")
    timm_model = timm.create_model(args.model, pretrained=True)
    state_dict = timm_model.state_dict()
    print(f"  {len(state_dict)} keys in timm state dict")

    print("Mapping weights...")
    cfg     = _MODEL_CONFIGS[args.model]
    state   = _map_weights(state_dict, num_layers=cfg["num_layers"])
    print(f"  {len(state)} keys exported")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.model}_pretrained.npy")
    np.save(out_path, state, allow_pickle=True)
    print(f"\nSaved → {out_path}")

    if args.validate:
        from vit import VisionTransformer
        our_model = VisionTransformer(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            mlp_dim=cfg["mlp_dim"],
            num_classes=cfg["num_classes"],
        )
        print("Loading weights into NumPy model...")
        load_weights(our_model, out_path)
        print("Validating...")
        _validate(timm_model, our_model, args.model)


if __name__ == "__main__":
    main()
