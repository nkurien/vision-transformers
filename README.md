# Vision Transformer from Scratch

Pure NumPy implementation of multi-head self-attention and Vision Transformers (ViT). No PyTorch, no TensorFlow. The end goal is a dog breed classifier on Stanford Dogs (120 classes) using transfer learning from timm pre-trained weights.

## Status

| Phase | Goal | Status |
|-------|------|--------|
| 1 | MultiHeadAttention + backprop | ✅ Done |
| 2 | ViT on MNIST (≥95% val acc) | ✅ Done — 98.58% val acc @ epoch 37 |
| 3 | Fine-tune on Stanford Dogs (≥75%) | ⏳ Not started |
| 4 | Attention visualisation | ⏳ Not started |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For Phase 3 fine-tuning, also install `timm` to export pre-trained weights (one-time, not a runtime dependency).

## Training

```bash
# MNIST baseline (downloads dataset automatically)
python scripts/train_mnist.py

# Optional overrides
python scripts/train_mnist.py --epochs 10 --batch-size 128
```

Checkpoints are saved to `weights/checkpoint_epoch_{n}.npy` after every epoch. Training history is saved to `weights/mnist_history.npy` on completion.

**CPU runtime:** ~18 min/epoch for ViT-Small at 96x96. Early stopping triggers after 10 epochs of no val_loss improvement.

**MNIST results:** 98.58% val accuracy at epoch 37 (train acc 98.78%). Training loss continued to decrease past ~epoch 20 while val loss plateaued around 0.07, indicating the model has saturated the task rather than catastrophically overfit — val accuracy kept improving.

## Testing

```bash
pytest tests/ -v -m "not slow"                        # fast (~2s)
pytest tests/test_attention.py -k "gradient"          # numerical gradient check (~30s)
```

## Architecture

Everything, forward pass, backward pass, weight updates, is implemented in NumPy.

**ViT-Small** (MNIST baseline): `embed_dim=192, layers=6, heads=3, mlp_dim=768, img_size=96, patch_size=16`

**ViT-Base** (Dogs fine-tuning): `embed_dim=768, layers=12, heads=12, mlp_dim=3072, img_size=224, patch_size=16`

Key implementation choices:
- Pre-norm transformer blocks (LayerNorm -> sublayer -> residual)
- Learnable positional encoding and [CLS] token
- Numerically stable softmax backward
- LayerNorm backward via the Ba et al. (2016) closed-form expression
- Adam with L2 weight decay and cosine LR schedule with linear warmup

## Stanford Dogs — Phase 3 notes

The dataset has ~171 images per class, which is tight. A few things that matter at this scale:

- **Weight transfer** — exports ViT-Base weights from timm, splits the fused QKV matrix `(3D, D)` into separate W_q/W_k/W_v, saves to `.npy`. Run `python scripts/export_timm_weights.py --validate` to confirm forward pass matches timm before fine-tuning.
- **Freezing** — first 8 of 12 blocks frozen via `model.freeze_layers(until=-4)`. Full fine-tuning risks overfitting at 171/class.
- **Augmentation** — random crop, horizontal flip, color jitter applied per-batch in the training loop (not pre-cached), so each epoch sees different augmentations.

Expected val accuracy with this setup: ~80-85% without augmentation, ~85-90% with. The ≥75% target is conservative.

## References

- Dosovitskiy et al. (2021) [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Vaswani et al. (2017) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- timm [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
