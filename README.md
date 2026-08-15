# CoVLA reproduction

The current Week 2 path trains trajectory prediction from one camera frame,
ego speed, and a fixed text instruction. It follows the course starter's
trajectory-only loss; rich-caption generation is a later extension.

Run the offline architecture/training smoke test on a few real samples:

```bash
uv run python -m src.train --epochs 3 --samples 4
```

`src/model.py` also contains `build_model`, which loads the real frozen CLIP
ViT-L/14 and Mistral-7B backbones for training on a GPU server.
