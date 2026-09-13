# CoVLA reproduction

The data loader and visualization work live in `src/data.py` and
`src/visualize.py`.

Run the one-epoch smoke test with:

```bash
uv run python -m src.train
```

Without CUDA, this uses tiny random models to verify training, caption
generation, W&B/TensorBoard logging, trajectory visualization, worst-case
ranking, and speed sensitivity. CUDA runs the configured CLIP and Mistral
models on two scenes; adjust `main()` before a full run.

## Full data on Colab

`scripts/prepare_covla_full.py` builds a resumable, compact training set on
Colab's `/mnt/local-scratch` disk. It keeps every tenth full-dataset frame as
a 224×224 JPEG, retains every tenth Mini frame as its original-resolution PNG
for validation, and excludes the 50 Mini scenes from training by timestamp.

```python
mini_root = prepare_covla_mini(hf_token)
train_root = prepare_covla_full(hf_token, mini_root=mini_root)
```

Use `train_root` as `data_dir`, `mini_root` as `val_data_dir`, and keep
`frame_interval=10` in `Week2CoVLAConfig`. The scratch disk is runtime-local,
so a Colab disconnect removes the prepared data. For a long training run,
mount Google Drive and enable compact checkpoints there:

```python
config = Week2CoVLAConfig(
    data_dir=str(train_root),
    val_data_dir=str(mini_root),
    num_scenes=None,
    frame_interval=10,
    checkpoint_dir="/content/drive/MyDrive/CoVLA/checkpoints",
    checkpoint_every_steps=1000,
    resume_from_checkpoint=True,
)
```

The checkpoint contains only the trainable layers and AdamW state, not the
frozen CLIP and Mistral weights. Resuming reconstructs the same shuffled epoch
and continues at the next batch without rereading skipped images.
