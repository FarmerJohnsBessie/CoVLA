from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPImageProcessor

from src.data import CoVLADataset, get_scene_ids
from src.model import (
    Week2CoVLAConfig,
    Week2VLAModel,
    build_model,
    compute_ade,
    compute_fde,
)
from src.utils import generate_caption, predict_trajectory
from src.visualize import plot_prediction


def _autocast(device: torch.device):
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=device.type == "cuda",
    )


class Week2DataCollator:
    """Convert dataset samples into a padded multimodal model batch."""

    def __init__(self, config: Week2CoVLAConfig) -> None:
        self.config = config

        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_encoder_hf
        )

        self.tokenizer= AutoTokenizer.from_pretrained(
            config.language_model_hf
        )

        if getattr(self.tokenizer, "chat_template", None):
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": config.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = config.prompt

        prompt = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        self.prompt_ids = prompt["input_ids"][0]
        self.prompt_mask = prompt["attention_mask"][0]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token



    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(samples)

        caption_rows = []

        for sample in samples:
            caption_ids = self.tokenizer.encode(
                sample["caption"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_caption_tokens - 1,
            )

            caption_ids.append(self.tokenizer.eos_token_id)

            caption_rows.append(
                torch.tensor(caption_ids, dtype=torch.long)
            )

        caption_input_ids = pad_sequence(
            caption_rows,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        caption_attention_mask = pad_sequence(
            [torch.ones_like(row) for row in caption_rows],
            batch_first=True,
            padding_value=0,
        )



        pixel_values = self.image_processor(
            images=[sample["image"] for sample in samples],
            return_tensors="pt"
        )["pixel_values"]

        return {
            "pixel_values": pixel_values, # (B, 3, 244, 244)
            "ego_speed": torch.stack(
                [sample["speed"] for sample in samples]
            ), # (B, )
            "prompt_ids": self.prompt_ids.unsqueeze(0).repeat(batch_size,1), # (B, Number of words)
            "prompt_mask": self.prompt_mask.unsqueeze(0).repeat(batch_size,1),
            "caption_ids": caption_input_ids,
            "caption_mask": caption_attention_mask,
            "gt_trajectory": torch.stack(
                [sample["trajectory"] for sample in samples]
            )
        }


class Week2VLATrainer:
    """Implement optimization, validation, metrics, and logging here."""

    def __init__(
        self,
        model: Week2VLAModel,
        config: Week2CoVLAConfig,
        wandb_run: Any,
        collator: Week2DataCollator,
    ) -> None:
        self.model = model
        self.config = config

        self.trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=config.learning_rate,
        )

        self.global_step = 0
        self.total_sample = 0
        self.current_epoch = 0
        self.next_batch = 0
        self.writer = SummaryWriter(config.log_dir)
        self.device = torch.device(self.config.device)
        self.wandb_run = wandb_run
        self.collator = collator

    @property
    def checkpoint_path(self) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir) / "latest.pt"

    def save_checkpoint(
        self,
        epoch: int,
        next_batch: int,
        archive_epoch: bool = False,
    ) -> None:
        """Atomically save trainable weights and optimizer state."""
        path = self.checkpoint_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        state = {
            "model": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
                if name in trainable_names
            },
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "global_step": self.global_step,
            "total_sample": self.total_sample,
            "batch_size": self.config.batch_size,
            "seed": self.config.seed,
        }
        destinations = [path]
        if archive_epoch:
            destinations.append(path.parent / f"epoch_{epoch:04d}.pt")
        for destination in destinations:
            temporary = destination.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(destination)
        print(f"Saved checkpoint: {path}", flush=True)

    def load_checkpoint(self, checkpoint: str | Path) -> tuple[int, int]:
        """Restore a compact checkpoint and return (epoch, next_batch)."""
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("batch_size") != self.config.batch_size:
            raise ValueError("Checkpoint batch_size does not match the config")
        if state.get("seed") != self.config.seed:
            raise ValueError("Checkpoint seed does not match the config")
        self.model.load_state_dict(state["model"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate
        self.global_step = state["global_step"]
        self.total_sample = state["total_sample"]
        print(
            f"Resuming {path} at epoch {state['epoch'] + 1}, "
            f"batch {state['next_batch']} (step {self.global_step})."
        )
        return state["epoch"], state["next_batch"]



    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    def log_training_visualization(
        self,
        sample: dict[str, Any],
        prediction: torch.Tensor,
    ) -> None:
        """Upload one camera/BEV training prediction to W&B."""
        figure = None
        try:
            generated_caption = generate_caption(
                self.model,
                sample,
                self.collator,
            )
            figure = plot_prediction(sample, prediction, generated_caption)
            self.wandb_run.log({
                "samples_seen": self.total_sample,
                "training_example/visualization": wandb.Image(figure),
                "training_example/ego_speed_mps": float(sample["speed"]),
                "training_example/scene_id": sample["scene_id"],
                "training_example/frame_id": sample["frame_id"],
                "training_example/ground_truth_caption": sample["caption"],
                "training_example/generated_caption": generated_caption,
            })
            self.writer.add_figure(
                "training/example",
                figure,
                self.global_step,
            )
            print(
                f"Uploaded training visualization at step={self.global_step} "
                f"scene={sample['scene_id']} frame={sample['frame_id']}",
                flush=True,
            )
        finally:
            if figure is not None:
                plt.close(figure)
            self.model.train()

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        batch_offset: int = 0,
        val_loader: DataLoader | None = None,
        visualization_dataset: CoVLADataset | None = None,
        sample_indices: list[int] | None = None,
    ) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "caption_loss": 0.0,
            "trajectory_loss": 0.0,
        }
        sample_count = 0
        total_batches = batch_offset + len(loader)
        self.current_epoch = epoch
        self.next_batch = batch_offset

        for batch_index, batch in enumerate(loader):
            batch = self._move(batch)

            self.optimizer.zero_grad()
            with _autocast(self.device):
                output = self.model(**batch)
                loss = output["loss"]

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                self.config.max_grad_norm,
            )
            speed_gradient = self.model.speed_projection.weight.grad
            speed_gradient_norm = (
                speed_gradient.norm().item()
                if speed_gradient is not None
                else 0.0
            )

            self.optimizer.step()

            batch_size = batch["pixel_values"].shape[0]
            sample_count += batch_size

            for key in totals:
                value = output[key]
                totals[key] += value.item() * batch_size

            # --- Record Batch Info ---
            self.writer.add_scalar(
                "train/total_loss",
                output["loss"].item(),
                self.global_step,
            )
            self.writer.add_scalar(
                "train/caption_loss",
                output["caption_loss"].item(),
                self.global_step,
            )
            self.writer.add_scalar(
                "train/trajectory_loss",
                output["trajectory_loss"].item(),
                self.global_step,
            )
            self.writer.add_scalar(
                "train/gradient_norm",
                float(grad_norm),
                self.global_step,
            )
            self.writer.add_scalar(
                "train/speed_gradient_norm_after_clip",
                speed_gradient_norm,
                self.global_step,
            )
            self.global_step += 1
            self.total_sample += batch_size

            self.wandb_run.log({
                "samples_seen": self.total_sample,
                "batch/loss": output["loss"].item(),
                "batch/caption_loss": output["caption_loss"].item(),
                "batch/trajectory_loss": output["trajectory_loss"].item(),
                "batch/gradient_norm_before_clip": float(grad_norm),
                "batch/speed_gradient_norm_after_clip": speed_gradient_norm,
            })

            absolute_batch = batch_offset + batch_index + 1
            self.next_batch = absolute_batch
            if (
                self.config.log_every_steps > 0
                and (
                    self.global_step % self.config.log_every_steps == 0
                    or absolute_batch == total_batches
                )
            ):
                print(
                    f"epoch={epoch + 1}/{self.config.num_epochs} "
                    f"batch={absolute_batch}/{total_batches} "
                    f"step={self.global_step} "
                    f"frames_seen={self.total_sample} "
                    f"loss={output['loss'].item():.5f} "
                    f"caption={output['caption_loss'].item():.5f} "
                    f"trajectory={output['trajectory_loss'].item():.5f}",
                    flush=True,
                )

            if (
                self.config.use_wandb
                and self.config.training_visualization_every_steps > 0
                and self.global_step
                % self.config.training_visualization_every_steps
                == 0
                and visualization_dataset is not None
                and sample_indices is not None
            ):
                sample_offset = batch_index * self.config.batch_size
                sample = visualization_dataset[sample_indices[sample_offset]]
                self.log_training_visualization(
                    sample,
                    output["pred_trajectory"][0].detach().float().cpu(),
                )

            if (
                self.checkpoint_path is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self.save_checkpoint(
                    epoch,
                    absolute_batch,
                )

            if (
                val_loader is not None
                and self.config.validation_every_steps > 0
                and self.global_step % self.config.validation_every_steps == 0
                and absolute_batch < total_batches
            ):
                val_metrics = self.evaluate(
                    val_loader,
                    max_batches=self.config.validation_max_batches,
                )
                self.wandb_run.log({
                    "samples_seen": self.total_sample,
                    **{
                        f"validation/{key}": value
                        for key, value in val_metrics.items()
                    },
                })
                for key, value in val_metrics.items():
                    self.writer.add_scalar(
                        f"validation/{key}",
                        value,
                        self.global_step,
                    )
                print(
                    f"validation step={self.global_step} "
                    f"frames_seen={self.total_sample} "
                    f"samples={int(val_metrics['samples'])} "
                    f"loss={val_metrics['loss']:.5f} "
                    f"ADE={val_metrics['ade']:.3f} "
                    f"FDE={val_metrics['fde']:.3f}",
                    flush=True,
                )
                self.model.train()

        return {
            key: value / sample_count
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        self.model.eval()

        totals = {
            "loss": 0.0,
            "caption_loss": 0.0,
            "trajectory_loss": 0.0,
        }
        sample_count = 0
        predictions = []
        targets = []

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = self._move(batch)

            with _autocast(self.device):
                output = self.model(**batch)

            batch_size = batch["pixel_values"].shape[0]
            sample_count += batch_size

            for key in totals:
                value = output[key]
                if value is not None:
                    totals[key] += value.item() * batch_size

            predictions.append(
                output["pred_trajectory"].detach().cpu()
            )
            targets.append(
                batch["gt_trajectory"].detach().cpu()
            )

        if sample_count == 0:
            raise ValueError("Validation loader produced no samples")

        pred = torch.cat(predictions)
        target = torch.cat(targets)

        metrics = {
            key: value / sample_count
            for key, value in totals.items()
        }

        metrics["ade"] = compute_ade(pred, target)
        metrics["fde"] = compute_fde(pred, target)
        metrics["samples"] = float(sample_count)

        return metrics


def run_week2_training(config: Week2CoVLAConfig):
    if config.val_data_dir is None:
        raise ValueError("val_data_dir must point to the Mini validation dataset")

    train_root = Path(config.data_dir)
    val_root = Path(config.val_data_dir)
    train_scene_ids = get_scene_ids(train_root, config.num_scenes)
    val_scene_ids = get_scene_ids(val_root, config.num_val_scenes)
    if not train_scene_ids:
        raise ValueError(f"No training scenes found in {train_root}")
    if not val_scene_ids:
        raise ValueError(f"No validation scenes found in {val_root}")

    train_dataset = CoVLADataset(
        train_root,
        frame_interval=config.frame_interval,
        scene_ids=train_scene_ids,
    )
    val_dataset = CoVLADataset(
        val_root,
        frame_interval=config.frame_interval,
        scene_ids=val_scene_ids,
    )
    print(
        f"Training: {len(train_scene_ids)} scenes, {len(train_dataset)} frames | "
        f"Validation: {len(val_scene_ids)} scenes, {len(val_dataset)} frames",
        flush=True,
    )
    collator = Week2DataCollator(config)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )

    device = torch.device(config.device)
    model = build_model(config).to(device)
    wandb_run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        id=config.wandb_run_id,
        resume="allow" if config.wandb_run_id else None,
        allow_val_change=True,
        config=asdict(config),
        mode=None if config.use_wandb else "disabled",
    )
    wandb_run.define_metric("samples_seen")
    wandb_run.define_metric("batch/*", step_metric="samples_seen")
    wandb_run.define_metric("training_example/*", step_metric="samples_seen")
    wandb_run.define_metric("validation/*", step_metric="samples_seen")
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("train/*", step_metric="epoch")
    wandb_run.define_metric("val/*", step_metric="epoch")
    trainer = Week2VLATrainer(model, config, wandb_run, collator)
    start_epoch, start_batch = (
        trainer.load_checkpoint(config.resume_from_checkpoint)
        if config.resume_from_checkpoint
        else (0, 0)
    )

    batches_per_epoch = (
        len(train_dataset) + config.batch_size - 1
    ) // config.batch_size
    if start_batch >= batches_per_epoch:
        start_epoch += 1
        start_batch = 0

    history = []

    # Training Loop
    try:
        for epoch in range(start_epoch, config.num_epochs):
            generator = torch.Generator().manual_seed(config.seed + epoch)
            order = torch.randperm(
                len(train_dataset),
                generator=generator,
            ).tolist()
            batch_offset = start_batch if epoch == start_epoch else 0
            if batch_offset:
                order = order[batch_offset * config.batch_size:]
            train_loader = DataLoader(
                Subset(train_dataset, order),
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=config.num_workers,
                pin_memory=config.device.startswith("cuda"),
            )
            train_metrics = trainer.train_epoch(
                train_loader,
                epoch,
                batch_offset=batch_offset,
                val_loader=val_loader,
                visualization_dataset=train_dataset,
                sample_indices=order,
            )
            start_batch = 0
            val_metrics = trainer.evaluate(val_loader)

            sample = val_dataset[config.visualization_index % len(val_dataset)]
            generated_caption = generate_caption(model, sample, collator)
            prediction = predict_trajectory(
                model,
                {**sample, "caption": generated_caption},
                collator,
            )
            figure = plot_prediction(
                sample,
                prediction,
                generated_caption,
            )

            epoch_metrics = {
                "epoch": epoch + 1,
                **{
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                },
                **{
                    f"val_{key}": value
                    for key, value in val_metrics.items()
                },
            }
            history.append(epoch_metrics)

            wandb_run.log({
                "epoch": epoch + 1,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "val/generated_caption": generated_caption,
                "val/trajectory_prediction": wandb.Image(figure),
            })

            for key, value in train_metrics.items():
                trainer.writer.add_scalar(
                    f"epoch/train_{key}",
                    value,
                    epoch + 1,
                )

            for key, value in val_metrics.items():
                trainer.writer.add_scalar(
                    f"epoch/val_{key}",
                    value,
                    epoch + 1,
                )

            trainer.writer.add_figure(
                "epoch/validation_prediction",
                figure,
                epoch + 1,
            )
            plt.close(figure)

            print(epoch_metrics, flush=True)
            trainer.save_checkpoint(
                epoch + 1,
                0,
                archive_epoch=True,
            )

    except KeyboardInterrupt:
        print("Interrupted; saving the latest completed batch.", flush=True)
        trainer.save_checkpoint(
            trainer.current_epoch,
            trainer.next_batch,
        )
        raise

    finally:
        trainer.writer.close()
        wandb_run.finish()

    return model, val_dataset, collator, history
