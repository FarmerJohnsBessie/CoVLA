from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPImageProcessor

from src.data import CoVLADataset, get_scene_ids
from src.model import Week2CoVLAConfig, Week2VLAModel

use_wandb = True
wandb_api = "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"

if use_wandb:
  wandb.login(key=wandb_api)
  wandb.init(project="CoVLA", name="mini-training")
else:
  wandb.init(mode='disabled')  


def compute_ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean Euclidean distance over all waypoints and batch elements (meters if data is in m)."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    dist = torch.sqrt(((pred - gt) ** 2).sum(dim=-1))
    return float(dist.mean().item())


def compute_fde(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean Euclidean distance at the final waypoint."""
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    final = torch.sqrt(((pred[:, -1, :] - gt[:, -1, :]) ** 2).sum(dim=-1))
    return float(final.mean().item())

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

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": config.prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

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


@torch.inference_mode()
def generate_caption(
    model: Week2VLAModel,
    sample: dict[str, Any],
    collator: Week2DataCollator,
    max_new_tokens: int = 64,
) -> str:
    """Generate a caption for one dataset sample from its image and speed."""
    model.eval()
    device = next(model.parameters()).device
    batch = {
        key: value.to(device)
        for key, value in collator([sample]).items()
    }

    prefix_tokens, prefix_mask = model._encode_prefix(
        batch["pixel_values"],
        batch["ego_speed"],
        batch["prompt_ids"],
        batch["prompt_mask"],
    )
    language_dtype = next(model.language_model.parameters()).dtype
    generated_ids = model.language_model.generate(
        inputs_embeds=prefix_tokens.to(language_dtype),
        attention_mask=prefix_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=collator.tokenizer.eos_token_id,
        pad_token_id=collator.tokenizer.pad_token_id,
    )
    return collator.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()


class Week2VLATrainer:
    """Implement optimization, validation, metrics, and logging here."""

    def __init__(self, model: Week2VLAModel, config: Week2CoVLAConfig) -> None:
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
        self.writer = SummaryWriter(config.log_dir)
        self.device = torch.device(self.config.device) 



    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device)
            for key, value in batch.items()
        }

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "caption_loss": 0.0,
            "trajectory_loss": 0.0,
        }
        sample_count = 0

        for batch in loader:
            batch = self._move(batch)

            self.optimizer.zero_grad()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output = self.model(**batch)
                loss = output["loss"]

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                self.config.max_grad_norm,
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
            self.global_step += 1
            self.total_sample += batch_size

            wandb.log({
                "samples_seen": self.total_sample,
                "batch/loss": output["loss"].item(),
                "batch/caption_loss": output["caption_loss"].item(),
                "batch/trajectory_loss": output["trajectory_loss"].item(),
                "batch/gradient_norm_before_clip": float(grad_norm),
            })


        return {
            key: value / sample_count
            for key, value in totals.items()
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()

        totals = {
            "loss": 0.0,
            "caption_loss": 0.0,
            "trajectory_loss": 0.0,
        }
        sample_count = 0
        predictions = []
        targets = []

        for batch in loader:
            batch = self._move(batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
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

        pred = torch.cat(predictions)
        target = torch.cat(targets)

        metrics = {
            key: value / sample_count
            for key, value in totals.items()
        }

        metrics["ade"] = compute_ade(pred, target)
        metrics["fde"] = compute_fde(pred, target)

        return metrics


def build_train_val_datasets(config: Week2CoVLAConfig) -> tuple[CoVLADataset, CoVLADataset]:
    """Split scenes, then construct separate training and validation datasets."""
    root = Path(config.data_dir)
    scene_ids = get_scene_ids(root, config.num_scenes)

    if len(scene_ids) < 2:
        raise ValueError(
            "Training and validation require at least two scenes"
        )

    # random.Random(42).shuffle(scene_ids)

    split = round(len(scene_ids) * config.train_ratio)
    split = min(max(split, 1), len(scene_ids) - 1)

    train_scene_ids = scene_ids[:split]
    val_scene_ids = scene_ids[split:]

    train_dataset = CoVLADataset(
        root,
        frame_interval=config.frame_interval,
        scene_ids=train_scene_ids,
    )

    val_dataset = CoVLADataset(
        root,
        frame_interval=config.frame_interval,
        scene_ids=val_scene_ids,
    )

    return train_dataset, val_dataset


def run_week2_training(config: Week2CoVLAConfig):
    train_dataset, val_dataset = build_train_val_datasets(config)
    collator = Week2DataCollator(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device(config.device)
    model = build_model(config).to(device)
    trainer = Week2VLATrainer(model, config)

    history = []

    try:
        for epoch in range(config.num_epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            # Visualize one sample
            sample = val_dataset[0]
            batch = trainer._move(collator([sample]))
            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                output = model(**batch)

            figure = plot_prediction(
                sample,
                output["pred_trajectory"][0],
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

            # Same logging, with prediction image
            wandb.log({
                "epoch": epoch + 1,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
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

            plt.close(figure)

            print(epoch_metrics)

    finally:
        trainer.writer.close()

    return model, val_dataset, collator, history
