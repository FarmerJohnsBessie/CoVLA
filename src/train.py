from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoTokenizer, CLIPImageProcessor

from src.data import CoVLADataset, get_scene_ids
from src.model import (
    Week2CoVLAConfig,
    Week2VLAModel,
    build_model,
    build_tiny_model,
    compute_ade,
    compute_fde,
)


class Week2DataCollator:
    """Convert PIL dataset samples into CLIP pixels and Mistral prompt tokens."""

    def __init__(self, config: Week2CoVLAConfig) -> None:
        self.config = config
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_encoder_hf
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_hf)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        prompt = self.tokenizer(
            [self.config.prompt] * len(samples),
            padding=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.image_processor(
                images=[sample["image"] for sample in samples],
                return_tensors="pt",
            )["pixel_values"],
            "ego_speed": torch.stack([sample["speed"] for sample in samples]),
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "target_trajectory": torch.stack(
                [sample["trajectory"] for sample in samples]
            ),
        }


class Week2VLATrainer:
    def __init__(
        self,
        model: Week2VLAModel,
        config: Week2CoVLAConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        sample_count = 0

        for batch in loader:
            batch = self._move(batch)
            output = self.model(**batch)
            loss = output["loss"]
            if loss is None:
                raise ValueError("Training requires target_trajectory")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_size = batch["pixel_values"].shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

        return {"loss": total_loss / sample_count}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses = []
        predictions = []
        targets = []

        for batch in loader:
            batch = self._move(batch)
            output = self.model(**batch)
            losses.append(output["loss"].item())
            predictions.append(output["pred_trajectory"].cpu())
            targets.append(batch["target_trajectory"].cpu())

        pred = torch.cat(predictions)
        target = torch.cat(targets)
        return {
            "loss": sum(losses) / len(losses),
            "ade": compute_ade(pred, target),
            "fde": compute_fde(pred, target),
        }


def build_train_val_datasets(
    config: Week2CoVLAConfig,
) -> tuple[CoVLADataset, CoVLADataset]:
    """Split by scene so adjacent frames never leak across train and validation."""
    root = Path(config.data_dir)
    scene_ids = get_scene_ids(root, config.num_scenes)
    if len(scene_ids) < 2:
        raise ValueError("Real train/validation splitting requires at least two scenes")

    split = min(max(round(len(scene_ids) * config.train_ratio), 1), len(scene_ids) - 1)
    return (
        CoVLADataset(root, config.frame_interval, scene_ids=scene_ids[:split]),
        CoVLADataset(root, config.frame_interval, scene_ids=scene_ids[split:]),
    )


def run_week2_training(config: Week2CoVLAConfig) -> list[dict[str, float]]:
    """Run the real pretrained-backbone training path on a GPU machine."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real CLIP + Mistral training requires a CUDA GPU; run the smoke test locally"
        )
    train_dataset, val_dataset = build_train_val_datasets(config)
    collator = Week2DataCollator(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
    )
    trainer = Week2VLATrainer(build_model(config), config, device="cuda")

    history = []
    for epoch in range(config.num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
    return history


def _build_smoke_loader(
    dataset: CoVLADataset,
    config: Week2CoVLAConfig,
    sample_count: int,
) -> DataLoader:
    transform = Compose(
        [
            Resize((config.image_size, config.image_size)),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    subset = Subset(dataset, range(min(sample_count, len(dataset))))

    def collate(samples: list[dict]) -> dict[str, torch.Tensor]:
        batch_size = len(samples)
        # Arbitrary token IDs are sufficient for an offline randomly initialized LM smoke test.
        prompt_ids = torch.tensor([1, 7, 11, 19, 2], dtype=torch.long).repeat(
            batch_size, 1
        )
        return {
            "pixel_values": torch.stack([transform(sample["image"]) for sample in samples]),
            "ego_speed": torch.stack([sample["speed"] for sample in samples]),
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": torch.ones_like(prompt_ids),
            "target_trajectory": torch.stack(
                [sample["trajectory"] for sample in samples]
            ),
        }

    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )


def run_smoke_training(
    data_dir: str | Path = "data/covla-mini",
    epochs: int = 2,
    sample_count: int = 4,
) -> list[dict[str, float]]:
    """Train the tiny offline stack on a few real CoVLA samples."""
    torch.manual_seed(0)
    config = Week2CoVLAConfig(
        data_dir=str(data_dir),
        image_size=32,
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=epochs,
    )
    dataset = CoVLADataset(Path(data_dir), frame_interval=config.frame_interval, number=1)
    loader = _build_smoke_loader(dataset, config, sample_count)
    model = build_tiny_model(config)
    trainer = Week2VLATrainer(model, config)

    history = []
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(loader)
        eval_metrics = trainer.evaluate(loader)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                **{f"eval_{key}": value for key, value in eval_metrics.items()},
            }
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline CoVLA model smoke test")
    parser.add_argument("--data-dir", default="data/covla-mini")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke_training(args.data_dir, args.epochs, args.samples),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
