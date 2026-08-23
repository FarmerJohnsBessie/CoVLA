from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

from src.data import CoVLADataset
from src.model import Week2CoVLAConfig, Week2VLAModel


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

        self.prompt_text = self.tokenizer.apply_chat_template(
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
            "pixel_vales": pixel_values,
            "ego_speed": torch.stack(
                [sample["speed"] for sample in samples]
            ),
            "prompt_ids": self.prompt_ids.unsqueeze(0).repeat(batch_size,1),
            "prmopt_mask": self.prompt_mask.unsqueeze(0).repeat(batch_size,1),
            "caption_ids": caption_input_ids,
            "caption_mask": caption_attention_mask,
            "gt_trajectory": torch.stack(
                [sample["trajectory"] for sample in samples]
            )
        }


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


    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()


    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        raise NotImplementedError("Implement Week2VLATrainer.evaluate")


def build_train_val_datasets(
    config: Week2CoVLAConfig,
) -> tuple[CoVLADataset, CoVLADataset]:
    """Split scenes, then construct separate training and validation datasets."""
    raise NotImplementedError("Implement build_train_val_datasets")


def run_week2_training(config: Week2CoVLAConfig) -> list[dict[str, float]]:
    """Build datasets, loaders, model, and trainer, then run the epoch loop."""
    raise NotImplementedError("Implement run_week2_training")
