from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.data import CoVLADataset
from src.model import Week2CoVLAConfig, Week2VLAModel


class Week2DataCollator:
    """Convert dataset samples into a padded multimodal model batch."""

    def __init__(self, config: Week2CoVLAConfig) -> None:
        self.config = config

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError("Implement Week2DataCollator.__call__")


class Week2VLATrainer:
    """Implement optimization, validation, metrics, and logging here."""

    def __init__(self, model: Week2VLAModel, config: Week2CoVLAConfig) -> None:
        self.model = model
        self.config = config

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        raise NotImplementedError("Implement Week2VLATrainer.train_epoch")

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
