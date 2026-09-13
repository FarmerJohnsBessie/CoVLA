from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, CLIPImageProcessor

from src.data import CoVLADataset, get_scene_ids
from src.model import Week2CoVLAConfig, Week2VLAModel, build_model
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

# =============== Seperate Cell ===============
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


@torch.inference_mode()
def predict_trajectory(
    model: Week2VLAModel,
    sample: dict[str, Any],
    collator: Week2DataCollator,
) -> torch.Tensor:
    """Predict one sample's trajectory on the model's current device."""
    model.eval()
    device = next(model.parameters()).device
    batch = {
        key: value.to(device)
        for key, value in collator([sample]).items()
    }
    with _autocast(device):
        output = model(**batch)
    return output["pred_trajectory"][0].float().cpu()


@torch.inference_mode()
def find_worst_scenes(
    model: Week2VLAModel,
    dataset: Dataset,
    collator: Week2DataCollator,
    count: int = 10,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Rank scenes by mean teacher-forced ADE, retaining each worst frame."""
    model.eval()
    device = next(model.parameters()).device
    results = []
    offset = 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            output = model(**moved)

        prediction = output["pred_trajectory"].float().cpu()
        target = batch["gt_trajectory"]
        errors = torch.linalg.vector_norm(prediction - target, dim=-1)

        for row in range(len(prediction)):
            sample = dataset[offset + row]
            results.append({
                "index": offset + row,
                "scene_id": sample["scene_id"],
                "frame_id": sample["frame_id"],
                "ade": errors[row].mean().item(),
                "fde": errors[row, -1].item(),
                "pred_trajectory": prediction[row],
            })
        offset += len(prediction)

    by_scene = {}
    for result in results:
        by_scene.setdefault(result["scene_id"], []).append(result)

    ranked = []
    for scene_id, scene_results in by_scene.items():
        worst_frame = max(scene_results, key=lambda result: result["ade"])
        ranked.append({
            **worst_frame,
            "scene_id": scene_id,
            "scene_ade": sum(row["ade"] for row in scene_results)
            / len(scene_results),
        })

    return sorted(
        ranked,
        key=lambda result: result["scene_ade"],
        reverse=True,
    )[:count]


def measure_speed_sensitivity(
    model: Week2VLAModel,
    sample: dict[str, Any],
    collator: Week2DataCollator,
    speeds: tuple[float, ...] = (0.0, 3.0, 7.0),
) -> list[dict[str, Any]]:
    """Measure how one fixed scene's outputs change with only speed varied."""
    results = []
    for speed in speeds:
        counterfactual = {
            **sample,
            "speed": torch.tensor(speed, dtype=torch.float32),
        }
        prediction = predict_trajectory(model, counterfactual, collator)
        results.append({
            "speed_mps": speed,
            "predicted_final_distance_m": torch.linalg.vector_norm(
                prediction[-1, :2]
            ).item(),
            "generated_caption": generate_caption(
                model,
                counterfactual,
                collator,
            ),
        })
    return results
# =============== Seperate Cell ===============


class Week2VLATrainer:
    """Implement optimization, validation, metrics, and logging here."""

    def __init__(
        self,
        model: Week2VLAModel,
        config: Week2CoVLAConfig,
        wandb_run: Any,
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
        self.writer = SummaryWriter(config.log_dir)
        self.device = torch.device(self.config.device)
        self.wandb_run = wandb_run

    @property
    def checkpoint_path(self) -> Path | None:
        if self.config.checkpoint_dir is None:
            return None
        return Path(self.config.checkpoint_dir) / "latest.pt"

    def save_checkpoint(self, epoch: int, next_batch: int) -> None:
        """Atomically save only trainable weights and optimizer state."""
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
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self) -> tuple[int, int]:
        """Restore a compact checkpoint and return (epoch, next_batch)."""
        path = self.checkpoint_path
        if path is None or not path.is_file():
            return 0, 0
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("batch_size") != self.config.batch_size:
            raise ValueError("Checkpoint batch_size does not match the config")
        if state.get("seed") != self.config.seed:
            raise ValueError("Checkpoint seed does not match the config")
        self.model.load_state_dict(state["model"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
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

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        batch_offset: int = 0,
    ) -> dict[str, float]:
        self.model.train()
        totals = {
            "loss": 0.0,
            "caption_loss": 0.0,
            "trajectory_loss": 0.0,
        }
        sample_count = 0

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

            if (
                self.checkpoint_path is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self.save_checkpoint(
                    epoch,
                    batch_offset + batch_index + 1,
                )


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

        pred = torch.cat(predictions)
        target = torch.cat(targets)

        metrics = {
            key: value / sample_count
            for key, value in totals.items()
        }

        metrics["ade"] = compute_ade(pred, target)
        metrics["fde"] = compute_fde(pred, target)

        return metrics


def build_train_val_datasets(config: Week2CoVLAConfig) -> tuple[Dataset, Dataset]:
    """Split scenes, then construct separate training and validation datasets."""
    root = Path(config.data_dir)
    scene_ids = get_scene_ids(root, config.num_scenes)

    if not scene_ids:
        raise ValueError(f"No scenes found in {root}")

    if config.val_data_dir is not None:
        val_root = Path(config.val_data_dir)
        val_scene_ids = get_scene_ids(val_root)
        if not val_scene_ids:
            raise ValueError(f"No validation scenes found in {val_root}")
        return (
            CoVLADataset(
                root,
                frame_interval=config.frame_interval,
                scene_ids=scene_ids,
            ),
            CoVLADataset(
                val_root,
                frame_interval=config.frame_interval,
                scene_ids=val_scene_ids,
            ),
        )

    if len(scene_ids) == 1:
        dataset = CoVLADataset(
            root,
            frame_interval=config.frame_interval,
            scene_ids=scene_ids,
        )
        if len(dataset) < 2:
            raise ValueError("Training and validation require at least two samples")

        split = round(len(dataset) * config.train_ratio)
        split = min(max(split, 1), len(dataset) - 1)
        print("Only one scene found; using a frame split for the local smoke test.")
        return (
            Subset(dataset, range(split)),
            Subset(dataset, range(split, len(dataset))),
        )

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
        config=asdict(config),
        mode=None if config.use_wandb else "disabled",
    )
    wandb_run.define_metric("samples_seen")
    wandb_run.define_metric("batch/*", step_metric="samples_seen")
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("train/*", step_metric="epoch")
    wandb_run.define_metric("val/*", step_metric="epoch")
    trainer = Week2VLATrainer(model, config, wandb_run)
    start_epoch, start_batch = (
        trainer.load_checkpoint()
        if config.resume_from_checkpoint
        else (0, 0)
    )

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

            print(epoch_metrics)
            trainer.save_checkpoint(epoch + 1, 0)

    finally:
        trainer.writer.close()
        wandb_run.finish()

    return model, val_dataset, collator, history


def main() -> None:
    torch.manual_seed(42)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    config = Week2CoVLAConfig(
        device=device,
        num_scenes=2,
        num_epochs=1,
        batch_size=1,
        num_workers=0 if device != "cuda" else 2,
        use_wandb=device == "cuda",
    )

    if device != "cuda":
        config.vision_encoder_hf = (
            "optimum-intel-internal-testing/tiny-random-CLIPModel"
        )
        config.language_model_hf = (
            "HuggingFaceM4/tiny-random-MistralForCausalLM"
        )
        config.frame_interval = 100
        config.max_caption_tokens = 32
        config.learning_rate = 1e-3
        print("No CUDA device found; using tiny random models for a local smoke test.")

    model, val_dataset, collator, _ = run_week2_training(config)
    worst = find_worst_scenes(
        model,
        val_dataset,
        collator,
        count=10,
        batch_size=config.batch_size,
    )

    output_dir = Path(config.log_dir) / "worst_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, result in enumerate(worst, start=1):
        sample = val_dataset[result["index"]]
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
        errors = torch.linalg.vector_norm(
            prediction - sample["trajectory"],
            dim=-1,
        )
        path = output_dir / f"{rank:02d}_index_{result['index']}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(
            f"Worst scene #{rank}: {result['scene_id']} "
            f"(worst frame {result['frame_id']}, index {result['index']}), "
            f"scene ADE={result['scene_ade']:.2f} m, "
            f"worst-frame teacher-forced ADE={result['ade']:.2f} m, "
            f"end-to-end ADE={errors.mean().item():.2f} m, "
            f"FDE={errors[-1].item():.2f} m -> {path}"
        )

    print("Speed sensitivity:")
    for result in measure_speed_sensitivity(model, val_dataset[0], collator):
        print(result)


if __name__ == "__main__":
    main()
