from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import torch

from src.model import Week2CoVLAConfig
from src.train import run_week2_training


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    defaults = Week2CoVLAConfig()
    parser = argparse.ArgumentParser(
        description="Train CoVLA with a separate Mini validation set.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name used for the run folder and W&B run.",
    )
    parser.add_argument("--data-dir", required=True, help="Full training dataset.")
    parser.add_argument(
        "--val-data-dir",
        required=True,
        help="Separate Mini-50 validation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Run folders are stored under OUTPUT_DIR/RUN_NAME.",
    )
    parser.add_argument("--num-scenes", type=positive_int)
    parser.add_argument("--num-val-scenes", type=positive_int)
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=defaults.frame_interval,
    )

    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=defaults.num_epochs,
        help="Total target epoch count, including already completed epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=defaults.batch_size,
    )
    parser.add_argument(
        "--num-workers",
        type=nonnegative_int,
        default=defaults.num_workers,
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=defaults.learning_rate,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=positive_float,
        default=defaults.max_grad_norm,
    )
    parser.add_argument(
        "--max-caption-tokens",
        type=positive_int,
        default=defaults.max_caption_tokens,
    )
    parser.add_argument(
        "--speed-scale",
        type=positive_float,
        default=defaults.speed_scale,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, mps, or cpu",
    )

    parser.add_argument(
        "--checkpoint-every-steps",
        type=nonnegative_int,
        default=defaults.checkpoint_every_steps,
        help="Save latest.pt every N optimizer steps; 0 disables mid-epoch saves.",
    )
    parser.add_argument(
        "--validation-every-steps",
        type=nonnegative_int,
        default=defaults.validation_every_steps,
        help="Run periodic validation every N optimizer steps; 0 disables it.",
    )
    parser.add_argument(
        "--validation-max-batches",
        type=nonnegative_int,
        default=defaults.validation_max_batches,
        help="Periodic validation batch limit; 0 uses the full Mini set.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=nonnegative_int,
        default=defaults.log_every_steps,
        help="Print training progress every N steps; 0 disables batch printing.",
    )
    parser.add_argument(
        "--visualize-every-steps",
        type=nonnegative_int,
        default=defaults.training_visualization_every_steps,
        help="Upload one training sample to W&B every N steps; 0 disables it.",
    )

    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        action="store_true",
        help="Resume this run from its checkpoints/latest.pt.",
    )
    resume.add_argument(
        "--resume-from",
        type=Path,
        help="Load a specific checkpoint, then save into this named run.",
    )

    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_wandb,
    )
    parser.add_argument("--wandb-project", default=defaults.wandb_project)
    parser.add_argument("--vision-encoder", default=defaults.vision_encoder_hf)
    parser.add_argument("--language-model", default=defaults.language_model_hf)
    parser.add_argument("--prompt", default=defaults.prompt)
    parser.add_argument(
        "--visualization-index",
        type=nonnegative_int,
        default=defaults.visualization_index,
    )
    return parser


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = Path(args.output_dir) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.pt"
    resume_from = latest_checkpoint if args.resume else args.resume_from

    if resume_from is not None and not resume_from.is_file():
        parser.error(f"checkpoint does not exist: {resume_from}")
    if latest_checkpoint.exists() and not args.resume:
        parser.error(
            f"run {args.run_name!r} already has a checkpoint; "
            "use --resume or choose another --run-name"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_id_path = run_dir / "wandb_run_id.txt"
    wandb_run_id = None
    if args.wandb:
        if wandb_id_path.exists():
            wandb_run_id = wandb_id_path.read_text(encoding="utf-8").strip()
            if not wandb_run_id:
                parser.error(f"W&B run ID file is empty: {wandb_id_path}")
        else:
            wandb_run_id = uuid.uuid4().hex
            wandb_id_path.write_text(wandb_run_id + "\n", encoding="utf-8")

    config = Week2CoVLAConfig(
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        num_scenes=args.num_scenes,
        num_val_scenes=args.num_val_scenes,
        frame_interval=args.frame_interval,
        log_dir=str(run_dir / "tensorboard"),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint=(
            str(resume_from) if resume_from is not None else None
        ),
        validation_every_steps=args.validation_every_steps,
        validation_max_batches=args.validation_max_batches or None,
        log_every_steps=args.log_every_steps,
        training_visualization_every_steps=args.visualize_every_steps,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_run_id=wandb_run_id,
        visualization_index=args.visualization_index,
        speed_scale=args.speed_scale,
        vision_encoder_hf=args.vision_encoder,
        language_model_hf=args.language_model,
        prompt=args.prompt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_caption_tokens=args.max_caption_tokens,
        device=resolve_device(args.device),
        max_grad_norm=args.max_grad_norm,
    )

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)
        file.write("\n")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Device: {config.device}", flush=True)
    run_week2_training(config)
    print(f"Training complete. Latest checkpoint: {latest_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
