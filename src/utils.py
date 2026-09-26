from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.model import Week2VLAModel

if TYPE_CHECKING:
    from src.train import Week2DataCollator


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
