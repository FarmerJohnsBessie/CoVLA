from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlamaConfig,
    LlamaModel,
)


@dataclass
class Week2CoVLAConfig:
    data_dir: str = "data/covla-mini"
    num_scenes: int | None = None
    frame_interval: int = 10
    trajectory_points: int = 10
    trajectory_dim: int = 3
    image_size: int = 224
    speed_scale: float = 30.0
    vision_encoder_hf: str = "openai/clip-vit-large-patch14"
    language_model_hf: str = "mistralai/Mistral-7B-Instruct-v0.2"
    prompt: str = "Predict the ego vehicle's future trajectory."
    batch_size: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 5
    train_ratio: float = 0.8


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Euclidean waypoint error across a batch."""
    return torch.linalg.vector_norm(pred - target, dim=-1).mean().item()


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Euclidean error at the final waypoint."""
    return torch.linalg.vector_norm(pred[:, -1] - target[:, -1], dim=-1).mean().item()


class Week2VLAModel(nn.Module):
    """Frozen vision/language backbones with trainable modality adapters and action head."""

    def __init__(
        self,
        config: Week2CoVLAConfig,
        vision_encoder: nn.Module | None = None,
        language_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder or CLIPVisionModel.from_pretrained(
            config.vision_encoder_hf
        )
        self.language_model = language_model or AutoModel.from_pretrained(
            config.language_model_hf,
            dtype="auto",
        )

        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False

        vision_width = self.vision_encoder.config.hidden_size
        language_width = self.language_model.config.hidden_size
        head_width = max(language_width // 4, config.trajectory_dim)

        self.vision_projection = nn.Linear(vision_width, language_width)
        self.speed_projection = nn.Linear(1, language_width)
        self.trajectory_queries = nn.Parameter(
            torch.empty(config.trajectory_points, language_width)
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(language_width, head_width),
            nn.GELU(),
            nn.Linear(head_width, config.trajectory_dim),
        )
        nn.init.normal_(self.trajectory_queries, std=0.02)

    def train(self, mode: bool = True) -> Week2VLAModel:
        super().train(mode)
        self.vision_encoder.eval()
        self.language_model.eval()
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        ego_speed: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
        target_trajectory: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = pixel_values.shape[0]

        with torch.no_grad():
            # CLIP emits [CLS] followed by one token per image patch.
            vision_tokens = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True,
            ).last_hidden_state[:, 1:]
        vision_tokens = self.vision_projection(vision_tokens)

        speed = ego_speed.reshape(batch_size, 1) / self.config.speed_scale
        speed_token = self.speed_projection(speed).unsqueeze(1)

        prompt_tokens = self.language_model.get_input_embeddings()(prompt_input_ids)
        query_tokens = self.trajectory_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Pretrained Mistral commonly loads as bfloat16 while the small trainable
        # adapters stay float32. Cast only at the backbone boundaries.
        language_dtype = prompt_tokens.dtype
        vision_tokens = vision_tokens.to(language_dtype)
        speed_token = speed_token.to(language_dtype)
        query_tokens = query_tokens.to(language_dtype)

        inputs_embeds = torch.cat(
            [vision_tokens, speed_token, prompt_tokens, query_tokens],
            dim=1,
        )

        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_input_ids)
        prefix_length = vision_tokens.shape[1] + 1
        prefix_mask = torch.ones(
            batch_size,
            prefix_length,
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        query_mask = torch.ones(
            batch_size,
            self.config.trajectory_points,
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        attention_mask = torch.cat(
            [prefix_mask, prompt_attention_mask, query_mask],
            dim=1,
        )

        hidden = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        query_hidden = hidden[:, -self.config.trajectory_points :].to(
            self.trajectory_head[0].weight.dtype
        )
        pred_trajectory = self.trajectory_head(query_hidden)

        loss = None
        if target_trajectory is not None:
            expected_shape = (
                batch_size,
                self.config.trajectory_points,
                self.config.trajectory_dim,
            )
            if target_trajectory.shape != expected_shape:
                raise ValueError(
                    f"Expected target trajectory {expected_shape}, "
                    f"got {tuple(target_trajectory.shape)}"
                )
            loss = F.mse_loss(pred_trajectory, target_trajectory)

        return {
            "loss": loss,
            "pred_trajectory": pred_trajectory,
        }


def build_model(config: Week2CoVLAConfig) -> Week2VLAModel:
    """Load the real pretrained CLIP and Mistral backbones."""
    return Week2VLAModel(config)


def build_tiny_model(config: Week2CoVLAConfig) -> Week2VLAModel:
    """Build an offline model that exercises the same tensor path on a laptop."""
    vision = CLIPVisionModel(
        CLIPVisionConfig(
            image_size=config.image_size,
            patch_size=16,
            num_channels=3,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
    )
    language = LlamaModel(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    return Week2VLAModel(config, vision_encoder=vision, language_model=language)
