from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, CLIPVisionModel, PreTrainedModel


@dataclass
class Week2CoVLAConfig:
    """Data, model, and training settings"""

    data_dir: str = "data/covla-mini"
    num_scenes: int | None = None
    frame_interval: int = 10

    log_dir: str = "runs/week2"

    trajectory_points: int = 10
    trajectory_dim: int = 3
    image_size: int = 224
    speed_scale: float = 30.0
    vision_encoder_hf: str = "openai/clip-vit-large-patch14"
    language_model_hf: str = "mistralai/Mistral-7B-Instruct-v0.2"

    prompt: str = "Predict the ego vehicle's future trajectory."

    # --- Optimization ---
    batch_size: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 5
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_ratio: float = 0.8


def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Euclidean waypoint error across a batch."""
    return torch.linalg.vector_norm(pred - target, dim=-1).mean().item()


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Average Euclidean error at the final waypoint."""
    return torch.linalg.vector_norm(pred[:, -1] - target[:, -1], dim=-1).mean().item()


class Week2VLAModel(nn.Module):
    """Implement the CLIP, speed, language, caption, and trajectory stack here."""

    def __init__(self, 
                 config: Week2CoVLAConfig,
                 vision_encoder: CLIPVisionModel | None = None,
                 language_model: PreTrainedModel | None = None
                ) -> None:

        super().__init__()
        self.config = config

        # --- Pretrained CLIP & Mistral ---
        self.vision_encoder: CLIPVisionModel = vision_encoder or CLIPVisionModel.from_pretrained(
            config.vision_encoder_hf
        )
        self.language_model = language_model or AutoModelForCausalLM.from_pretrained(
            config.language_model_hf,
            dtype="auto",
        )

        # Freeze CLIP
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        self.vision_encoder.eval()

        # Freeze Mistral Original parameters
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        self.language_model.eval()

        # --- Trainable Layers ---
        vision_width = self.vision_encoder.config.hidden_size
        language_width = self.language_model.config.hidden_size

        self.vision_projection = nn.Linear(vision_width, language_width)
        self.speed_projection = nn.Linear(1, language_width)
        
        # Parameter (10, language_width), the 10 queries
        self.trajectory_queries = nn.Parameter(
            torch.empty(config.trajectory_points, language_width)
        )
        nn.init.normal_(self.trajectory_queries, std=0.02)

        # Trajectory MLP: (B, 10, language_width) -> (B, 10, 3)
        head_hidden = language_width // 4
        self.trajectory_head = nn.Sequential(
            nn.Linear(language_width, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, config.trajectory_dim)
        )

    def train(self, mode: bool=True):
        super().train(mode)
        self.vision_encoder.eval()
        self.language_model.eval()
        return self

    def _encode_prefix(self,
                       pixel_values: torch.Tensor, # (B, 3, 244, 244) Already condensed
                       ego_speed: torch.Tensor, # (B,) list of speed
                       prompt_ids: torch.Tensor, # (B, T_prompt)
                       prompt_mask: torch.Tensor):
        batch_size = pixel_values.shape[0]

        # Create the Image tokens
        with torch.no_grad():
            vision_output = self.vision_encoder(
                pixel_values=pixel_values, 
                return_dict=True
            ) # (B, 257, 1024)
        
        vision_tokens = vision_output.last_hidden_state[:, 1:] # Remove CLS token -> (B, 256, 1024)
        vision_tokens = self.vision_projection(vision_tokens) # (B, 256, 4096)

        # Create the speed tokens
        speed = ego_speed.reshape(batch_size, 1) # (B, 1)
        speed_tokens = self.speed_projection(speed) # (B, 4096)
        speed_tokens = speed_tokens.unsqueeze(1) # (B, 1, 4096)

        # Create the prompt tokens
        embedding_layer = self.language_model.get_input_embeddings()
        prompt_tokens = embedding_layer(prompt_ids) # (B, T_prompt, 4096)

        # Combine the tokens
        prefix_tokens = torch.cat(
            [vision_tokens, speed_tokens, prompt_tokens], dim=1
        )

        # Get the mask
        visual_mask = torch.ones(
            batch_size,
            vision_tokens.shape[1] + 1
        )
        prefix_mask = torch.cat(
            [visual_mask, prompt_mask], dim=1
        )

        return prefix_tokens, prefix_mask


    def forward(self,
                pixel_values: torch.Tensor,
                ego_speed: torch.Tensor,
                prompt_ids: torch.Tensor,
                prompt_mask: torch.Tensor,
                caption_ids: torch.Tensor,
                caption_mask: torch.Tensor,
                gt_trajectory: torch.Tensor) -> dict[str, Any]:
        """Return losses, generated-caption outputs, and predicted trajectories."""

        # --- Setup ---
        batch_size = prompt_ids.shape[0]
        prefix_tokens, prefix_mask = self._encode_prefix(pixel_values, 
                                                                          ego_speed, 
                                                                          prompt_ids, 
                                                                          prompt_mask)

        embedding_layer = self.language_model.get_input_embeddings()
        caption_tokens = embedding_layer(caption_ids) # (B, T_caption, 4096)

        query_tokens = self.trajectory_queries.unsqueeze(0) # (1, 10, 4096)
        query_tokens = query_tokens.expand(batch_size, -1, -1) # (B, 10, )

        query_mask = torch.ones(
                    batch_size,
                    self.config.trajectory_points,
                ) # (B, 10)

        # --- Finalize Input to LLM ---
        input_tokens = torch.cat(
            [prefix_tokens, caption_tokens, query_tokens]
        ) # Query after token because we need attention from caption for query

        input_mask = torch.cat(
            [prefix_mask, caption_mask, query_mask], dim=1
        )
        
        # --- Calculate Position ---
        position_ids = input_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(input_mask == 0, 0)

        # --- Calculate Labels ---
        labels = torch.full(
            input_tokens.shape[:2],
            -100
        )

        start = prefix_tokens.shape[1]
        end = start + caption_tokens.shape[1]
        labels_filtered = caption_ids.masked_fill(caption_mask == 0, -100)

        labels[:, start:end] = labels_filtered

        # --- Get LLM output ---
        language_output = self.language_model(
            input_embeds=input_tokens,
            attention_mask=input_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )
        caption_loss = language_output.loss
        hidden = language_output.hidden_states[-1]

        return {}



def build_model(config: Week2CoVLAConfig) -> Week2VLAModel:
    """Construct and return your Week2VLAModel."""
    raise NotImplementedError("Implement build_model")
