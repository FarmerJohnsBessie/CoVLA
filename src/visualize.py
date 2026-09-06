import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def project_trajectory(state):
    """Convert the raw trajectory into image pixels."""
    trajectory = torch.as_tensor(
        state["trajectory"],
        dtype=torch.float64,
    )
    extrinsic = torch.as_tensor(
        state["extrinsic_matrix"],
        dtype=torch.float64,
    )
    intrinsic = torch.as_tensor(
        state["intrinsic_matrix"],
        dtype=torch.float64,
    )

    ones = torch.ones((len(trajectory), 1), dtype=trajectory.dtype)
    trajectory_extended = torch.cat([trajectory, ones], dim=1) # N x 4

    camera_points = (extrinsic @ trajectory_extended.T).T[:, :3] # N x 3

    projected = (intrinsic @ camera_points.T).T # N x 3
    # (x, y, depth)

    pixels = projected[:, :2] / projected[:, 2:3]

    return pixels


def plot_sample(sample):
    """Display image + projected trajectory and GT trajectory in BEV."""

    state = sample["state"]
    pixels = project_trajectory(state)

    with Image.open(sample["image_path"]) as raw_image:
        image = raw_image.convert("RGB")

    width, height = image.size

    # Two plots side-by-side
    figure, (image_axis, bev_axis) = plt.subplots(
        1, 2,
        figsize=(16, 7),
    )

    # ============================================================
    # LEFT: Camera image + projected GT trajectory
    # ============================================================

    image_axis.imshow(image)

    if len(pixels):
        image_axis.plot(
            pixels[:, 0],
            pixels[:, 1],
            color="#7CFC00",
            linewidth=4,
        )

    image_axis.set_xlim(0, width)
    image_axis.set_ylim(height, 0)
    image_axis.axis("off")
    image_axis.set_title("Camera View")

    # ============================================================
    # RIGHT: GT Bird's Eye View
    # ============================================================

    gt_trajectory = state["trajectory"] 
    gt_trajectory = np.asarray(
        state["trajectory"],
        dtype=float,
    )

    bev_axis.plot(
        gt_trajectory[:, 0],
        gt_trajectory[:, 1],
        marker="o",
        label="GT",
    )

    # Ego vehicle starts at the origin
    bev_axis.scatter(
        0,
        0,
        marker="*",
        s=200,
        label="Ego",
    )

    bev_axis.set_title("Bird's Eye View")
    bev_axis.set_xlabel("Forward (m)")
    bev_axis.set_ylabel("Lateral (m)")
    bev_axis.grid(True)
    bev_axis.legend()
    bev_axis.axis("equal")

    # ============================================================
    # Caption
    # ============================================================

    caption = sample["caption"]["rich_caption"]
    wrapped_caption = "\n".join(
        textwrap.wrap(caption, width=120)
    )

    figure.text(
        0.5,
        0.02,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    figure.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.18,
        wspace=0.15,
    )

    return figure


def plot_prediction(sample, predicted_trajectory):
    """Plot one dataset sample and its predicted and ground-truth BEV paths."""
    prediction = torch.as_tensor(predicted_trajectory).detach().cpu().numpy()
    ground_truth = torch.as_tensor(sample["trajectory"]).detach().cpu().numpy()

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match "
            f"ground truth shape {ground_truth.shape}"
        )

    figure, (image_axis, bev_axis) = plt.subplots(1, 2, figsize=(16, 7))

    image_axis.imshow(sample["image"])
    image_axis.axis("off")
    image_axis.set_title(
        f"Camera View — scene {sample['scene_id']}, frame {sample['frame_id']}"
    )

    bev_axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        marker="o",
        label="Ground truth",
    )
    bev_axis.plot(
        prediction[:, 0],
        prediction[:, 1],
        marker="o",
        label="Prediction",
    )
    bev_axis.scatter(0, 0, marker="*", s=200, label="Ego")
    bev_axis.set_title("Bird's Eye View")
    bev_axis.set_xlabel("Forward (m)")
    bev_axis.set_ylabel("Lateral (m)")
    bev_axis.grid(True)
    bev_axis.legend()
    bev_axis.axis("equal")

    caption = "\n".join(textwrap.wrap(sample["caption"], width=120))
    figure.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    figure.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.18,
        wspace=0.15,
    )
    return figure
