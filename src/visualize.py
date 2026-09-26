import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from PIL import Image


def project_trajectory(trajectory, extrinsic_matrix, intrinsic_matrix):
    """Project ego-frame trajectory points into camera-image pixels."""
    trajectory = torch.as_tensor(
        trajectory,
        dtype=torch.float64,
    )
    extrinsic = torch.as_tensor(
        extrinsic_matrix,
        dtype=torch.float64,
    )
    intrinsic = torch.as_tensor(
        intrinsic_matrix,
        dtype=torch.float64,
    )

    ones = torch.ones((len(trajectory), 1), dtype=trajectory.dtype)
    trajectory_extended = torch.cat([trajectory, ones], dim=1) # N x 4

    camera_points = (extrinsic @ trajectory_extended.T).T[:, :3]
    projected = (intrinsic @ camera_points.T).T
    pixels = torch.full((len(trajectory), 2), torch.nan, dtype=trajectory.dtype)
    visible = camera_points[:, 2] > 1e-6
    pixels[visible] = projected[visible, :2] / projected[visible, 2:3]

    return pixels


def plot_sample(sample):
    """Display image + projected trajectory and GT trajectory in BEV."""

    state = sample["state"]
    pixels = project_trajectory(
        state["trajectory"],
        state["extrinsic_matrix"],
        state["intrinsic_matrix"],
    )

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


def plot_prediction(
    sample,
    predicted_trajectory,
    generated_caption=None,
    figure=None,
):
    """Plot camera/BEV trajectories, captions, speed, ADE, and FDE."""
    prediction = (
        torch.as_tensor(predicted_trajectory)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    ground_truth = torch.as_tensor(sample["trajectory"]).detach().cpu().numpy()

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match "
            f"ground truth shape {ground_truth.shape}"
        )

    if figure is None:
        figure = plt.figure(figsize=(16, 7))
    else:
        figure.clear()
    image_axis, bev_axis = figure.subplots(1, 2)

    image_axis.imshow(sample["image"])
    gt_pixels = project_trajectory(
        ground_truth,
        sample["extrinsic_matrix"],
        sample["intrinsic_matrix"],
    )
    pred_pixels = project_trajectory(
        prediction,
        sample["extrinsic_matrix"],
        sample["intrinsic_matrix"],
    )
    image_axis.plot(
        gt_pixels[:, 0],
        gt_pixels[:, 1],
        color="#7CFC00",
        linewidth=3,
        marker="o",
        label="Ground truth",
    )
    image_axis.plot(
        pred_pixels[:, 0],
        pred_pixels[:, 1],
        color="#FF8C00",
        linewidth=3,
        marker="o",
        label="Prediction",
    )
    image_axis.set_xlim(0, sample["image"].width)
    image_axis.set_ylim(sample["image"].height, 0)
    image_axis.axis("off")
    image_axis.legend(loc="upper right")
    image_axis.set_title(
        f"Camera View — scene {sample['scene_id']}, frame {sample['frame_id']}"
    )

    bev_axis.plot(
        ground_truth[:, 0],
        ground_truth[:, 1],
        color="#7CFC00",
        marker="o",
        label="Ground truth",
    )
    bev_axis.plot(
        prediction[:, 0],
        prediction[:, 1],
        color="#FF8C00",
        marker="o",
        label="Prediction",
    )
    bev_axis.scatter(0, 0, marker="*", s=200, label="Ego")
    bev_axis.set_title("Bird's Eye View")
    bev_axis.set_xlabel("Forward (m)")
    bev_axis.set_ylabel("Lateral (m)")
    bev_axis.grid(True)
    bev_axis.legend()
    bev_axis.set_aspect("equal", adjustable="box")
    all_points = np.concatenate([ground_truth, prediction])
    x_margin = max(2.0, np.ptp(all_points[:, 0]) * 0.1)
    y_margin = max(2.0, np.ptp(all_points[:, 1]) * 0.1)
    bev_axis.set_xlim(
        all_points[:, 0].min() - x_margin,
        all_points[:, 0].max() + x_margin,
    )
    bev_axis.set_ylim(
        all_points[:, 1].min() - y_margin,
        all_points[:, 1].max() + y_margin,
    )

    errors = np.linalg.norm(prediction - ground_truth, axis=-1)
    speed = float(torch.as_tensor(sample["speed"]).item())
    figure.suptitle(
        f"Speed: {speed:.2f} m/s  |  ADE: {errors.mean():.2f} m  |  "
        f"FDE: {errors[-1]:.2f} m"
    )

    ground_truth_caption = "\n".join(
        textwrap.wrap(f"Ground truth: {sample['caption']}", width=150)
    )
    generated = generated_caption or "(not generated)"
    generated_caption_text = "\n".join(
        textwrap.wrap(f"Generated: {generated}", width=150)
    )
    figure.text(
        0.5,
        0.075,
        ground_truth_caption,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#228B22",
    )
    figure.text(
        0.5,
        0.02,
        generated_caption_text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#B35A00",
    )
    figure.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.88,
        bottom=0.24,
        wspace=0.15,
    )
    return figure


def animate_scene(
    dataset,
    scene_index,
    predict_fn,
    caption_fn=None,
    fps=6,
    save_path=None,
):
    """Animate every sampled frame in one scene using ``plot_prediction``."""
    if scene_index < 0:
        raise IndexError("scene_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")

    samples = []
    current_scene_id = None
    current_scene_index = -1
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        if sample["scene_id"] != current_scene_id:
            current_scene_id = sample["scene_id"]
            current_scene_index += 1
        if current_scene_index == scene_index:
            samples.append(sample)
        elif current_scene_index > scene_index:
            break

    if not samples:
        raise IndexError(f"scene_index {scene_index} is outside this dataset")

    frames = []
    for sample in samples:
        caption = caption_fn(sample) if caption_fn else None
        prediction_sample = (
            {**sample, "caption": caption}
            if caption is not None
            else sample
        )
        frames.append((sample, predict_fn(prediction_sample), caption))

    figure = plt.figure(figsize=(16, 7))

    def draw(frame_index):
        plot_prediction(*frames[frame_index], figure=figure)
        return figure.axes

    video = FuncAnimation(
        figure,
        draw,
        frames=len(frames),
        interval=1000 / fps,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    if save_path is not None:
        if str(save_path).lower().endswith(".gif"):
            video.save(save_path, writer="pillow", fps=fps)
        else:
            video.save(save_path, fps=fps)
    return video
