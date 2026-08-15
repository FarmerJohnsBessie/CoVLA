import textwrap

import matplotlib.pyplot as plt
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
    """Display image, projected trajectory, caption, and metadata."""
    # REQUIRES: The sample should be obtained from the load_scene function, 
    #           Not the dataset
    state = sample["state"] 
    pixels = project_trajectory(state)

    with Image.open(sample["image_path"]) as raw_image:
        image = raw_image.convert("RGB")

    width, height = image.size

    figure, axis = plt.subplots(figsize=(12, 8))

    axis.imshow(image)

    if len(pixels):
        axis.plot(
            pixels[:, 0],
            pixels[:, 1],
            color="#7CFC00",
            linewidth=4,
        )

    # Prevent extreme near-camera points from resizing the axes.
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")

    caption = sample["caption"]["rich_caption"]
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=100))

    figure.text(
        0.5,
        0.02,
        wrapped_caption,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    figure.subplots_adjust(
        left=0,
        right=1,
        top=0.95,
        bottom=0.18,
    )

    return figure


def plot_BEV(sample):
    state = sample["state"]
    trajectory = state["trajectory"]
    plt.plot(torch.flatten(trajectory[:, 0]), torch.flatten(trajectory[:, 1]))
    plt.show()