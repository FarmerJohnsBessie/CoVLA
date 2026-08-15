import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def get_scene_ids(root, number=None):
    scene_ids = sorted(
        path.stem
        for path in (root / "states").glob("*.jsonl")
    )
    return scene_ids[:number]


def read_json_records(path):
    with open(path, "r", encoding="utf-8") as file:
        raw_data = file.read()

        decoder = json.JSONDecoder()
        position = 0
        length = len(raw_data)

        result = []

        while position < length:
            if raw_data[position].isspace():
                position += 1
                continue
            try:
                obj, position = decoder.raw_decode(s=raw_data, idx=position)
                result.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse {path} at character {position}"
                ) from e
        return result


def load_scene(root, scene_id):
    states = read_json_records(root / "states" / f"{scene_id}.jsonl")
    captions = read_json_records(root / "captions" / f"{scene_id}.jsonl")

    assert len(states) == len(captions), "states and captions are not equal"
    
    scene = []
    for position, (state, caption) in enumerate(zip(states, captions)):
        assert state["frame_id"] == position, "frame id not aligned"

        image_path = root / state["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        scene.append(
            {
                "scene_id": scene_id,
                "frame_id": position,
                "state": state,
                "caption" : caption,
                "image_path": image_path,
            }
        )
    return scene

def sample_trajectory(raw_trajectory, number=10):
    trajectory = torch.as_tensor(
        data=raw_trajectory,
        dtype=torch.float32
    )

    if trajectory.shape != (60, 3):
        raise ValueError(
            f"Expected trajectory shape (60, 3), got {trajectory.shape}"
        )

    if not torch.isfinite(trajectory).all():
        raise ValueError("Trajectory contains NaN or infinity")

    indices = torch.linspace(
        0,
        len(trajectory) - 1,
        steps=number,
    ).round().long()

    return trajectory[indices]


class CoVLADataset(Dataset):
    def __init__(self, root, frame_interval=10, scene_ids=None, number=None):
        self.root = Path(root)
        self.frame_interval = frame_interval

        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")

        # sample is a bundle of state, caption and image path.
        self.sample = [] # list of samples loaded
        selected_scene_ids = scene_ids or get_scene_ids(self.root, number)
        for scene_id in selected_scene_ids:
            scene = load_scene(self.root, scene_id=scene_id)
            self.sample.extend(
                sample 
                for sample in scene
                if sample["frame_id"] % self.frame_interval == 0
                and sample["state"]["trajectory_count"] == 60
            )

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        scene = self.sample[index]
        state = scene["state"]

        with Image.open(scene["image_path"]) as raw_image:
            image = raw_image.convert("RGB")

        return {
            "image": image,
            "speed": torch.tensor(
                state["ego_state"]["vEgo"],
                dtype=torch.float32,
            ),
            "caption": scene["caption"]["rich_caption"],
            "trajectory": sample_trajectory(state["trajectory"]),
            "scene_id": scene["scene_id"],
            "frame_id": scene["frame_id"],
        }
