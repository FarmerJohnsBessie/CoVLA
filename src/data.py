import json
from functools import reduce

import pandas as pd
from torch.utils.data import Dataset


def get_scene_ids(root, num=None):
    raw_data = pd.read_csv(root / "index.csv")
    cleaned_column = raw_data["video_id"].drop_duplicates(keep='first').tolist()
    return cleaned_column[:num]


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


def load_scene(root, scene_id, stride=1):
    states = read_json_records(root / "states" / f"{scene_id}.jsonl")
    captions = read_json_records(root / "captions" / f"{scene_id}.jsonl")

    assert len(states) == len(captions), "states and captions are not equal"
    
    scene = []
    for position, (state, caption) in enumerate(zip(states, captions)):
        assert state["frame_id"] == position, "frame id not aligned"

        # handle stride
        if position % stride != 0:
            continue

        image_path = root / state["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        scene.append(
            {
                "scene_id": scene_id,
                "frame_id": position,
                "state": state,
                "position" : position,
                "image_path": image_path,
            }
        )
    return scene


class CoVLADataset(Dataset):
    def __init__(self, root, frame_interval):
        self.root = root
        self.frame_interval = frame_interval
        self.scenes = reduce(
            lambda acc, x : acc.append(load_scene(self.root, scene_id=x, stride=frame_interval)), 
            get_scene_ids(self.root, 1), 
            []
        )

    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, index):
        return self.scenes[index]

