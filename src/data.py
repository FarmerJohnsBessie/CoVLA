from src import DATA_ROOT
import json

def read_json_records(path):
    with open(path, "r") as file:
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
                print(f"Parsing error encountered: {e}")
                break
        
        return result


def load_scene(scene_id):
    states = read_json_records(DATA_ROOT / "states" / f"{scene_id}.jsonl")
    captions = read_json_records(DATA_ROOT / "captions" / f"{scene_id}.jsonl")
    
    