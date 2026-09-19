"""Build a compact CoVLA training set from the full release's MP4 files."""

import json
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageOps


REPO_ID = "turing-motors/CoVLA-Dataset"
MINI_REPO_ID = "turing-motors/CoVLA-Dataset-Mini"
STATE_SAMPLE_KEYS = {
    "extrinsic_matrix",
    "intrinsic_matrix",
    "trajectory",
    "trajectory_count",
}


def _read_records(source) -> list[dict]:
    text = source.read().decode("utf-8")
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        record, position = decoder.raw_decode(text, position)
        # Full CoVLA archives wrap frames as {"0": {...}}, while Mini
        # stores the inner dictionaries directly.
        if (
            len(record) == 1
            and (frame_id := next(iter(record))).isdigit()
            and isinstance(record[frame_id], dict)
        ):
            record = {**record[frame_id], "frame_id": int(frame_id)}
        records.append(record)
    return records


def _read_record_file(path: Path) -> list[dict]:
    with path.open("rb") as source:
        return _read_records(source)


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")))
            output.write(chr(10))
    _remove_appledouble(path)


def _remove_appledouble(path: Path) -> None:
    """Remove the FAT32 sidecar macOS creates for a generated file."""
    path.with_name(f"._{path.name}").unlink(missing_ok=True)


def _scene_signature(records: list[dict]):
    if not records:
        return None
    return records[0].get("ego_state", {}).get(
        "timestamp",
        records[0].get("timestamp"),
    )


def _normalize_state(record: dict, scene_id: str) -> dict:
    frame_id = record["frame_id"]
    return {
        "frame_id": frame_id,
        "image_path": f"images/{scene_id}/{frame_id:04d}.jpg",
        "ego_state": {
            key: value
            for key, value in record.items()
            if key not in STATE_SAMPLE_KEYS | {"frame_id", "image_path"}
        },
        **{
            key: record[key]
            for key in STATE_SAMPLE_KEYS
        },
    }


def _download(
    filename: str,
    token: str,
    cache_dir: Path,
    repo_id: str = REPO_ID,
) -> Path:
    last_error = None
    for attempt in range(3):
        try:
            path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    token=token,
                    cache_dir=cache_dir,
                )
            )
            if path.stat().st_size:
                return path
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as error:
            last_error = error
        if attempt < 2:
            time.sleep(5 * 2**attempt)
    if last_error is not None:
        raise RuntimeError(f"Download failed three times: {filename}") from last_error
    raise RuntimeError(f"Downloaded an empty file three times: {filename}")


def _copy_download(
    filename: str,
    token: str,
    output_root: Path,
    cache_dir: Path,
) -> None:
    destination = output_root / filename
    if destination.is_file():
        return
    source = _download(filename, token, cache_dir, MINI_REPO_ID)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # copy2 creates AppleDouble `._*` files on FAT32 volumes.
    shutil.copyfile(source, destination)
    _remove_appledouble(destination)


def _compact_states(
    archive_path: Path,
    output_root: Path,
    frame_interval: int,
    validation_signatures: set,
    scene_limit: int | None,
) -> int:
    written = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            records = _read_records(source)
            if _scene_signature(records) in validation_signatures:
                continue
            scene_id = Path(member.name).stem
            selected = [
                _normalize_state(record, scene_id)
                for record in records
                if record["frame_id"] % frame_interval == 0
                and record.get("trajectory_count") == 60
            ]
            if selected:
                _write_records(
                    output_root / "states" / Path(member.name).name,
                    selected,
                )
                written += 1
                if scene_limit is not None and written >= scene_limit:
                    break
    return written


def _compact_captions(archive_path: Path, output_root: Path) -> int:
    written = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            filename = Path(member.name).name
            state_path = output_root / "states" / filename
            if not state_path.is_file():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            captions = {
                caption.get("frame_id", index): caption
                for index, caption in enumerate(_read_records(source))
            }
            frame_ids = [
                record["frame_id"]
                for record in _read_record_file(state_path)
            ]
            if any(frame_id not in captions for frame_id in frame_ids):
                raise RuntimeError(f"Caption alignment failed for {filename}")
            _write_records(
                output_root / "captions" / filename,
                [
                    {
                        key: value
                        for key, value in captions[frame_id].items()
                        if key != "frame_id"
                    }
                    for frame_id in frame_ids
                ],
            )
            written += 1
    return written


def _validation_signatures(mini_root: Path | None) -> set:
    if mini_root is None:
        return set()
    state_dir = mini_root / "states"
    if not state_dir.is_dir():
        raise FileNotFoundError(state_dir)
    signatures = {
        _scene_signature(_read_record_file(path))
        for path in state_dir.glob("*.jsonl")
        if not path.name.startswith("._")
    }
    signatures.discard(None)
    return signatures


def _image_path(output_root: Path, state: dict) -> Path:
    return (output_root / state["image_path"]).with_suffix(".jpg")


def _extract_scene(video_path: Path, states: list[dict], output_root: Path) -> None:
    missing = {
        state["frame_id"]: state
        for state in states
        if not _image_path(output_root, state).is_file()
    }
    if not missing:
        return

    capture = cv2.VideoCapture(str(video_path))
    frame_id = 0
    try:
        while missing:
            success, frame = capture.read()
            if not success:
                break
            state = missing.pop(frame_id, None)
            if state is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = ImageOps.fit(
                    image,
                    (224, 224),
                    method=Image.Resampling.BICUBIC,
                )
                destination = _image_path(output_root, state)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                image.save(temporary, format="JPEG", quality=85)
                temporary.replace(destination)
                _remove_appledouble(temporary)
                _remove_appledouble(destination)
            frame_id += 1
    finally:
        capture.release()

    if missing:
        raise RuntimeError(
            f"{video_path.name} ended before frames {sorted(missing)[:5]}"
        )


def _prepare_scene(
    scene_id: str,
    video_file: str,
    token: str,
    output_root: Path,
    temp_root: Path,
    cleanup_downloads: bool,
) -> str:
    states = _read_record_file(output_root / "states" / f"{scene_id}.jsonl")
    if all(_image_path(output_root, state).is_file() for state in states):
        return scene_id

    if shutil.disk_usage(output_root).free / 1024**3 < 5:
        raise RuntimeError("Less than 5 GB remains; stopping safely.")

    cache_dir = temp_root / "videos" / scene_id
    try:
        video_path = _download(video_file, token, cache_dir)
        _extract_scene(video_path, states, output_root)
    finally:
        if cleanup_downloads:
            shutil.rmtree(cache_dir, ignore_errors=True)
    return scene_id


def prepare_covla_mini(
    token: str,
    output_root="/mnt/local-scratch/covla-mini",
    temp_root="/mnt/local-scratch/covla-mini-cache",
    frame_interval=10,
    cleanup_downloads=True,
) -> Path:
    """Keep sampled Mini frames as original-resolution validation PNGs."""
    output_root = Path(output_root)
    temp_root = Path(temp_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    files = HfApi().list_repo_files(
        repo_id=MINI_REPO_ID,
        repo_type="dataset",
        token=token,
    )
    metadata = [
        filename
        for filename in files
        if filename.startswith(("states/", "captions/"))
        and filename.endswith(".jsonl")
    ]
    for filename in metadata:
        _copy_download(filename, token, output_root, temp_root / "metadata")

    scene_ids = sorted(
        path.stem
        for path in (output_root / "states").glob("*.jsonl")
        if not path.name.startswith("._")
    )
    for index, scene_id in enumerate(scene_ids, start=1):
        marker = output_root / ".complete" / scene_id
        if marker.exists():
            continue
        states = _read_record_file(output_root / "states" / f"{scene_id}.jsonl")
        selected_ids = {
            state["frame_id"]
            for state in states
            if state["frame_id"] % frame_interval == 0
            and state.get("trajectory_count") == 60
        }
        print(f"Mini [{index}/{len(scene_ids)}] {scene_id}")
        cache_dir = temp_root / "images" / scene_id
        archive_path = _download(
            f"images/{scene_id}.tar.gz",
            token,
            cache_dir,
            MINI_REPO_ID,
        )
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if (
                    member.isfile()
                    and member.name.endswith(".png")
                    and int(Path(member.name).stem) in selected_ids
                ):
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    destination = output_root / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    _remove_appledouble(destination)
        image_dir = output_root / "images" / scene_id
        images = [
            path
            for path in image_dir.glob("*.png")
            if not path.name.startswith("._")
        ]
        if len(images) != len(selected_ids):
            raise RuntimeError(f"Incomplete Mini extraction for {scene_id}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        _remove_appledouble(marker)
        if cleanup_downloads:
            shutil.rmtree(cache_dir)

    if cleanup_downloads:
        shutil.rmtree(temp_root / "metadata", ignore_errors=True)
    print(f"Prepared {len(scene_ids)} Mini validation scenes in {output_root}")
    return output_root


def prepare_covla_full(
    token: str,
    output_root="/mnt/local-scratch/covla-full",
    mini_root="/content/data/covla-mini",
    temp_root="/mnt/local-scratch/covla-cache",
    frame_interval=10,
    num_scenes=None,
    cleanup_downloads=True,
    video_workers=4,
) -> Path:
    """Prepare compact training images while excluding Mini validation scenes."""
    if not token:
        raise ValueError("A Hugging Face token is required")
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    if video_workers <= 0:
        raise ValueError("video_workers must be positive")

    output_root = Path(output_root)
    mini_root = Path(mini_root) if mini_root else None
    temp_root = Path(temp_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    signatures = _validation_signatures(mini_root)
    artifacts = (
        ("states.tar.gz", _compact_states),
        ("captions.tar.gz", _compact_captions),
    )
    for filename, processor in artifacts:
        marker = output_root / f".{filename}.complete"
        if marker.exists():
            continue
        cache_dir = temp_root / filename
        archive_path = _download(filename, token, cache_dir)
        if filename == "states.tar.gz":
            written = processor(
                archive_path,
                output_root,
                frame_interval,
                signatures,
                num_scenes,
            )
        else:
            written = processor(archive_path, output_root)
        if written == 0:
            raise RuntimeError(f"No scenes were read from {filename}")
        marker.touch()
        _remove_appledouble(marker)
        if cleanup_downloads:
            shutil.rmtree(cache_dir)

    scene_ids = sorted(
        path.stem
        for path in (output_root / "states").glob("*.jsonl")
        if not path.name.startswith("._")
    )
    if num_scenes is not None:
        scene_ids = scene_ids[:num_scenes]

    print("Reading full-dataset video list...")
    video_files = {
        Path(filename).stem: filename
        for filename in HfApi().list_repo_files(
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token,
        )
        if filename.startswith("videos/") and filename.endswith(".mp4")
    }
    missing_videos = set(scene_ids) - video_files.keys()
    if missing_videos:
        raise RuntimeError(
            "State/video IDs do not match; first missing IDs: "
            f"{sorted(missing_videos)[:5]}"
        )

    incomplete = []
    for scene_id in scene_ids:
        states = _read_record_file(output_root / "states" / f"{scene_id}.jsonl")
        if not all(_image_path(output_root, state).is_file() for state in states):
            incomplete.append(scene_id)

    completed = len(scene_ids) - len(incomplete)
    executor = ThreadPoolExecutor(max_workers=video_workers)
    futures = {
        executor.submit(
            _prepare_scene,
            scene_id,
            video_files[scene_id],
            token,
            output_root,
            temp_root,
            cleanup_downloads,
        ): scene_id
        for scene_id in incomplete
    }
    try:
        for future in as_completed(futures):
            scene_id = future.result()
            completed += 1
            print(f"[{completed}/{len(scene_ids)}] {scene_id}")
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    print(f"Prepared {len(scene_ids)} training scenes in {output_root}")
    return output_root
