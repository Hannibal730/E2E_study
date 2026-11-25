import math
from pathlib import Path

import yaml


def read_config(path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def camera_deltas(settings):
    """Return signed (left, right) offsets, including for historical run configs."""
    if "left_delta" in settings or "right_delta" in settings:
        if not {"left_delta", "right_delta"} <= settings.keys() or "steering_delta" in settings:
            raise ValueError("Specify both left_delta and right_delta without steering_delta")
        left, right = settings["left_delta"], settings["right_delta"]
    elif "steering_delta" in settings:
        left = settings["steering_delta"]
        right = -left
    else:
        raise ValueError("Missing model_b steering settings: specify left_delta and right_delta")
    if not (math.isfinite(left) and math.isfinite(right) and left >= 0 and right <= 0):
        raise ValueError("Expected finite left_delta >= 0 and right_delta <= 0")
    return left, right


def evaluation_run(config, base):
    selection = config["evaluation"]["run_dir"]
    root = base / config["output_dir"]
    if selection != "latest":
        path = Path(selection)
        if path.name == selection:
            return (root / path).resolve()
        # Earlier configurations used paths relative to the YAML, including ".".
        return (base / path).resolve()
    runs = sorted(path for path in root.glob("[0-9]*_[0-9]*_[0-9]*")
                  if (path / "config.yaml").is_file() or (path / "train.yaml").is_file())
    if not runs:
        raise FileNotFoundError(f"No timestamped training runs under {root.resolve()}")
    return runs[-1]


def output_path(run, filename):
    path = run / Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
