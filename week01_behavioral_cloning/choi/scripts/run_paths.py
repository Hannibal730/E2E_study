from pathlib import Path


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
