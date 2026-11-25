import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from experiment_config import camera_deltas, read_config


def image_transform(preprocessing):
    return transforms.Compose([
        transforms.Resize(tuple(preprocessing["size"])),
        transforms.ToTensor(),
        transforms.Normalize(preprocessing["mean"], preprocessing["std"]),
    ])


class SteeringDataset(Dataset):
    def __init__(self, csv_path, image_root, preprocessing, steering_delta=None, *, deltas=None):
        self.image_root = Path(image_root)
        if steering_delta is not None:
            deltas = camera_deltas({"steering_delta": steering_delta})
        offsets = None if deltas is None else {"center": 0.0, "left": deltas[0], "right": deltas[1]}
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if offsets is None:
                    target = float(row["steering_target"])
                else:
                    target = float(row["steering_recorded"]) + offsets[row["camera"]]
                self.samples.append((row["image_path"], target))
        if not self.samples:
            raise ValueError(f"Empty split: {csv_path}")
        self.transform = image_transform(preprocessing)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, target = self.samples[index]
        with Image.open(self.image_root / relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(target, dtype=torch.float32)


def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def make_model(config, weights):
    model = models.get_model(config["architecture"], weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def make_loader(config, base, split, shuffle, steering_delta=None, *, deltas=None):
    dataset = SteeringDataset(base / split, base / config["data"]["image_root"],
                              config["preprocessing"], steering_delta, deltas=deltas)
    training = config["training"]
    return DataLoader(dataset, batch_size=training["batch_size"],
                      num_workers=training["num_workers"], shuffle=shuffle)


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            if training:
                optimizer.zero_grad()
            predictions = model(images.to(device)).squeeze(1)
            loss = criterion(predictions, targets.to(device))
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * targets.numel()
    return total / len(loader.dataset)


def save_learning_curve(directory):
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    with (directory / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in rows]
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    for metric, marker in (("train_loss", "o"), ("val_mse", "s")):
        ax.plot(epochs, [float(row[metric]) for row in rows], marker=marker, label=metric)
    ax.set(title=directory.name, xlabel="Epoch", ylabel="Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(directory / "learning_curve.png", dpi=150)


def train(config, base, name, run_dir):
    settings = config["training"]
    if settings["epochs"] < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(settings["seed"])
    device = select_device(settings["device"])
    deltas = camera_deltas(settings) if name == "model_b" else None
    train_loader = make_loader(config, base, config["experiments"][name], settings["shuffle_train"], deltas=deltas)
    val_loader = make_loader(config, base, config["data"]["validation"], False)
    model = make_model(config["model"], config["model"]["weights"]).to(device)
    criterion = getattr(nn, settings["loss"])()
    optimizer = getattr(torch.optim, settings["optimizer"])(model.parameters(), **settings["optimizer_args"])
    directory = run_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    best = math.inf
    with (directory / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "optimizer_steps", "train_loss", "val_mse"])
        for epoch in range(1, settings["epochs"] + 1):
            loss = run_epoch(model, train_loader, device, criterion, optimizer)
            mse = run_epoch(model, val_loader, device, nn.MSELoss())
            if not math.isfinite(loss) or not math.isfinite(mse):
                raise ValueError(f"Non-finite loss in {name}, epoch {epoch}")
            writer.writerow([epoch, epoch * len(train_loader), loss, mse])
            stream.flush()
            print(f"{name} epoch {epoch}: train_loss={loss:.6f} val_mse={mse:.6f}", flush=True)
            if mse < best:
                best = mse
                temporary = directory / "best.pt.tmp"
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_mse": mse,
                            "config": config, "experiment": name}, temporary)
                temporary.replace(directory / "best.pt")
    save_learning_curve(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the configured behavioral cloning models.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--experiment", help="Run one experiment, overriding the YAML model switches.")
    args = parser.parse_args()
    config = read_config(args.config)
    switches = config.get("train_models", {})
    for name, enabled in switches.items():
        if name not in config["experiments"] or not isinstance(enabled, bool):
            parser.error(f"Invalid train_models entry: {name}={enabled!r}")
    if args.experiment and args.experiment not in config["experiments"]:
        parser.error(f"Unknown experiment: {args.experiment}")
    if args.experiment:
        config["experiments"] = {args.experiment: config["experiments"][args.experiment]}
    else:
        config["experiments"] = {name: path for name, path in config["experiments"].items()
                                 if switches.get(name, True)}
    config.pop("train_models", None)  # The run config records only the selected experiments.
    if not config["experiments"]:
        parser.error("Enable at least one experiment in the YAML file.")
    base = args.config.resolve().parent
    timestamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base / config["output_dir"] / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    # Keep the saved configuration usable relative to its new location.
    snapshot = dict(config, output_dir="..", evaluation=dict(config["evaluation"], run_dir="."))
    if config["evaluation"].get("after_training", False):
        snapshot["evaluation"]["checkpoints"] = {name: f"{name}/best.pt" for name in config["experiments"]}
    for group in ("data", "experiments"):
        snapshot[group] = {key: os.path.relpath(base / value, run_dir)
                           for key, value in config[group].items()}
    plot_settings = config.get("plot", {})
    if "input_csvs" in plot_settings:
        snapshot["plot"] = dict(plot_settings,
                                input_csvs={label: os.path.relpath(base / path, run_dir)
                                            for label, path in plot_settings["input_csvs"].items()},
                                output_png=os.path.relpath(base / plot_settings["output_png"], run_dir))
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    print(f"Run directory: {run_dir.resolve()}", flush=True)
    for name in config["experiments"]:
        train(config, base, name, run_dir)
    if config["evaluation"].get("after_training", False):
        subprocess.run([sys.executable, Path(__file__).with_name("evaluate.py"),
                        "--config", run_dir / "config.yaml"], check=True)
