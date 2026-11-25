import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from experiment_config import evaluation_run, output_path, read_config
from train import make_loader, make_model, select_device


@torch.inference_mode()
def predict(model, loader, device):
    predictions, targets = [], []
    model.eval()
    for images, batch_targets in loader:
        predictions.append(model(images.to(device)).squeeze(1).cpu())
        targets.append(batch_targets)
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def evaluate(config, base, output):
    device = select_device(config["training"]["device"])
    series = {}
    truth = None
    options = config["evaluation"]
    checkpoints = options["checkpoints"]
    if isinstance(checkpoints, str):
        experiments = config["experiments"]
        for filename in ("config.yaml", "train.yaml"):
            if (output / filename).is_file():
                experiments = read_config(output / filename)["experiments"]
                break
        checkpoints = {name: checkpoints.format(experiment=name) for name in experiments}
    if not checkpoints:
        raise ValueError("Specify at least one evaluation checkpoint")
    for name, filename in checkpoints.items():
        checkpoint = torch.load(output / filename, map_location="cpu", weights_only=True)
        if checkpoint["experiment"] != name:
            raise ValueError(f"Checkpoint experiment mismatch: {name}")
        saved = checkpoint["config"]
        model = make_model(saved["model"], weights=None)
        model.load_state_dict(checkpoint["state_dict"])
        loader = make_loader({**config, "preprocessing": saved["preprocessing"]},
                             base, config["data"]["test"], False)
        series[name], targets = predict(model.to(device), loader, device)
        if truth is not None and not np.array_equal(truth, targets):
            raise ValueError("Test targets differ between experiments")
        truth = targets

    edges = np.histogram_bin_edges(np.concatenate([truth, *series.values()]),
                                   bins=options["histogram_bins"])
    counts = {"steering_recorded": np.histogram(truth, bins=edges)[0]}
    smooth = options["kl_pseudocount"]
    if smooth <= 0:
        raise ValueError("kl_pseudocount must be positive")
    with output_path(output, options["metrics_csv"]).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "mse", "mae", "kl_steering_recorded_to_prediction"])
        for name, predictions in series.items():
            counts[name] = np.histogram(predictions, bins=edges)[0]
            p = counts["steering_recorded"] + smooth
            q = counts[name] + smooth
            p, q = p / p.sum(), q / q.sum()
            errors = predictions - truth
            writer.writerow([name, np.mean(errors ** 2), np.mean(np.abs(errors)),
                             np.sum(p * np.log(p / q))])
    with output_path(output, options["histogram_csv"]).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["left", "right", *counts])
        writer.writerows(zip(edges[:-1], edges[1:], *counts.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best checkpoints on the shared test split.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    base = args.config.resolve().parent
    config = read_config(args.config)
    run = evaluation_run(config, base)
    print(f"Evaluation directory: {run.resolve()}")
    evaluate(config, base, run)
