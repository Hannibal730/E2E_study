import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from run_paths import evaluation_run, output_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_panels(panels, destination):
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, sharex=True, sharey=True, squeeze=False,
                             figsize=(4.0 * columns, 3.25 * rows), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.26, wspace=0.15, hspace=0.18)
    for ax, (title, edges, recorded, prediction) in zip(axes.flat, panels):
        ax.stairs(recorded, edges, fill=True, color="#aeb1b8", alpha=0.65, label="Recorded")
        ax.stairs(prediction, edges, color="#147477", label="Prediction")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.supxlabel("Steering")
    fig.supylabel("Probability density")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def save_metrics(sources, destination):
    measurements = []
    for label, path in sources.items():
        snapshot = path.parent / "config.yaml"
        settings = yaml.safe_load(snapshot.read_text()) if snapshot.is_file() else {}
        metrics = path.parent / settings.get("evaluation", {}).get("metrics_csv", "metrics.csv")
        if not metrics.is_file():
            print(f"Skipping metrics for {label}: missing {metrics}")
            continue
        delta = settings.get("training", {}).get("steering_delta")
        with metrics.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["experiment"] == "model_b" and delta is None:
                    try:
                        delta = float(label.split("=")[-1])
                    except ValueError:
                        print(f"Skipping model_b metrics for {label}: unknown steering_delta")
                        continue
                measurements.append((label, delta, row))
    if not measurements:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, key, title in zip(axes, ("mse", "mae", "kl_steering_recorded_to_prediction"),
                             ("MSE", "MAE", "KL divergence")):
        points = []
        for label, delta, row in measurements:
            value = float(row[key] if key in row else row["kl_expert_to_prediction"])
            if row["experiment"] == "model_a":
                ax.axhline(value, ls="--", color="#58566b", label="model_a")
            elif row["experiment"] == "model_b":
                points.append((delta, value))
        if points:
            x, y = zip(*sorted(points))
            ax.plot(x, y, "-o", ms=4, color="#147477", label="model_b")
        ax.set(xlabel="model_b steering correction δ", title=title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Saved {destination.resolve()}")


def plot(sources, destination, alpha):
    if not sources:
        raise ValueError("Specify at least one histogram in plot.input_csvs")
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
    colors = ("#0066ad", "#c34400", "#007a55", "#943998", "#ad8000", "#00859b",
              "#bc315e", "#524ab3", "#64852b", "#805430", "#df654a")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")
    prediction_index = 0
    panels = []
    recorded_series = set()
    for label, path in sources.items():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Empty histogram: {path}")
        left = np.array([float(row["left"]) for row in rows])
        right = np.array([float(row["right"]) for row in rows])
        widths = right - left
        if np.any(widths <= 0) or not np.array_equal(right[:-1], left[1:]):
            raise ValueError(f"Invalid histogram edges: {path}")
        edges = np.append(left, right[-1])
        densities = {}
        for column in reader.fieldnames:
            if column in ("left", "right"):
                continue
            counts = np.array([int(row[column]) for row in rows])
            if np.any(counts < 0) or counts.sum() == 0:
                raise ValueError(f"Invalid counts in {path}: {column}")
            recorded = column in ("steering_recorded", "expert")
            name = "steering_recorded" if recorded else column
            density = counts / (counts.sum() * widths)
            densities[name] = density
            if recorded:
                signature = (tuple(edges), tuple(counts))
                if signature in recorded_series:
                    continue
                recorded_series.add(signature)
                ax.stairs(density, edges, fill=True, color="#777777",
                          alpha=0.22, label=f"{label} / {name}", zorder=1)
            else:
                if name == "model_a":
                    color, marker, legend_label = "#181818", "p", name
                else:
                    i = prediction_index
                    prediction_index += 1
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    legend_label = f"{name} · {label}"
                # Place the marker at the center of the highest-density bin.
                midpoints = (left + right) / 2
                x = np.column_stack((left, midpoints, right)).ravel()
                ax.plot(x, np.repeat(density, 3), color=color, linestyle="-",
                        linewidth=1.7, alpha=alpha, marker=marker,
                        markevery=[3 * int(density.argmax()) + 1], markersize=6,
                        markerfacecolor="white", markeredgewidth=1.4,
                        label=legend_label, zorder=4)
        recorded_density = densities.pop("steering_recorded")
        for name, density in densities.items():
            title = name if name == "model_a" else f"{name} ({label})"
            panels.append((title, edges, recorded_density, density))
    ax.set(xlabel="Steering", ylabel="Probability density")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=3.5)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(f"Saved {destination.resolve()}")
    panel_path = destination.with_name("distributions_by_delta.png")
    save_panels(panels, panel_path)
    print(f"Saved {panel_path.resolve()}")
    save_metrics(sources, destination.with_name("metrics_by_delta.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay steering histograms from the configured CSV files.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base = args.config.resolve().parent
    options = config["plot"]
    if "input_csvs" in options:
        sources = {label: base / path for label, path in options["input_csvs"].items()}
    else:
        # Saved configs from earlier runs resolve a single input relative to that run.
        base = evaluation_run(config, base)
        sources = {base.name: base / (options["input_csv"] or config["evaluation"]["histogram_csv"])}
    destination = output_path(base, options["output_png"])
    plot(sources, destination, options["alpha"])
