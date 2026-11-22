import csv
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SteeringDataset(Dataset):
    def __init__(self, csv_path, image_root, preprocessing, steering_delta=None):
        self.image_root = Path(image_root)
        if steering_delta is not None and (not math.isfinite(steering_delta) or steering_delta < 0):
            raise ValueError("steering_delta must be finite and nonnegative")
        camera_offset = {"center": 0, "left": 1, "right": -1}
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if steering_delta is None:
                    target = float(row["steering_target"])
                else:
                    target = float(row["steering_recorded"]) + camera_offset[row["camera"]] * steering_delta
                self.samples.append((row["image_path"], target))
        if not self.samples:
            raise ValueError(f"Empty split: {csv_path}")
        self.transform = transforms.Compose([
            transforms.Resize(tuple(preprocessing["size"])),
            transforms.ToTensor(),
            transforms.Normalize(preprocessing["mean"], preprocessing["std"]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        relative_path, target = self.samples[index]
        with Image.open(self.image_root / relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(target, dtype=torch.float32)
