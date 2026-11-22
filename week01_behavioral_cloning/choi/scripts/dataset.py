import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from steering import camera_deltas


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
