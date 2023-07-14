import csv
import random
import sys

from torch.utils.data import Dataset


class Spam(Dataset):
    def __init__(self, filename, split="train"):
        self.filename: str = filename
        self.split: str = split
        self.samples: list[tuple] = []

        self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _get_samples(self):
        csv.field_size_limit(sys.maxsize)
        with open(self.filename, "r") as f:
            reader = csv.reader(f)
            dataset = []
            for row in reader:
                dataset.append((row[0], row[1]))

        random.shuffle(dataset)
        split_idx = int(len(dataset) * 0.85)
        split_idx_val = split_idx + int(len(dataset) * 0.10)
        if self.split == "train":
            self.samples = dataset[:split_idx]
        elif self.split == "test":
            self.samples = dataset[split_idx:split_idx_val]
        elif self.split == "val":
            self.samples = dataset[split_idx_val:]

        random.shuffle(self.samples)
