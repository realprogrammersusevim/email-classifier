import os
import random

from torch.utils.data import Dataset


class Spam(Dataset):
    """
    Spam dataset

    Attributes
    ----------
    root_dir : str
        Path to the root directory of the dataset
    split : str, default="train"
        Split of the dataset to use
    """

    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.samples = []
        self.index = len(self.samples)

        self._get_files_and_labels()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with open(sample["file_path"], "r") as f:
            text = f.read()

        sample = (sample["label"], text)
        return sample

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self[self.index]

    def _get_files_and_labels(self):
        classes = os.listdir(self.root_dir)
        for class_name in classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                files = os.listdir(class_dir)

                split_idx = int(len(files) * 0.85)  # 85% for training
                split_idx_val = split_idx + int(len(files) * 0.10)  # 10% for testing
                if self.split == "train":
                    selected_files = files[:split_idx]
                elif self.split == "test":
                    selected_files = files[split_idx:split_idx_val]
                elif self.split == "val":
                    selected_files = files[split_idx_val:]  # 5% for validation
                elif self.split == "all":
                    selected_files = files

                for file_name in selected_files:
                    current_sample = {}
                    file_path = os.path.join(class_dir, file_name)
                    current_sample["file_path"] = str(file_path)
                    current_sample["label"] = int(class_name)
                    self.samples.append(current_sample)

        random.shuffle(self.samples)
        self.index = len(self.samples)
