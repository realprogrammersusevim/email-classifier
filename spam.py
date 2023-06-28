import os

import chardet
from torch.utils.data import Dataset


class Spam(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []
        self.split = split

        self._get_file_paths_and_labels()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        with open(file_path, "rb") as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)["encoding"]
            if encoding is None:
                encoding = "utf-8"
            text = raw_data.decode(encoding, errors="replace")

        sample = {"text": text, "label": label}
        return sample

    def _get_file_paths_and_labels(self):
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

                for file_name in selected_files:
                    file_path = os.path.join(class_dir, file_name)
                    self.file_paths.append(file_path)
                    self.labels.append(int(class_name))
