import torch.utils
from torch.utils.data import Dataset
import torch.utils.data

import os
from torchvision.io import read_image
import pandas as pd
import numpy as np


class PlantDataset(Dataset):
    def __init__(
        self, csv_file, img_dir, num_labels=6, image_transform=None, csv_transform=None
    ):
        self.csv_data = pd.read_csv(csv_file, sep=",", header=0)
        self.img_dir = img_dir

        self.ids = self.csv_data.iloc[:, 0]
        self.csv_data = self.csv_data.iloc[:, 1:]

        self.labels = None
        if num_labels is not None and num_labels != 0:
            self.labels = self.csv_data.iloc[:, -num_labels:]
            self.csv_data = self.csv_data.iloc[:, :-num_labels]

        self.image_transform = image_transform

        if csv_transform:
            self.csv_data = csv_transform(self.csv_data)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        # if type(idx) != int:
        #     row = self.csv_data.iloc[idx]
        #     img_ids = [str(int(x)) for x in self.ids[idx]]
        #     img_paths = [
        #         os.path.join(self.img_dir, img_id + ".jpeg") for img_id in img_ids
        #     ]
        #     imgs = [
        #         read_image(img_path).to(dtype=torch.float32) / 255
        #         for img_path in img_paths
        #     ]

        #     if self.image_transform:
        #         imgs = [self.image_transform(img) for img in imgs]

        #     y = None
        #     if self.labels is not None:
        #         y = torch.tensor(np.array(self.labels.iloc[idx]), dtype=torch.float32)
        #     return (
        #         torch.tensor(np.array(row), dtype=torch.float32),
        #         torch.tensor(np.array(imgs)),
        #         y,
        #     )

        row = self.csv_data.iloc[idx]
        img_id = str(int(self.ids[idx]))
        img_path = os.path.join(self.img_dir, img_id + ".jpeg")
        img = read_image(img_path).to(dtype=torch.float32) / 255
        if self.image_transform:
            img = self.image_transform(img)

        y = np.empty(1)
        if self.labels is not None:
            y = torch.tensor(np.array(self.labels.iloc[idx]), dtype=torch.float32)
        return idx, torch.tensor(np.array(row), dtype=torch.float32), img, y
