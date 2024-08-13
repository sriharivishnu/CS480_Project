import torch.utils
from torch.utils.data import Dataset
import torch.utils.data

import os
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from PIL import Image


class PlantDataset(Dataset):
    def __init__(self, csv_file, img_dir, num_labels=6, image_transform=None):
        self.csv_data = pd.read_csv(csv_file, sep=",", header=0)
        self.img_dir = img_dir

        self.ids = self.csv_data.iloc[:, 0]
        self.csv_data = self.csv_data.iloc[:, 1:]

        self.labels = None
        if num_labels is not None and num_labels != 0:
            self.labels = self.csv_data.iloc[:, -num_labels:]
            self.csv_data = self.csv_data.iloc[:, :-num_labels]

        self.image_transform = image_transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        img_id = str(int(self.ids[idx]))
        img_path = os.path.join(self.img_dir, img_id + ".jpeg")
        img = Image.open(img_path)
        if self.image_transform:
            img = self.image_transform(img)

        y = np.empty(1)
        if self.labels is not None:
            y = torch.tensor(np.array(self.labels.iloc[idx]), dtype=torch.float32)
        return torch.tensor(np.array(row), dtype=torch.float32), img, y


class AugmentedDataset(Dataset):
    def __init__(
        self,
        plant_dataset: PlantDataset,
        model,
        cached_file: str,
        device,
        batch_size=64,
    ):
        self.plant = plant_dataset
        self.batch_size = batch_size
        if os.path.exists(cached_file):
            self.embeddings = pd.read_parquet(cached_file)
        else:
            self.embeddings = self.get_embeddings_(plant_dataset, model, device)
            self.embeddings.to_parquet(cached_file)

        self.csv_aug = pd.concat((self.plant.csv_data, self.embeddings), axis=1)
        self.labels = plant_dataset.labels

    def __len__(self):
        return len(self.plant)

    def __getitem__(self, idx):
        return (
            torch.tensor(np.array(self.csv_aug.iloc[idx]), dtype=torch.float32),
            (
                torch.tensor(np.array(self.labels.iloc[idx]), dtype=torch.float32)
                if self.labels is not None
                else None
            ),
        )

    def get_embeddings_(self, dataset: PlantDataset, model, device):
        # for batched reads
        print(f"Using batch size: {self.batch_size}")
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        results = []
        with torch.no_grad():
            for idx, (_, imgs, _) in enumerate(loader):
                results.append(model(imgs.to(device)))

                if idx % 10 == 0:
                    print(
                        f"Done augmenting {round((idx + 1) * len(imgs) / len(self.plant) * 100, 2)}%"
                    )
            return pd.DataFrame(torch.cat(results).cpu().numpy())


class PandasDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X.iloc[idx].values, dtype=torch.float32), torch.tensor(
            self.y.iloc[idx].values, dtype=torch.float32
        )
