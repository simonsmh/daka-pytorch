import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils.utils import StrtoLabel

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class selfdataset(Dataset):
    def __init__(self, folder):
        self.train_image_file_paths = [
            os.path.join(folder, image_file) for image_file in os.listdir(folder)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        label_text = image_name.split("_")[0]
        labelTensor = Tensor(StrtoLabel(label_text))
        image = Image.open(image_root)
        image = self.transform(image)
        return image, labelTensor


def get_train_data_loader():
    dataset = selfdataset("data/train")
    return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)


def get_test_data_loader():
    dataset = selfdataset("data/test")
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)


def get_predict_data_loader():
    dataset = selfdataset("data/valid")
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
