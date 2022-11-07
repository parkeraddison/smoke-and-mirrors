from typing import Tuple

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


class Const:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Resize to shape of ImageNet
        transforms.Resize((224, 224)),
        # Normalize using the mean and std of ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dataset(
    path: str,
    transform: torch.transforms = Const.transform,
    train_prop: float = 0.7,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:

    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=transform,
    )

    print(f"Class balance: {np.mean(dataset.targets):.1%} smoke ({len(dataset)} total images)")

    train_set, valid_set = torch.utils.data.random_split(dataset, [
        (trsize := int(len(dataset)*train_prop)),
        len(dataset)-trsize
    ])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
