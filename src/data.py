from typing import Iterable

from pathlib import Path
import shutil

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm


# class Const:
IMG_SIZE = 224
TOP_CROP_PROP = 0.25
_amount_to_crop = int((IMG_SIZE // (1-TOP_CROP_PROP)) - IMG_SIZE)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # Resize to shape of ImageNet (height will be further cropped in next step)
    transforms.Resize((IMG_SIZE+_amount_to_crop, IMG_SIZE)),
    # Crop out the top portion of pixels (sky/clouds) in accordance with SmokeyNet preprocessing
    transforms.Lambda(lambda img: F.crop(img, _amount_to_crop, 0, IMG_SIZE, IMG_SIZE)),
    # Normalize using the mean and std of ImageNet
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

###
# Files
###

def move_fires(source, dest, firefile):
    """Move fire names listed in a newline-separate file from source to dest."""
    with open(firefile, 'r') as infile:
        firenames = infile.read().split()

    Path(dest).mkdir(exist_ok=True, parents=True)

    for firename in firenames:
        try:
            shutil.move(Path(source, firename), Path(dest))
        except:
            print(f"Couldn't move {firename}")


def copy_raw_to_dest(source, dest, threshold=0):

    Path(dest).mkdir(exist_ok=True, parents=True)

    # Use a folder structure that can have labels parsed by PyTorch
    smokedir = Path(dest, 'smoke')
    nosmokedir = Path(dest, 'nosmoke')
    smokedir.mkdir(exist_ok=True)
    nosmokedir.mkdir(exist_ok=True)

    all_smoke_files = Path(source).rglob('*+*.jpg')
    all_nosmoke_files = Path(source).rglob('*-*.jpg')

    # Only copy files at least threshold minutes from/till ignition
    seconds_per_minute = 60
    thr_filter = lambda p: abs(int(p.stem.split('_')[-1])) / seconds_per_minute >= threshold
    smoke_files_to_copy = filter(thr_filter, all_smoke_files)
    nosmoke_files_to_copy = filter(thr_filter, all_nosmoke_files)

    # Copy into dest folders
    for smokefile in tqdm(list(smoke_files_to_copy)):
        shutil.copy(smokefile, smokedir)
    for nosmokefile in tqdm(list(nosmoke_files_to_copy)):
        shutil.copy(nosmokefile, nosmokedir)


def copy_virtual_to_dest(source, dest, threshold=0):

    Path(dest).mkdir(exist_ok=True, parents=True)

    smokedir = Path(dest, 'smoke')
    nosmokedir = Path(dest, 'nosmoke')
    smokedir.mkdir(exist_ok=True)
    nosmokedir.mkdir(exist_ok=True)

    all_files = Path(source).rglob('*.jpeg')

    # Our smoke data is captured from 0 to 1580 at 20-frame increments (where every 20 frames is one
    # minute), with smoke becoming visible at frame 800 (40 minutes)
    frames_per_minute = 20

    for file in tqdm(list(all_files)):
        frame = int(file.stem)
        if abs(frame - 800) / frames_per_minute >= threshold:
            if frame < 800:
                shutil.copy(file, Path(nosmokedir, f"{file.parent.name}_{file.name}"))
            else:
                shutil.copy(file, Path(smokedir, f"{file.parent.name}_{file.name}"))


###
# Data Loaders
###

def load_images_from_folder(
    source: str,
    transform: torchvision.transforms = TRANSFORM,
    batch_size: int = 1,
    shuffle: bool = True,
    **loader_args,
) -> DataLoader:

    dataset = torchvision.datasets.ImageFolder(root=source, transform=transform)
    print(f"Loaded from: {source}")
    print(f"\tClass balance: {np.mean(dataset.targets):.1%} smoke ({len(dataset)} total images)")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_args)
    return data_loader


def sample(dataset: Dataset, n: int, **loader_args) -> DataLoader:

    data_loader = DataLoader(dataset, **loader_args, sampler=RandomSampler(dataset, num_samples=n))
    return data_loader


def combine(datasets: Iterable[Dataset], **loader_args) -> DataLoader:

    combined_dataset = ConcatDataset(datasets)
    combined_loader = DataLoader(combined_dataset, **loader_args)
    return combined_loader


def save_tensors_to_folder(data_loader: DataLoader, dest: str, labels=(0,1)):
    for l in labels:
        try:
            Path(dest, str(l)).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise FileExistsError("Don't overwrite existing transformed data! Delete to proceed.")
    # Loop through loader and save transformed images to a destination folder
    for i, (ims, labs) in enumerate(tqdm(data_loader)):
        for j, label in enumerate(labs):
            torch.save(ims[j], Path(dest, str(int(label)), f"{i+j}.pt"))


def load_tensors_from_folder(source: str, transform=None, batch_size=1, shuffle=True) -> DataLoader:
    dataset = torchvision.datasets.DatasetFolder(
        root=source, transform=transform, extensions=['pt'],
        loader=lambda path: torch.load(path)
    )
    print(f"Loaded from: {source}")
    print(f"\tClass balance: {np.mean(dataset.targets):.1%} smoke ({len(dataset)} total images)")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
