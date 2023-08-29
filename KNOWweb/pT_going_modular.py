# create DataSet and DataLoaders for the GOING MODULAR experiment

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from torch import nn




NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    # shuffle: bool = True,
    # pin_memory: bool = True,
    # drop_last: bool = False,
    ):

    train_dataset = Dataset.ImageFolder(train_dir, transform)
    test_dataset = Dataset.ImageFolder(test_dir, transform)

    # get class names
    class_names = train_dataset.classes

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, test_loader, class_names


## making a model TinyVGG with in model_builder.py   


