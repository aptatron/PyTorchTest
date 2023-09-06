from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
#import transforms
from torchvision import transforms


from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "data/images"

train_path = image_path + "/train"
test_path = image_path + "/test"    

img_size = 224

manual_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor()
                                        ])

print(f"Manual Transforms: {manual_transforms}")

BATCH_SIZE = 32

train_loader, test_loader, class_names = data_setup.create_dataloaders(train_path, test_path, manual_transforms, BATCH_SIZE)

# get a single image from the train_loader
image_batch, label_batch = next(iter(train_loader))

# replicating ViT overview
# ViT-base, ViT-large, ViT-huge are different sizes of the model
# split data into patches

# ViT-base: 16x16 patches, 12 layers, 768 hidden size, 12 attention heads
# ViT-large: 16x16 patches, 24 layers, 1024 hidden size, 16 attention heads
# ViT-huge: 16x16 patches, 32 layers, 1280 hidden size, 16 attention heads

# Create example values
height = 224 # H ("The training resolution is 224.")
width = 224 # W
color_channels = 3 # C
patch_size = 16 # P

# Calculate N (number of patches)
number_of_patches = int((height * width) / patch_size**2)
print(f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}")

