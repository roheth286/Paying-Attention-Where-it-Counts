import os
import tarfile
import torch
from torch.utils.data import DataLoader

def extract_dataset(dataset_dir="Segmentattion_dataset"):
    """
    Extracts images.tar.gz and annotations.tar.gz into local workspace directories
    if they do not already exist.
    """
    images_tar = os.path.join(dataset_dir, "images.tar.gz")
    annotations_tar = os.path.join(dataset_dir, "annotations.tar.gz")

    if not os.path.exists("images"):
        print("Extracting images.tar.gz...")
        with tarfile.open(images_tar, "r:gz") as tar:
            tar.extractall()
        print("Images extracted successfully.")
    else:
        print("Images directory already exists.")

    if not os.path.exists("annotations"):
        print("Extracting annotations.tar.gz...")
        with tarfile.open(annotations_tar, "r:gz") as tar:
            tar.extractall()
        print("Annotations extracted successfully.")
    else:
        print("Annotations directory already exists.")


def compute_mean_std(dataset):
    """
    Computes the channel-wise mean and standard deviation of a dataset.
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in loader:
        # images: (B, C, H, W)
        b, c, h, w = images.shape
        num_pixels = b * h * w

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])

        total_pixels += num_pixels

    mean = channel_sum / total_pixels
    std = torch.sqrt(channel_sq_sum / total_pixels - mean ** 2)

    return mean, std
