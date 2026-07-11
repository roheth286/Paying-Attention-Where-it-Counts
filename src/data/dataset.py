import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.ignore_value = 255

        # Only allow valid image extensions
        valid_extensions = (".jpg", ".jpeg", ".png")

        all_files = os.listdir(image_dir)
        valid_images = []

        for fname in all_files:
            # Skip hidden files
            if fname.startswith("._"):
                continue

            # Skip non-image files
            if not fname.lower().endswith(valid_extensions):
                continue

            # Ensure matching mask exists
            mask_name = os.path.splitext(fname)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                valid_images.append(fname)

        self.images = sorted(valid_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]

        image = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask = Image.open(
            os.path.join(
                self.mask_dir,
                os.path.splitext(fname)[0] + ".png"
            )
        ).convert("L")

        # Resize
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # Remap mask
        mask_np = np.array(mask)
        remapped = np.full(mask_np.shape, self.ignore_value, dtype=np.uint8)

        remapped[mask_np == 1] = 1
        remapped[mask_np == 2] = 0

        # Convert to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(remapped).long()

        return image, mask


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # ----- Spatial Aug -----
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        angle = random.uniform(-10, 10)

        image = TF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.rotate(
            mask.unsqueeze(0),  # (1, H, W)
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=255
        ).squeeze(0)  # Back to (H, W)

        # ----- Color Aug (image only) -----
        image = TF.adjust_brightness(
            image,
            1 + random.uniform(-0.1, 0.1)
        )

        image = TF.adjust_contrast(
            image,
            1 + random.uniform(-0.1, 0.1)
        )

        # ----- Normalize -----
        image = TF.normalize(image, self.mean, self.std)

        return image, mask.long()


class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = TF.normalize(image, self.mean, self.std)
        return image, mask
