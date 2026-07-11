import random
import numpy as np
import matplotlib.pyplot as plt

def visualize_random_samples(dataset, num_samples=3):
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        image, mask = dataset[idx]

        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        # Create blank overlay (black background)
        overlay = np.zeros_like(image_np)

        # Define regions
        background = mask_np == 0
        foreground = mask_np == 1
        boundary = mask_np == 255

        # Assign clear distinct colors
        overlay[background] = [0, 0, 1]   # BLUE → 0 (background)
        overlay[foreground] = [1, 0, 0]   # RED → 1 (pet)
        overlay[boundary]   = [0, 1, 0]   # GREEN → 255 (boundary)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title(f"Original Image (idx={idx})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Raw Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay (Blue=0, Red=1, Green=255)")
        plt.axis("off")

        plt.show()
