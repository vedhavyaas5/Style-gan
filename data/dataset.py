import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_image_transforms(image_size: int = 64):
    """Returns standard preprocessing pipeline normalizing to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class CustomImageDataset(Dataset):
    """
    Loads images from a folder structure containing .jpg, .jpeg, or .png images.
    """
    def __init__(self, root_dir: str, image_size: int = 64):
        self.root_dir = root_dir
        self.image_paths = []
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No valid images found in directory: {root_dir}")

        self.transform = get_image_transforms(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image)


class SyntheticShapeDataset(Dataset):
    """
    Synthetic dataset generating procedural geometric shapes (circles, squares, gradients).
    Ideal for zero-dependency training validation, testing, and debugging.
    """
    def __init__(self, num_samples: int = 1000, image_size: int = 64):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Seed by index for reproducibility
        np.random.seed(idx)
        img = Image.new("RGB", (self.image_size, self.image_size), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Draw 2-4 random colored geometric primitives
        num_shapes = np.random.randint(2, 5)
        for _ in range(num_shapes):
            x1, y1 = np.random.randint(0, self.image_size // 2, size=2)
            x2, y2 = np.random.randint(self.image_size // 2, self.image_size, size=2)
            color = tuple(int(c) for c in np.random.randint(50, 255, size=3))

            shape_type = np.random.choice(["ellipse", "rectangle"])
            if shape_type == "ellipse":
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)

        return self.transform(img)
