"""
dataset.py
----------
PyTorch Dataset for Fight Detection.
Loads 16 frames per video on-the-fly (memory safe).

Folder structure expected:
    processed/
    ├── fight/
    │   ├── video_1/
    │   │   ├── frame_0.jpg ... frame_15.jpg
    └── non_fight/
        ├── video_2/
            ├── frame_0.jpg ... frame_15.jpg
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path


# ──────────────────────────────────────────────
# ImageNet normalization (required for EfficientNet)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(augment: bool = False):
    """
    Returns torchvision transform pipeline.
    augment=True  → used for training split
    augment=False → used for val/test split
    """
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ──────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────
class FightDataset(Dataset):
    """
    Loads video frame sequences for fight detection.

    Returns per sample:
        frames → Tensor of shape (16, 3, 224, 224)   [T, C, H, W]
        label  → Tensor scalar: 1 (fight) or 0 (non_fight)
    """

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 16,
        augment: bool = False,
        frame_size: tuple = (224, 224),
    ):
        """
        Args:
            root_dir   : Path to processed/ folder containing fight/ and non_fight/
            num_frames : Number of frames per video (must match extraction step)
            augment    : Apply training augmentations
            frame_size : (H, W) - must match what was extracted
        """
        self.root_dir   = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform  = get_transforms(augment)

        self.samples = []   # list of (video_folder_path, label)
        self._build_index()

    def _build_index(self):
        """Scans root_dir and builds list of (path, label) pairs."""
        class_map = {"fight": 1, "non_fight": 0}

        for class_name, label in class_map.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}\n"
                    f"Make sure root_dir points to the 'processed/' folder."
                )

            video_folders = sorted([
                d for d in class_dir.iterdir()
                if d.is_dir()
            ])

            for video_folder in video_folders:
                self.samples.append((video_folder, label))

        print(f"[Dataset] Total videos indexed : {len(self.samples)}")
        fight_count    = sum(1 for _, l in self.samples if l == 1)
        nonfight_count = sum(1 for _, l in self.samples if l == 0)
        print(f"[Dataset]   fight     : {fight_count}")
        print(f"[Dataset]   non_fight : {nonfight_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_folder, label = self.samples[idx]

        frames = self._load_frames(video_folder)  # (16, 3, 224, 224)
        label  = torch.tensor(label, dtype=torch.long)

        return frames, label

    def _load_frames(self, video_folder: Path) -> torch.Tensor:
        """
        Loads exactly self.num_frames frames from a video folder.
        Returns Tensor: (num_frames, C, H, W)
        """
        frame_tensors = []

        for i in range(self.num_frames):
            frame_path = video_folder / f"frame_{i}.jpg"

            if not frame_path.exists():
                # Fallback: duplicate last valid frame if missing
                if frame_tensors:
                    frame_tensors.append(frame_tensors[-1])
                else:
                    # Edge case: first frame missing → black frame
                    dummy = np.zeros((*self.frame_size, 3), dtype=np.uint8)
                    frame_tensors.append(self.transform(dummy))
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                # Corrupt image fallback
                dummy = np.zeros((*self.frame_size, 3), dtype=np.uint8)
                frame_tensors.append(self.transform(dummy))
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # BGR → RGB
            img = cv2.resize(img, self.frame_size)           # ensure 224×224
            frame_tensors.append(self.transform(img))        # (3, 224, 224)

        return torch.stack(frame_tensors)   # (16, 3, 224, 224)


# ──────────────────────────────────────────────
# Train / Val / Test Split Utility
# ──────────────────────────────────────────────
def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float   = 0.15,
    # test is remainder: 0.15
    seed: int = 42,
):
    """
    Creates train / val / test DataLoaders from the processed folder.

    Args:
        root_dir    : Path to processed/ folder
        batch_size  : Videos per batch (keep low: 8–16 for laptop GPU)
        num_workers : Parallel workers for loading (4 is safe for Windows)
        train_ratio : Fraction for training
        val_ratio   : Fraction for validation

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split

    # Full dataset (no augmentation yet — we'll re-init train with augment)
    full_dataset = FightDataset(root_dir=root_dir, augment=False)
    total = len(full_dataset)

    n_train = int(total * train_ratio)
    n_val   = int(total * val_ratio)
    n_test  = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Re-build train dataset WITH augmentation
    train_dataset = FightDataset(root_dir=root_dir, augment=True)

    # Use same indices as train_subset
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_subset.indices)

    print(f"\n[Split] Train : {len(train_dataset)}")
    print(f"[Split] Val   : {len(val_subset)}")
    print(f"[Split] Test  : {len(test_subset)}")

    # Windows: num_workers > 0 requires if __name__ == '__main__' guard
    # Set num_workers=0 if you get multiprocessing errors on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    # faster GPU transfer
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Quick Smoke Test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    PROCESSED_DIR = r"D:\projects__\hackstreet_26\datasets\processed"

    print("=" * 50)
    print("SMOKE TEST: FightDataset")
    print("=" * 50)

    # Test single dataset load
    dataset = FightDataset(root_dir=PROCESSED_DIR, augment=False)

    sample_frames, sample_label = dataset[0]
    print(f"\nSample frames shape : {sample_frames.shape}")   # (16, 3, 224, 224)
    print(f"Sample label        : {sample_label.item()} ({'fight' if sample_label == 1 else 'non_fight'})")
    print(f"Frames dtype        : {sample_frames.dtype}")
    print(f"Frames min/max      : {sample_frames.min():.3f} / {sample_frames.max():.3f}")

    # Test DataLoaders
    print("\n" + "=" * 50)
    print("SMOKE TEST: DataLoaders")
    print("=" * 50)

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=PROCESSED_DIR,
        batch_size=4,
        num_workers=0,   # set 0 for Windows safety during testing
    )

    # One batch test
    for batch_frames, batch_labels in train_loader:
        print(f"\nBatch frames shape : {batch_frames.shape}")  # (4, 16, 3, 224, 224)
        print(f"Batch labels       : {batch_labels}")
        break

    print("\n✅ Dataset and DataLoader working correctly!")
