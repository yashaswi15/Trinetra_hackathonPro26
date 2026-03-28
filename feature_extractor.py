
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from pathlib import Path

# Import our dataset
from dataset import create_dataloaders


# ──────────────────────────────────────────────
# EfficientNet-B0 Feature Extractor
# ──────────────────────────────────────────────
class EfficientNetExtractor(nn.Module):
    """
    Wraps EfficientNet-B0 pretrained on ImageNet.
    Removes the final classification head → outputs 1280-dim feature vectors.

    Input  : (B, 3, 224, 224)   — single frame
    Output : (B, 1280)          — feature vector
    """

    def __init__(self, device: torch.device):
        super().__init__()

        print("[CNN] Loading EfficientNet-B0 pretrained weights...")
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Remove classifier head: keep features only
        # efficientnet.features   → conv layers (outputs 1280 channels)
        # efficientnet.avgpool    → adaptive avg pool → (B, 1280, 1, 1)
        # We skip efficientnet.classifier
        self.features = efficientnet.features
        self.pool     = efficientnet.avgpool   # → (B, 1280, 1, 1)

        self.to(device)
        self.eval()  # always in eval mode (no dropout/batchnorm updates)

        # Freeze all weights — we're using it as a fixed feature extractor
        for param in self.parameters():
            param.requires_grad = False

        print(f"[CNN] EfficientNet-B0 loaded on {device}")
        print(f"[CNN] Output feature dim: 1280")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            features: (B, 1280)
        """
        x = self.features(x)    # (B, 1280, 7, 7)
        x = self.pool(x)        # (B, 1280, 1, 1)
        x = x.flatten(1)        # (B, 1280)
        return x


# ──────────────────────────────────────────────
# Frame-by-Frame Extraction (Memory Efficient)
# ──────────────────────────────────────────────
def extract_features_from_loader(
    loader,
    extractor: EfficientNetExtractor,
    device: torch.device,
    split_name: str = "split",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates through a DataLoader, extracts CNN features per frame.

    Args:
        loader      : DataLoader yielding (frames, labels)
                      frames shape: (B, T, 3, 224, 224)
        extractor   : EfficientNetExtractor (frozen)
        device      : cuda or cpu
        split_name  : For tqdm display

    Returns:
        all_features : np.ndarray of shape (N, T, 1280)
        all_labels   : np.ndarray of shape (N,)
    """
    all_features = []
    all_labels   = []

    extractor.eval()

    for batch_frames, batch_labels in tqdm(loader, desc=f"Extracting [{split_name}]"):
        # batch_frames: (B, T, 3, 224, 224)
        B, T, C, H, W = batch_frames.shape

        # Reshape: merge batch and time dims → process all frames at once
        frames_flat = batch_frames.view(B * T, C, H, W).to(device)   # (B*T, 3, 224, 224)

        # Extract features
        features_flat = extractor(frames_flat)                         # (B*T, 1280)

        # Reshape back to (B, T, 1280)
        features = features_flat.view(B, T, -1)                       # (B, 16, 1280)

        all_features.append(features.cpu().numpy())
        all_labels.append(batch_labels.numpy())

    all_features = np.concatenate(all_features, axis=0)   # (N, 16, 1280)
    all_labels   = np.concatenate(all_labels,   axis=0)   # (N,)

    return all_features, all_labels


#main_pipe
def run_feature_extraction(
    processed_dir: str,
    output_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
):
    """
    Full pipeline:
    1. Create DataLoaders
    2. Load EfficientNet-B0
    3. Extract features for train/val/test
    4. Save .npy files

    Args:
        processed_dir : Path to processed/ folder (frame folders)
        output_dir    : Where to save .npy feature files
        batch_size    : Batch size for extraction (can be higher than training)
        num_workers   : DataLoader workers
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # DataLoaders
    print("\n[DataLoader] Building splits...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=processed_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Load EfficientNet
    extractor = EfficientNetExtractor(device=device)

    # ── Extract Train ──
    print("\n[Extracting] Training set...")
    train_features, train_labels = extract_features_from_loader(
        train_loader, extractor, device, split_name="train"
    )
    np.save(output_path / "features_train.npy", train_features)
    np.save(output_path / "labels_train.npy",   train_labels)
    print(f"  ✅ Train features saved: {train_features.shape}")

    # ── Extract Val ──
    print("\n[Extracting] Validation set...")
    val_features, val_labels = extract_features_from_loader(
        val_loader, extractor, device, split_name="val"
    )
    np.save(output_path / "features_val.npy", val_features)
    np.save(output_path / "labels_val.npy",   val_labels)
    print(f"  ✅ Val features saved  : {val_features.shape}")

    # ── Extract Test ──
    print("\n[Extracting] Test set...")
    test_features, test_labels = extract_features_from_loader(
        test_loader, extractor, device, split_name="test"
    )
    np.save(output_path / "features_test.npy", test_features)
    np.save(output_path / "labels_test.npy",   test_labels)
    print(f"  ✅ Test features saved : {test_features.shape}")

    # ── Summary ──
    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Output directory : {output_path.resolve()}")
    print(f"Files saved:")
    for f in sorted(output_path.glob("*.npy")):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name:<30} {size_mb:.1f} MB")

    total_mb = sum(
        f.stat().st_size for f in output_path.glob("*.npy")
    ) / (1024 ** 2)
    print(f"\nTotal size: {total_mb:.1f} MB")
    print("\n✅ Ready for LSTM training!")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":

    PROCESSED_DIR = r"D:\projects__\hackstreet_26\datasets\processed"
    OUTPUT_DIR    = r"D:\projects__\hackstreet_26\datasets\features"

    run_feature_extraction(
        processed_dir=PROCESSED_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=16,      
        num_workers=0,      
                            
    )





























"""
feature_extractor.py
---------------------
EfficientNet-B0 CNN Feature Extractor.

Takes raw frame sequences from the Dataset and converts them to
compact feature vectors, which will be fed into the LSTM.

Input  : (batch, 16, 3, 224, 224)   — raw frames
Output : (batch, 16, 1280)          — per-frame feature vectors

Saves:
    features_train.npy  → shape (N_train, 16, 1280)
    features_val.npy    → shape (N_val,   16, 1280)
    features_test.npy   → shape (N_test,  16, 1280)
    labels_train.npy
    labels_val.npy
    labels_test.npy
"""