

import shutil
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ══════════════════════════════════════════════
#  CONFIG — matches your exact folder structure
# ══════════════════════════════════════════════
UCF_TRAIN_DIR  = r"D:\projects__\hackstreet_26\datasets\ucf_crime\Train"
BINARY_DIR     = r"D:\projects__\hackstreet_26\datasets\ucf_binary"
MODEL_SAVE     = r"D:\projects__\hackstreet_26\models\cnn_image_model.pth"

BATCH_SIZE     = 64
EPOCHS         = 15
LR             = 1e-4
IMG_SIZE       = 224
NUM_WORKERS    = 0        # Windows: keep at 0

# Exact folder names inside ucf_crime/Train/ — confirmed from your screenshot
FIGHT_CLASSES     = ["Fighting", "Assault", "Abuse"]
NON_FIGHT_CLASSES = ["NormalVideos"]

# Max images per class — keeps dataset balanced
MAX_PER_CLASS  = 5000

# Train / val split ratio
TRAIN_RATIO    = 0.85


# ══════════════════════════════════════════════
#  STEP 1: BUILD BINARY DATASET
# ══════════════════════════════════════════════
def build_binary_dataset():
    """
    Scans ucf_crime/Train/ and builds a clean binary dataset at ucf_binary/.
    Copies images — does not modify the original dataset.
    """
    train_path = Path(UCF_TRAIN_DIR)
    out_path   = Path(BINARY_DIR)

    print("[Build] Scanning UCF Crime Train folder...")
    print(f"[Build] Path: {train_path.resolve()}\n")

    # Verify path exists
    if not train_path.exists():
        raise FileNotFoundError(
            f"UCF Crime Train folder not found:\n  {train_path}\n"
            f"Check that UCF_TRAIN_DIR is correct."
        )

    # Show what folders are available
    available = [d.name for d in train_path.iterdir() if d.is_dir()]
    print(f"[Build] Available folders: {available}\n")

    # ── Collect fight images ──
    fight_imgs = []
    for cls in FIGHT_CLASSES:
        cls_dir = train_path / cls
        if cls_dir.exists():
            imgs = list(cls_dir.rglob("*.jpg")) + \
                   list(cls_dir.rglob("*.jpeg")) + \
                   list(cls_dir.rglob("*.png"))
            fight_imgs.extend(imgs)
            print(f"[Build]   {cls:<20} → {len(imgs):>6} images  [fight]")
        else:
            print(f"[Build]   {cls:<20} → NOT FOUND (skipped)")

    # ── Collect non-fight images ──
    nonfight_imgs = []
    for cls in NON_FIGHT_CLASSES:
        cls_dir = train_path / cls
        if cls_dir.exists():
            imgs = list(cls_dir.rglob("*.jpg")) + \
                   list(cls_dir.rglob("*.jpeg")) + \
                   list(cls_dir.rglob("*.png"))
            nonfight_imgs.extend(imgs)
            print(f"[Build]   {cls:<20} → {len(imgs):>6} images  [non_fight]")
        else:
            print(f"[Build]   {cls:<20} → NOT FOUND (skipped)")

    # ── Validate ──
    if not fight_imgs:
        raise ValueError(
            f"No fight images found!\n"
            f"Looking for folders: {FIGHT_CLASSES}\n"
            f"Available: {available}"
        )
    if not nonfight_imgs:
        raise ValueError(
            f"No non-fight images found!\n"
            f"Looking for folders: {NON_FIGHT_CLASSES}\n"
            f"Available: {available}"
        )

    # ── Balance + cap ──
    random.seed(42)
    random.shuffle(fight_imgs)
    random.shuffle(nonfight_imgs)

    # Balance to the smaller class, then cap at MAX_PER_CLASS
    min_count     = min(len(fight_imgs), len(nonfight_imgs), MAX_PER_CLASS)
    fight_imgs    = fight_imgs[:min_count]
    nonfight_imgs = nonfight_imgs[:min_count]

    print(f"\n[Build] Balanced to: {min_count} fight | {min_count} non_fight")

    # ── Split and copy ──
    def split_and_copy(imgs, label):
        n_train    = int(len(imgs) * TRAIN_RATIO)
        train_imgs = imgs[:n_train]
        val_imgs   = imgs[n_train:]

        for phase, subset in [("train", train_imgs), ("val", val_imgs)]:
            dest = out_path / phase / label
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(subset):
                ext      = src.suffix.lower()
                dst_name = f"{label}_{i:06d}{ext}"   # avoid filename collisions
                shutil.copy2(src, dest / dst_name)

        print(f"[Build]   {label:<12} → {n_train} train | {len(val_imgs)} val  ✓")

    print("\n[Build] Copying images (may take a minute)...")
    split_and_copy(fight_imgs,    "fight")
    split_and_copy(nonfight_imgs, "non_fight")

    print(f"\n[Build] ✅ Binary dataset ready: {out_path.resolve()}")


# ══════════════════════════════════════════════
#  STEP 2: PYTORCH DATASET
# ══════════════════════════════════════════════
def get_transforms(phase: str):
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


class ImageFightDataset(Dataset):
    def __init__(self, root_dir: str, phase: str):
        self.transform = get_transforms(phase)
        self.samples   = []

        root = Path(root_dir) / phase
        for label_name, label_idx in [("fight", 1), ("non_fight", 0)]:
            cls_dir = root / label_name
            if not cls_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {cls_dir}\n"
                    f"Run build_binary_dataset() first."
                )
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, label_idx))

        random.shuffle(self.samples)

        fight_count    = sum(1 for _, l in self.samples if l == 1)
        nonfight_count = sum(1 for _, l in self.samples if l == 0)
        print(f"[Dataset/{phase:<5}] {len(self.samples)} total  "
              f"(fight: {fight_count} | non_fight: {nonfight_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            return self.transform(img), torch.tensor(label, dtype=torch.long)
        except Exception:
            # Corrupt image fallback — return blank image
            dummy = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            return self.transform(dummy), torch.tensor(label, dtype=torch.long)


# ══════════════════════════════════════════════
#  STEP 3: MODEL
# ══════════════════════════════════════════════
def build_model(device: torch.device) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    Last 3 blocks unfrozen for fine-tuning.
    Classifier head replaced with binary output (fight / non_fight).
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks (fine-tune only these)
    for block in list(model.features.children())[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")

    return model


# ══════════════════════════════════════════════
#  STEP 4: TRAINING LOOP
# ══════════════════════════════════════════════
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Datasets + loaders
    print()
    train_ds = ImageFightDataset(BINARY_DIR, "train")
    val_ds   = ImageFightDataset(BINARY_DIR, "val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model, loss, optimizer, scheduler
    print("\n[Model] Building fine-tuned EfficientNet-B0...")
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_acc   = 0.0
    patience       = 5
    patience_count = 0

    print(f"\n[Train] {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
    print(f"        Early stop after {patience} epochs of no improvement\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>8}")
    print("-" * 58)

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * len(y)
            t_correct += (out.argmax(1) == y).sum().item()
            t_total   += len(y)

        # ── Val ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out  = model(X)
                loss = criterion(out, y)
                v_loss    += loss.item() * len(y)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total   += len(y)

        scheduler.step()

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total

        print(f"{epoch:>6} | {t_loss/t_total:>10.4f} | {t_acc*100:>8.2f}% | "
              f"{v_loss/v_total:>8.4f} | {v_acc*100:>8.2f}%")

        # ── Save best model ──
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            Path(MODEL_SAVE).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          v_acc,
            }, MODEL_SAVE)
            print(f"         ↑ Best model saved (val_acc={v_acc*100:.2f}%)")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[Early Stop] No improvement for {patience} epochs.")
                break

    print(f"\n{'='*58}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*58}")
    print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"  Model saved to    : {MODEL_SAVE}")
    print(f"\n  ✅ Ready to use with predict_dual.py")
    print(f"     Run: python work/predict_dual.py --source webcam")


# ══════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 58)
    print("  STEP 1 — Building binary dataset from UCF Crime")
    print("=" * 58)
    build_binary_dataset()

    print("\n" + "=" * 58)
    print("  STEP 2 — Training CNN image classifier")
    print("=" * 58)
    train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
cnn_image_classifier.py
-----------------------
Model 2: EfficientNet-B0 fine-tuned as a binary image classifier.
Trained on UCF Crime dataset images.

Your folder structure:
    ucf_crime/
    ├── Train/
    │   ├── Fighting/
    │   ├── Assault/
    │   ├── Abuse/
    │   ├── NormalVideos/
    │   └── ... (other classes, ignored)
    └── Test/   ← ignored entirely

Script automatically creates:
    ucf_binary/
    ├── train/
    │   ├── fight/
    │   └── non_fight/
    └── val/
        ├── fight/
        └── non_fight/

Usage:
    python work/cnn_image_classifier.py
"""