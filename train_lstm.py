

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from lstm_model import FightLSTM, get_model_summary


# ══════════════════════════════════════════════
#  CONFIG — edit these if needed
# ══════════════════════════════════════════════
FEATURES_DIR = r"D:\projects__\hackstreet_26\datasets\features"
MODELS_DIR   = r"D:\projects__\hackstreet_26\models"

BATCH_SIZE   = 32
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8          # early stopping: stop if val_loss doesn't improve


# ══════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════
def load_features(features_dir: str):
    """
    Loads pre-extracted .npy files and returns TensorDatasets.
    All data is in RAM — 234 MB total, safe for your system.
    """
    base = Path(features_dir)

    print("[Data] Loading .npy feature files...")

    X_train = np.load(base / "features_train.npy")   # (2100, 16, 1280)
    y_train = np.load(base / "labels_train.npy")     # (2100,)
    X_val   = np.load(base / "features_val.npy")     # (450,  16, 1280)
    y_val   = np.load(base / "labels_val.npy")       # (450,)
    X_test  = np.load(base / "features_test.npy")    # (450,  16, 1280)
    y_test  = np.load(base / "labels_test.npy")      # (450,)

    print(f"  Train : {X_train.shape}  labels: {y_train.shape}")
    print(f"  Val   : {X_val.shape}  labels: {y_val.shape}")
    print(f"  Test  : {X_test.shape}  labels: {y_test.shape}")

    # Class balance check
    fight_count    = int(y_train.sum())
    nonfight_count = len(y_train) - fight_count
    print(f"\n  Train class balance → fight: {fight_count} | non_fight: {nonfight_count}")

    # Convert to tensors
    def to_dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    train_ds = to_dataset(X_train, y_train)
    val_ds   = to_dataset(X_val,   y_val)
    test_ds  = to_dataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════
#  TRAIN / EVAL FUNCTIONS
# ══════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════
@torch.no_grad()
def evaluate_test(model, loader, device):
    """Detailed evaluation on test set with per-class accuracy."""
    model.eval()

    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits  = model(X_batch)
        preds   = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    overall_acc = (all_preds == all_labels).float().mean().item()

    # Per-class accuracy
    fight_mask    = all_labels == 1
    nonfight_mask = all_labels == 0

    fight_acc    = (all_preds[fight_mask]    == 1).float().mean().item()
    nonfight_acc = (all_preds[nonfight_mask] == 0).float().mean().item()

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Overall Accuracy  : {overall_acc * 100:.2f}%")
    print(f"  Fight Accuracy    : {fight_acc * 100:.2f}%")
    print(f"  Non-Fight Accuracy: {nonfight_acc * 100:.2f}%")
    print(f"{'='*50}")


# ══════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════
def train():
    # ── Setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    models_path = Path(MODELS_DIR)
    models_path.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──
    print()
    train_loader, val_loader, test_loader = load_features(FEATURES_DIR)

    # ── Model ──
    print("\n[Model] Building FightLSTM...")
    model = FightLSTM(
        input_dim=1280,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
    ).to(device)

    get_model_summary(model, device)

    # ── Loss, Optimizer, Scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # smoothing helps generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ── CSV Log Setup ──
    log_path = models_path / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    # ── Training Loop ──
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[Train] Starting training for {EPOCHS} epochs...")
    print(f"        Batch size : {BATCH_SIZE}")
    print(f"        LR         : {LR}")
    print(f"        Early stop : {PATIENCE} epochs\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Print Row ──
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
            f"{val_loss:>8.4f} | {val_acc*100:>8.2f}% | {current_lr:.2e}   [{elapsed:.1f}s]"
        )

        # ── Save CSV Log ──
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.2e}"])

        # ── Save Best Model ──
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":    val_acc,
                "val_loss":   val_loss,
            }, models_path / "best_model.pth")
            print(f"         ↑ New best model saved (val_acc={val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Early Stopping ──
        if patience_counter >= PATIENCE:
            print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Stopping.")
            break

    # ── Save Last Model ──
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
    }, models_path / "last_model.pth")

    print(f"\n[Done] Best Val Accuracy : {best_val_acc * 100:.2f}%")
    print(f"[Done] Models saved to   : {models_path.resolve()}")
    print(f"[Done] Training log      : {log_path.resolve()}")

    # ── Final Test Evaluation ──
    print("\n[Test] Loading best model for final evaluation...")
    checkpoint = torch.load(models_path / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_test(model, test_loader, device)


if __name__ == "__main__":
    train()
    
    
    
    



"""
train_lstm.py
-------------
Training script for FightLSTM.

Loads pre-extracted .npy features → trains LSTM → saves best model.

Usage:
    python work/train_lstm.py

Outputs:
    D:/projects__/hackstreet_26/models/best_model.pth   ← best val accuracy
    D:/projects__/hackstreet_26/models/last_model.pth   ← final epoch
    D:/projects__/hackstreet_26/models/training_log.csv ← epoch metrics
"""