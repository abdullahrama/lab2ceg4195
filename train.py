"""
train.py — CEG4195 Lab 2
-------------------------
Trains a UNet (ResNet34 encoder, pretrained on ImageNet) on the
Massachusetts Buildings Dataset prepared by dataset_prep.py.

Metrics tracked: IoU (Intersection over Union) and Dice score.
Best checkpoint is saved by highest validation IoU.

Run:
    python train.py

Requires:
    data/processed/  (created by dataset_prep.py)

Environment variables (via .env):
    MODEL_SAVE_PATH   path to save best checkpoint   (default: models/unet_resnet34.pth)
    BATCH_SIZE        mini-batch size                 (default: 16)
    EPOCHS            max training epochs             (default: 30)
    LEARNING_RATE     Adam learning rate              (default: 0.0001)
    SEG_THRESHOLD     pixel decision boundary         (default: 0.5)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dotenv import load_dotenv

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models/unet_resnet34.pth")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", 16))
EPOCHS          = int(os.getenv("EPOCHS", 30))
LR              = float(os.getenv("LEARNING_RATE", 1e-4))
THRESHOLD       = float(os.getenv("SEG_THRESHOLD", 0.5))
DATA_DIR        = Path("data/processed")
OUTPUT_DIR      = Path("outputs")
SEED            = 42

# ImageNet normalisation stats (must match what the ResNet34 encoder expects)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Dataset ────────────────────────────────────────────────────────────────────

class BuildingDataset(Dataset):
    """
    Loads pre-processed 256×256 patches from .npy files saved by dataset_prep.py.
    Applies albumentations augmentations for the training split only.
    """

    def __init__(self, split: str):
        split_dir = DATA_DIR / split
        self.images = np.load(split_dir / "images.npy", mmap_mode="r")  # (N,256,256,3) float32
        self.masks  = np.load(split_dir / "masks.npy",  mmap_mode="r")  # (N,256,256)   uint8

        if split == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # mmap arrays are read-only; copy to avoid albumentations write errors
        image = np.array(self.images[idx])   # (256,256,3) float32
        mask  = np.array(self.masks[idx])    # (256,256)   uint8

        augmented = self.transform(image=image, mask=mask)
        img_tensor  = augmented["image"].float()           # (3,256,256)
        mask_tensor = augmented["mask"].unsqueeze(0).float()  # (1,256,256)

        return img_tensor, mask_tensor


# ── Metrics ────────────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Dice = 2 * |pred ∩ target| / (|pred| + |target| + smooth)
    pred is passed as raw logits; sigmoid + threshold applied internally.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    return float((2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth))


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    IoU = |pred ∩ target| / (|pred ∪ target| + smooth)
    pred is passed as raw logits; sigmoid + threshold applied internally.
    """
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


# ── Training helpers ───────────────────────────────────────────────────────────

def build_criterion():
    """Combined loss: 0.5 * Dice + 0.5 * BCE (standard for binary segmentation)."""
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = nn.BCEWithLogitsLoss()

    def criterion(logits, targets):
        return 0.5 * dice_loss(logits, targets) + 0.5 * bce_loss(logits, targets)

    return criterion


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    criterion = build_criterion()

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_iou  += iou_score(logits, masks, THRESHOLD)
            total_dice += dice_score(logits, masks, THRESHOLD)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_training_curves(train_losses, val_losses, val_ious, val_dices):
    OUTPUT_DIR.mkdir(exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train loss")
    axes[0].plot(epochs, val_losses,   label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_ious,   label="Val IoU")
    axes[1].plot(epochs, val_dices,  label="Val Dice")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"Training curves saved to {OUTPUT_DIR / 'training_curves.png'}")


def save_sample_predictions(model, loader, device, n_samples=4):
    """Saves a grid of: aerial image | ground truth mask | predicted mask."""
    model.eval()
    OUTPUT_DIR.mkdir(exist_ok=True)

    images_batch, masks_batch = next(iter(loader))
    images_batch = images_batch[:n_samples].to(device)
    masks_batch  = masks_batch[:n_samples]

    with torch.no_grad():
        logits = model(images_batch)
        preds  = (torch.sigmoid(logits) > THRESHOLD).squeeze(1).cpu().numpy()

    # Un-normalise images for display
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD).reshape(3, 1, 1)
    imgs_display = (images_batch.cpu().numpy() * std + mean).clip(0, 1)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    for i in range(n_samples):
        axes[i, 0].imshow(imgs_display[i].transpose(1, 2, 0))
        axes[i, 0].set_title("Aerial image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(masks_batch[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i], cmap="gray")
        axes[i, 2].set_title("Predicted mask")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=150)
    plt.close()
    print(f"Sample predictions saved to {OUTPUT_DIR / 'sample_predictions.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model directory exists
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ── Data loaders
    train_ds = BuildingDataset("train")
    val_ds   = BuildingDataset("val")
    test_ds  = BuildingDataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # ── Model: UNet with pretrained ResNet34 encoder (transfer learning)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,   # raw logits; sigmoid applied inside loss/metrics
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = build_criterion()

    # ── Training loop
    best_iou = 0.0
    train_losses, val_losses, val_ious, val_dices = [], [], [], []

    print(f"\nTraining for up to {EPOCHS} epochs ...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss             = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, val_dice = validate(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_IoU={val_iou:.4f}  "
              f"val_Dice={val_dice:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved best model  (val IoU={best_iou:.4f})")

    # ── Test evaluation
    print("\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_iou, test_dice = validate(model, test_loader, device)

    print("\n" + "=" * 50)
    print(f"Test IoU  : {test_iou:.4f}")
    print(f"Test Dice : {test_dice:.4f}")
    print("=" * 50)

    # ── Save outputs
    save_training_curves(train_losses, val_losses, val_ious, val_dices)
    save_sample_predictions(model, test_loader, device)

    results = {
        "test_iou":   round(test_iou, 4),
        "test_dice":  round(test_dice, 4),
        "test_loss":  round(test_loss, 4),
        "best_val_iou": round(best_iou, 4),
        "epochs_trained": EPOCHS,
        "model_path": MODEL_SAVE_PATH,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {OUTPUT_DIR / 'results.json'}")
    print("Run python lab2.py to start the inference API.")


if __name__ == "__main__":
    main()
