"""
dataset_prep.py — CEG4195 Lab 2
--------------------------------
Downloads the Massachusetts Buildings Dataset via the Kaggle API,
generates binary pixel masks for each aerial image, tiles them into
256x256 patches, and splits into train/val/test sets.

Pixel mask generation concept adapted from A7_CEG4195/main.py
get_pseudo_labels(): instead of applying a confidence threshold to
classifier probabilities, we apply an intensity threshold to raw
mask pixel values to produce a binary {0,1} segmentation mask per pixel.

Run:
    python dataset_prep.py

Requires in .env:
    KAGGLE_USERNAME, KAGGLE_KEY
"""

import os
import json
import zipfile
import shutil
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
PROCESSED_DIR  = Path("data/processed")
PATCH_SIZE     = 256
STRIDE         = 256          # non-overlapping patches
MASK_THRESHOLD = 0.5          # pixel intensity threshold for building vs background
                              # mirrors A7's confidence threshold in get_pseudo_labels()
SEED           = 42
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
# TEST_FRAC implicitly = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

KAGGLE_DATASET = "balraj98/massachusetts-buildings-dataset"


# ── Step 1: Download ───────────────────────────────────────────────────────────

def download_dataset():
    """
    Downloads the Massachusetts Buildings Dataset from Kaggle.
    Credentials are read from KAGGLE_USERNAME and KAGGLE_KEY in .env,
    which the kaggle CLI picks up via ~/.kaggle/kaggle.json or environment vars.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Write kaggle.json so the kaggle CLI can authenticate
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        creds = {
            "username": os.environ["KAGGLE_USERNAME"],
            "key":      os.environ["KAGGLE_KEY"]
        }
        kaggle_json.write_text(json.dumps(creds))
        kaggle_json.chmod(0o600)

    zip_dest = RAW_DIR / "massachusetts-buildings-dataset.zip"
    if not zip_dest.exists():
        print(f"Downloading {KAGGLE_DATASET} ...")
        os.system(
            f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --quiet"
        )
    else:
        print("Zip already downloaded, skipping.")

    # Unzip if not already done
    png_dir = RAW_DIR / "png"
    if not png_dir.exists():
        print("Extracting zip ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(RAW_DIR)
    else:
        print("Already extracted, skipping.")

    return RAW_DIR


# ── Step 2: Image + mask loading ───────────────────────────────────────────────

def load_image_mask_pair(img_path: Path, mask_path: Path):
    """
    Opens an aerial image and its corresponding mask .tiff file.

    Returns:
        image : np.ndarray float32, shape (H, W, 3), values in [0, 1]
        mask  : np.ndarray uint8,   shape (H, W),    values in {0, 1}
    """
    image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

    mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = generate_pixel_mask(mask_raw)

    return image, mask


# ── Step 3: Pixel mask generation ─────────────────────────────────────────────

def generate_pixel_mask(mask_array: np.ndarray, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    """
    Converts a grayscale mask array into a binary pixel mask.

    Concept adapted from A7_CEG4195/main.py get_pseudo_labels():
        In A7, a confidence threshold is applied to classifier probabilities:
            selected_mask = confidences >= threshold
        Here we apply the same threshold logic at the pixel level:
            pixel = 1 (building)  if intensity >= threshold
            pixel = 0 (background) otherwise

    Args:
        mask_array : float32 array with values in [0, 1]
        threshold  : intensity cutoff (default 0.5)

    Returns:
        binary uint8 array {0, 1}, same spatial dimensions as input
    """
    return (mask_array >= threshold).astype(np.uint8)


# ── Step 4: Patch extraction ───────────────────────────────────────────────────

def extract_patches(image: np.ndarray, mask: np.ndarray):
    """
    Tiles a large aerial image and its mask into PATCH_SIZE x PATCH_SIZE crops.
    Patches where the mask is entirely zero (no buildings) are discarded to
    avoid flooding the dataset with background-only samples.

    Returns:
        list of (img_patch, mask_patch) tuples, each (PATCH_SIZE, PATCH_SIZE, 3)
        and (PATCH_SIZE, PATCH_SIZE) respectively.
    """
    H, W = image.shape[:2]
    patches = []

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            img_patch  = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Keep patch only if it contains at least some building pixels
            if mask_patch.sum() > 0:
                patches.append((img_patch, mask_patch))

    return patches


# ── Step 5: Split and save ─────────────────────────────────────────────────────

def split_and_save(all_patches: list):
    """
    Shuffles all (image, mask) patches, then splits into train/val/test
    at 70/15/15 and saves as stacked .npy files for fast loading during training.

    Output structure:
        data/processed/train/images.npy   (N_train, 256, 256, 3)  float32
        data/processed/train/masks.npy    (N_train, 256, 256)     uint8
        data/processed/val/...
        data/processed/test/...
        data/processed/metadata.json
    """
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(all_patches))

    n_total = len(indices)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)

    splits = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train + n_val],
        "test":  indices[n_train + n_val:],
    }

    images_all = np.stack([p[0] for p in all_patches])
    masks_all  = np.stack([p[1] for p in all_patches])

    counts = {}
    for split_name, idx in splits.items():
        split_dir = PROCESSED_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "images.npy", images_all[idx])
        np.save(split_dir / "masks.npy",  masks_all[idx])
        counts[split_name] = int(len(idx))
        print(f"  {split_name:>5}: {len(idx):>5} patches saved → {split_dir}")

    metadata = {
        "patch_size":      PATCH_SIZE,
        "mask_threshold":  MASK_THRESHOLD,
        "seed":            SEED,
        "total_patches":   n_total,
        "split_counts":    counts,
    }
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata written to {PROCESSED_DIR / 'metadata.json'}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_raw_pairs(raw_dir: Path):
    """
    Locates matching (image, mask) .png pairs from the Kaggle mirror layout:
        data/raw/png/train/          <- aerial images
        data/raw/png/train_labels/   <- binary masks (same filenames)
        data/raw/png/val/
        data/raw/png/val_labels/
        data/raw/png/test/
        data/raw/png/test_labels/
    All splits are pooled together; dataset_prep.py re-splits at 70/15/15.
    """
    png_root = raw_dir / "png"
    splits   = ["train", "val", "test"]

    pairs = []
    for split in splits:
        img_dir  = png_root / split
        mask_dir = png_root / f"{split}_labels"

        if not img_dir.exists() or not mask_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask .png pairs found under {png_root}.\n"
            "Check the extracted directory structure."
        )

    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    print("=" * 60)
    print("CEG4195 Lab 2 — Dataset Preparation")
    print("=" * 60)

    extract_dir = download_dataset()

    print("\nLocating image/mask pairs ...")
    pairs = get_raw_pairs(extract_dir)
    print(f"Found {len(pairs)} image/mask pairs.")

    print("\nGenerating pixel masks and extracting patches ...")
    all_patches = []
    for i, (img_path, mask_path) in enumerate(pairs):
        image, mask = load_image_mask_pair(img_path, mask_path)
        patches = extract_patches(image, mask)
        all_patches.extend(patches)
        print(f"  [{i+1:>3}/{len(pairs)}] {img_path.name} → {len(patches)} patches")

    print(f"\nTotal patches (buildings only): {len(all_patches)}")

    print("\nSplitting and saving ...")
    split_and_save(all_patches)

    print("\nDone. Run train.py next.")


if __name__ == "__main__":
    main()
