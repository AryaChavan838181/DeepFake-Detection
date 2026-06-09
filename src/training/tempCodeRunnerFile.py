"""
NoiseNet Training Script - PyTorch + Intel XPU
===============================================
Script location : src/training/train_noise_net_xpu.py
Model location  : src/models/noise_net_xpu.py

Dataset priority (auto-detected from project root):
  1. --data flag                  explicit custom path
  2. data_prepared_noise/         our noise-specific cache (created on first run)
  3. dfdc_train_part_00/ + _01/   raw DFDC  (auto-preprocessed on first run)

NOTE: data_prepared/ and data_prepared_pt/ are intentionally skipped
      (face-crop data for a different project)

Usage:
  python src/training/train_noise_net_xpu.py
  python src/training/train_noise_net_xpu.py --epochs 30 --batch 16
  python src/training/train_noise_net_xpu.py --data ./my_images

Requirements:
  pip install torch torchvision tqdm scipy pillow
  pip install intel-extension-for-pytorch    # optional, XPU only
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Resolve import paths ──────────────────────────────────────────────────────
# Layout:
#   <root>/src/training/train_noise_net_xpu.py   <- this file
#   <root>/src/models/noise_net_xpu.py           <- model
_this_file    = os.path.abspath(__file__)
_training_dir = os.path.dirname(_this_file)          # src/training
_src_dir      = os.path.dirname(_training_dir)       # src
_models_dir   = os.path.join(_src_dir, "models")     # src/models
_project_root = os.path.dirname(_src_dir)            # sem4_edi  (project root)

for _p in (_models_dir, _src_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from noise_net_xpu import NoiseNet, combined_loss, get_device, ssim_loss  # noqa: E402

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
IMG_SIZE = 226


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────
class ImageFolderFlat(Dataset):
    """Recursively collects all images under root. No labels needed."""

    def __init__(self, root, transform=None):
        self.paths = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    self.paths.append(os.path.join(dirpath, f))
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {root}")
        self.transform = transform
        print(f"  Found {len(self.paths):,} images  ({root})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img) if self.transform else img
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


class CachedTensorDataset(Dataset):
    """Load pre-saved .pt tensors - much faster than live JPEG decode."""

    def __init__(self, cache_dir):
        self.files = sorted(
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files in: {cache_dir}")
        print(f"  Loaded {len(self.files):,} cached tensors  ({cache_dir})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            return torch.load(self.files[idx], weights_only=True)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


def preprocess_and_cache(raw_dirs, cache_dir):
    """
    One-time preprocessing: reads raw DFDC images, resizes to IMG_SIZE,
    saves as .pt tensors in cache_dir.  Skipped on subsequent runs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    existing = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    if existing:
        print(f"  Cache already contains {len(existing):,} tensors - skipping.")
        return

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("  One-time preprocessing (this will take a while)...")
    idx = 0
    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            continue
        all_files = [
            os.path.join(dp, f)
            for dp, _, fs in os.walk(raw_dir)
            for f in fs if os.path.splitext(f)[1].lower() in {'.mp4'}
        ]
        import cv2
        for fpath in tqdm(all_files, desc=f"  {os.path.basename(raw_dir)}"):
            try:
                cap = cv2.VideoCapture(fpath)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t = tf(Image.fromarray(frame))
                    torch.save(t, os.path.join(cache_dir, f"{idx:08d}.pt"))
                    idx += 1
            except Exception:
                pass

    print(f"  Cached {idx:,} images -> {cache_dir}")


# ─────────────────────────────────────────────
# Dataset resolver
# ─────────────────────────────────────────────
def resolve_dataset(args):
    tf_live = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    print("\n[Dataset] Resolving...")

    if args.data:
        print(f"  Using explicit path: {args.data}")
        return _loaders(ImageFolderFlat(args.data, tf_live), args)

    cache_dir = os.path.join(_project_root, "data_prepared_noise")

    # noise-specific cache already exists
    if os.path.isdir(cache_dir) and any(f.endswith(".pt") for f in os.listdir(cache_dir)):
        print(f"  Noise cache found: {cache_dir}")
        return _loaders(CachedTensorDataset(cache_dir), args)

    # raw DFDC -> preprocess -> load
    raw_dirs = [
        d for d in [
            os.path.join(_project_root, "dfdc_train_part_00"),
            os.path.join(_project_root, "dfdc_train_part_01"),
        ]
        if os.path.isdir(d)
    ]

    if raw_dirs:
        print(f"  Raw DFDC dirs found: {[os.path.basename(d) for d in raw_dirs]}")
        preprocess_and_cache(raw_dirs, cache_dir)
        return _loaders(CachedTensorDataset(cache_dir), args)

    raise FileNotFoundError(
        "No dataset found. Use --data <path>, or place data in:\n"
        "  dfdc_train_part_00/\n"
        "  dfdc_train_part_01/"
    )


def _loaders(dataset, args):
    val_n   = max(1, int(len(dataset) * 0.12))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    workers  = min(4, os.cpu_count() or 1)
    pin_mem  = getattr(args, '_pin_memory', False)   # True only for CUDA
    kw = dict(num_workers=workers, pin_memory=pin_mem)

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True, **kw)
    vl = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,                  **kw)

    print(f"  Train: {train_n:,}  |  Val: {val_n:,}  |  Batch: {args.batch}")
    return tr, vl


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    print("=" * 65)
    print("  NOISENET  |  PyTorch + Intel XPU")
    print("=" * 65)

    device, ipex = get_device()

    # pin_memory only valid for CUDA — silently wastes memory on XPU
    _is_xpu  = str(device) == "xpu"
    _is_cuda = str(device).startswith("cuda")
    args._pin_memory = _is_cuda   # passed into _loaders via args

    train_loader, val_loader = resolve_dataset(args)

    print(f"\n[Model]  epsilon={args.epsilon}  device={device}")
    model = NoiseNet(epsilon=args.epsilon).to(device)

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"  Resumed: {args.weights}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    if ipex and str(device) == "xpu":
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
        print("  Intel IPEX optimisation applied")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path  = os.path.join(args.checkpoint_dir, "noise_net_best.pt")
    final_path = os.path.join(args.checkpoint_dir, "noise_net_final.pt")

    best_val   = float("inf")
    no_improve = 0
    history    = {"train_loss": [], "val_loss": [], "val_ssim": []}

    print(f"\n[Train]  {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs}", ncols=95, unit="batch")

        for step, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            noise     = model(images, training=True)
            perturbed = (images + noise).clamp(0.0, 1.0)
            loss      = combined_loss(images, perturbed, noise,
                                      ssim_w=1.0,
                                      l2_w=args.l2_weight,
                                      disrupt_w=args.disrupt_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            run_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{run_loss / (step + 1):.4f}",
                "MAE" : f"{noise.abs().mean().item():.5f}",
                "lr"  : f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        pbar.close()

        # Validation
        model.eval()
        v_loss = 0.0
        v_ssim = 0.0
        with torch.no_grad():
            for images in tqdm(val_loader, desc="  Val", ncols=70, leave=False):
                images    = images.to(device, non_blocking=True)
                noise     = model(images, training=False)
                perturbed = (images + noise).clamp(0.0, 1.0)
                v_loss += combined_loss(images, perturbed, noise,
                                        ssim_w=1.0,
                                        l2_w=args.l2_weight,
                                        disrupt_w=args.disrupt_weight).item()
                v_ssim += (1.0 - ssim_loss(images, perturbed)).item()

        val_loss = v_loss / len(val_loader)
        val_ssim = v_ssim / len(val_loader)
        tr_loss  = run_loss / len(train_loader)
        flag     = "OK" if val_ssim >= 0.95 else "WARN <0.95"

        print(
            f"\n  Ep {epoch:02d}/{args.epochs} | "
            f"Train {tr_loss:.4f} | Val {val_loss:.4f} | "
            f"SSIM {val_ssim:.4f} [{flag}] | "
            f"{time.time() - t0:.1f}s\n"
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_ssim)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), best_path)
            print(f"  Best model saved -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop (patience={args.patience})")
                break

    torch.save(model.state_dict(), final_path)
    np.save(os.path.join(args.checkpoint_dir, "history.npy"), history)

    print(f"\n  Done.  best_val={best_val:.4f}  final_ssim={history['val_ssim'][-1]:.4f}")
    print(f"  Final model -> {final_path}")
    return model, history


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train NoiseNet - Full-Image Deepfake Prevention")
    p.add_argument("--data",           type=str,   default=None)
    p.add_argument("--epochs",         type=int,   default=25)
    p.add_argument("--batch",          type=int,   default=16,   help="Lower to 8 if OOM")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--epsilon",        type=float, default=0.012,
                   help="Max perturbation magnitude (0.012 ~ 3/255, imperceptible at 8K)")
    p.add_argument("--l2-weight",      type=float, default=0.01,  dest="l2_weight")
    p.add_argument("--disrupt-weight", type=float, default=0.1,   dest="disrupt_weight")
    p.add_argument("--patience",       type=int,   default=4)
    p.add_argument("--checkpoint-dir", type=str,   default="saved_models/noise_net", dest="checkpoint_dir")
    p.add_argument("--weights",        type=str,   default=None,  help="Resume from .pt file")
    train(p.parse_args())