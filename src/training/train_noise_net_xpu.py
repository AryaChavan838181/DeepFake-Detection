"""
NoiseNet Training Script — PyTorch + Intel XPU  (OPTIMISED)
============================================================
Location : src/training/train_noise_net_xpu.py
Model    : src/models/noise_net_xpu.py

Key features:
  - Intel XPU / CUDA / CPU auto-detection + IPEX
  - Crash-safe checkpoint every N batches + auto-resume
  - Real-time tqdm: loss, SSIM, MAE, LR, ETA
  - DeepFace mid-epoch eval with ACTIVE feedback:
      if noise fails to disrupt -> disrupt_weight boosted for next 50 batches
  - LR warm-up (300 steps) + cosine annealing
  - Early stopping

Usage:
  python src/training/train_noise_net_xpu.py
  python src/training/train_noise_net_xpu.py --resume
  python src/training/train_noise_net_xpu.py --deepface-every 50
"""

import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")

# ── Import path resolution ────────────────────────────────────────────────────
_this_file    = os.path.abspath(__file__)
_training_dir = os.path.dirname(_this_file)
_src_dir      = os.path.dirname(_training_dir)
_models_dir   = os.path.join(_src_dir, "models")
_project_root = os.path.dirname(_src_dir)

for _p in (_models_dir, _src_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from noise_net_xpu import (
    NoiseNet, combined_loss, get_device, ssim_loss, perceptual_frequency_loss,
)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
IMG_SIZE = 226


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────
class ImageFolderFlat(Dataset):
    def __init__(self, root, transform=None):
        self.paths = []
        for dp, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    self.paths.append(os.path.join(dp, f))
        if not self.paths:
            raise FileNotFoundError(f"No images found in: {root}")
        self.transform = transform
        print(f"  Found {len(self.paths):,} images  ({root})")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img) if self.transform else img
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


class CachedTensorDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files in: {cache_dir}")
        print(f"  Loaded {len(self.files):,} cached tensors  ({cache_dir})")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            return torch.load(self.files[idx], weights_only=True)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


def preprocess_and_cache(raw_dirs, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    existing = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    if existing:
        print(f"  Cache already contains {len(existing):,} tensors - skipping.")
        return
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    print("  One-time preprocessing...")
    import cv2
    idx = 0
    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            continue
        all_files = [
            os.path.join(dp, f)
            for dp, _, fs in os.walk(raw_dir)
            for f in fs if os.path.splitext(f)[1].lower() == '.mp4'
        ]
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


def resolve_dataset(args):
    tf_live = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    print("\n[Dataset] Resolving...")

    if args.data:
        return _make_loaders(ImageFolderFlat(args.data, tf_live), args)

    cache_dir = os.path.join(_project_root, "data_prepared_noise")
    if os.path.isdir(cache_dir) and any(f.endswith(".pt") for f in os.listdir(cache_dir)):
        print(f"  Noise cache found: {cache_dir}")
        return _make_loaders(CachedTensorDataset(cache_dir), args)

    raw_dirs = [d for d in [
        os.path.join(_project_root, "dfdc_train_part_00"),
        os.path.join(_project_root, "dfdc_train_part_01"),
    ] if os.path.isdir(d)]

    if raw_dirs:
        preprocess_and_cache(raw_dirs, cache_dir)
        return _make_loaders(CachedTensorDataset(cache_dir), args)

    raise FileNotFoundError(
        "No dataset found. Use --data <path>, or place data in:\n"
        "  dfdc_train_part_00/\n  dfdc_train_part_01/"
    )


def _make_loaders(dataset, args):
    val_n   = max(1, int(len(dataset) * 0.12))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    is_cuda = str(args._device).startswith("cuda")
    workers = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=workers, pin_memory=is_cuda,
              persistent_workers=(workers > 0))

    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True,  **kw)
    vl = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False, **kw)
    print(f"  Train: {train_n:,}  |  Val: {val_n:,}  |  Batch: {args.batch}  |  Workers: {workers}")
    return tr, vl


# ─────────────────────────────────────────────
# DeepFace evaluator WITH active feedback loop
# ─────────────────────────────────────────────
class DeepFaceEvaluator:
    """
    Evaluates whether the protected image still fools DeepFace every N batches.

    ACTIVE FEEDBACK: if a sample is NOT disrupted (noise too weak),
    we signal the training loop to temporarily boost disrupt_weight
    for the next `feedback_batches` steps — forcing the model to
    learn stronger disruption on the spot.
    """

    def __init__(self, every: int = 50,
                 model_name: str = "Facenet",
                 feedback_batches: int = 50):
        self.every            = every
        self.model_name       = model_name
        self.feedback_batches = feedback_batches
        self._available       = False
        self._results         = []
        # Feedback state
        self.boost_remaining  = 0
        self.boost_factor     = 3.0   # multiply disrupt_weight by this when failing

        try:
            from deepface import DeepFace
            self._DeepFace = DeepFace
            self._available = True
            print(f"  [DeepFace] Available — eval every {every} batches ({model_name})")
            print(f"  [DeepFace] Active feedback: disrupt_weight x{self.boost_factor} for {feedback_batches} batches on failure")
        except ImportError:
            print("  [DeepFace] Not installed — skipping eval  (pip install deepface)")

    def get_disrupt_multiplier(self) -> float:
        """Returns current disrupt_weight multiplier. >1.0 means we're in feedback mode."""
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            return self.boost_factor
        return 1.0

    def maybe_evaluate(self, model, images, global_step, device) -> float | None:
        """
        Call once per batch.
        Returns disruption_rate (1.0=disrupted, 0.0=failed) or None if skipped.
        Side-effect: sets boost_remaining if noise failed.
        """
        if not self._available or self.every <= 0:
            return None
        if global_step % self.every != 0:
            return None
        if images.shape[0] == 0:
            return None

        try:
            import tempfile, cv2
            model.eval()
            with torch.no_grad():
                sample_orig = images[:1]
                sample_prot = model.protect(sample_orig)

            def to_np(t):
                return (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            orig_np = to_np(sample_orig)
            prot_np = to_np(sample_prot)

            disrupted = False
            with tempfile.TemporaryDirectory() as td:
                op = os.path.join(td, "orig.jpg")
                pp = os.path.join(td, "prot.jpg")
                cv2.imwrite(op, cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(pp, cv2.cvtColor(prot_np, cv2.COLOR_RGB2BGR))
                try:
                    res = self._DeepFace.verify(
                        img1_path=op, img2_path=pp,
                        model_name=self.model_name,
                        enforce_detection=True, silent=True,
                    )
                    disrupted = not res.get("verified", True)
                except Exception:
                    disrupted = True   # face detection failed = protection worked

            rate = 1.0 if disrupted else 0.0
            self._results.append((global_step, rate))
            status = "DISRUPTED ✓" if disrupted else "MATCHED ✗ — boosting disrupt_weight"

            if not disrupted:
                self.boost_remaining = self.feedback_batches

            print(f"\n  [DeepFace @step {global_step}] {status}")
            model.train()
            return rate

        except Exception:
            model.train()
            return None

    @property
    def disruption_history(self):
        return self._results


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_val, history, df_history):
    torch.save({
        "epoch": epoch, "global_step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "history": history, "df_history": df_history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return (ckpt.get("epoch", 0),
            ckpt.get("global_step", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("history", {"train_loss": [], "val_loss": [], "val_ssim": [], "deepface_disruption": []}),
            ckpt.get("df_history", []))


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train(args):
    print("=" * 70)
    print("  NOISENET  |  PyTorch + Intel XPU  |  OPTIMISED")
    print("=" * 70)

    device, ipex = get_device()
    args._device = device
    is_xpu  = str(device) == "xpu"
    is_cuda = str(device).startswith("cuda")

    train_loader, val_loader = resolve_dataset(args)

    print(f"\n[Model]  epsilon={args.epsilon}  device={device}")
    model = NoiseNet(epsilon=args.epsilon).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8
    )

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = min(300, total_steps // 10)

    warmup_sched  = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: float(s) / float(max(1, warmup_steps))
    )
    cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.05
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    try:
        import matplotlib.pyplot as plt
        CAN_PLOT = True
    except ImportError:
        CAN_PLOT = False

    if ipex and is_xpu:
        # Optimize with BFloat16 to massively speed up XPU training
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
        print("  Intel IPEX optimisation applied (BFloat16 Fast Path)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path  = os.path.join(args.checkpoint_dir, "noise_net_ckpt.pt")
    best_path  = os.path.join(args.checkpoint_dir, "noise_net_best.pt")
    final_path = os.path.join(args.checkpoint_dir, "noise_net_final.pt")

    start_epoch = 1
    global_step = 0
    best_val    = float("inf")
    history     = {"train_loss": [], "val_loss": [], "val_ssim": [], "deepface_disruption": []}
    df_history  = []
    no_improve  = 0

    resume_path = args.weights or (ckpt_path if args.resume and os.path.exists(ckpt_path) else None)
    if resume_path and os.path.exists(resume_path):
        start_epoch, global_step, best_val, history, df_history = \
            load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"  Resumed from epoch {start_epoch-1}  |  best_val={best_val:.4f}  |  step={global_step}")
    else:
        print("  Starting fresh training")

    df_eval = DeepFaceEvaluator(
        every=args.deepface_every,
        model_name=args.deepface_model,
        feedback_batches=50,
    )

    print(f"\n[Train]  {args.epochs} epochs  |  {len(train_loader)} batches/epoch")
    print(f"  Checkpoint every {args.save_every} batches")
    print(f"  DeepFace eval + feedback every {args.deepface_every} batches\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:02d}/{args.epochs}",
                    ncols=110, unit="batch")

        for step, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            global_step += 1

            # DeepFace feedback: check if we should boost disrupt_weight
            df_multiplier = df_eval.get_disrupt_multiplier()
            effective_disrupt_w = args.disrupt_weight * df_multiplier

            optimizer.zero_grad(set_to_none=True)
            
            # Using autocast for mixed precision
            if is_xpu and ipex:
                with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    noise     = model(images, training=True)
                    perturbed = (images + noise).clamp(0.0, 1.0)
                    loss      = combined_loss(
                        images, perturbed, noise,
                        ssim_w=1.0,
                        l2_w=args.l2_weight,
                        disrupt_w=effective_disrupt_w,
                        freq_w=args.freq_weight,
                    )
            else:
                noise     = model(images, training=True)
                perturbed = (images + noise).clamp(0.0, 1.0)
                loss      = combined_loss(
                    images, perturbed, noise,
                    ssim_w=1.0,
                    l2_w=args.l2_weight,
                    disrupt_w=effective_disrupt_w,
                    freq_w=args.freq_weight,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item()
            elapsed  = time.time() - t0
            eta      = (elapsed / (step + 1)) * (len(train_loader) - step - 1)

            pbar.set_postfix({
                "loss": f"{run_loss/(step+1):.4f}",
                "MAE" : f"{noise.abs().mean().item():.5f}",
                "lr"  : f"{optimizer.param_groups[0]['lr']:.2e}",
                "dw"  : f"{effective_disrupt_w:.3f}",
                "ETA" : f"{eta/60:.1f}m",
            })

            # Mid-batch DeepFace eval (active feedback applied next batch)
            df_rate = df_eval.maybe_evaluate(model, images, global_step, device)
            if df_rate is not None:
                df_history.append((global_step, df_rate))
                model.train()

            # Crash-safe checkpoint
            if global_step % args.save_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler,
                                epoch, global_step, best_val, history, df_history)

        pbar.close()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        v_loss = v_ssim = 0.0
        viz_orig = viz_noise = viz_prot = None
        with torch.no_grad():
            for images in tqdm(val_loader, desc="  Val", ncols=70, leave=False):
                images    = images.to(device, non_blocking=True)
                
                if is_xpu and ipex:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        noise     = model(images, training=False)
                        perturbed = (images + noise).clamp(0.0, 1.0)
                        batch_loss = combined_loss(images, perturbed, noise,
                                                   ssim_w=1.0, l2_w=args.l2_weight,
                                                   disrupt_w=args.disrupt_weight,
                                                   freq_w=args.freq_weight)
                else:
                    noise     = model(images, training=False)
                    perturbed = (images + noise).clamp(0.0, 1.0)
                    batch_loss = combined_loss(images, perturbed, noise,
                                               ssim_w=1.0, l2_w=args.l2_weight,
                                               disrupt_w=args.disrupt_weight,
                                               freq_w=args.freq_weight)
                
                v_loss   += batch_loss.item()
                v_ssim   += (1.0 - ssim_loss(images, perturbed)).item()
                viz_orig, viz_noise, viz_prot = images, noise, perturbed

        try:
            import torchvision.utils as vutils
            if viz_orig is not None:
                n_v = min(15, viz_orig.shape[0])
                n_vis = (viz_noise[:n_v] / args.epsilon + 1.0) / 2.0  # scale noise [-eps, eps] -> [0, 1]
                grid = torch.cat([viz_orig[:n_v], n_vis, viz_prot[:n_v]], dim=0)
                vutils.save_image(grid, os.path.join(args.checkpoint_dir, f"visuals_ep{epoch:02d}.jpg"), nrow=n_v)
        except Exception as e:
            print(f"  [Visual Warning] Failed to save visuals: {e}")

        if is_xpu:
            torch.xpu.synchronize()

        val_loss = v_loss / len(val_loader)
        val_ssim = v_ssim / len(val_loader)
        tr_loss  = run_loss / len(train_loader)
        epoch_t  = time.time() - t0

        # Epoch DeepFace summary
        epoch_df = [r for s, r in df_history
                    if s > global_step - len(train_loader) and s <= global_step]
        df_avg   = np.mean(epoch_df) if epoch_df else float("nan")
        ssim_flag = "OK" if val_ssim >= 0.95 else "WARN <0.95"
        df_flag   = f"{df_avg:.0%}" if not np.isnan(df_avg) else "n/a"

        print(
            f"\n  Ep {epoch:02d}/{args.epochs} | "
            f"Train {tr_loss:.4f} | Val {val_loss:.4f} | "
            f"SSIM {val_ssim:.4f} [{ssim_flag}] | "
            f"DeepFace disruption {df_flag} | "
            f"{epoch_t/60:.1f}min\n"
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_ssim)
        history["deepface_disruption"].append(float(df_avg) if not np.isnan(df_avg) else -1.0)

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(model.state_dict(), best_path)
            print(f"  Best model saved -> {best_path}  (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\n  Early stopping (patience={args.patience})")
                break

        save_checkpoint(ckpt_path, model, optimizer, scheduler,
                        epoch, global_step, best_val, history, df_history)
                        
        if CAN_PLOT and len(history["train_loss"]) > 0:
            try:
                plt.figure(figsize=(15, 4))
                
                plt.subplot(131)
                plt.plot(history["train_loss"], label="Train Loss")
                plt.plot(history["val_loss"], label="Val Loss")
                plt.title("Loss vs Epochs")
                plt.grid(True)
                plt.legend()
                
                plt.subplot(132)
                plt.plot(history["val_ssim"], label="Val SSIM", color="green")
                plt.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Target (0.95)")
                plt.title("SSIM vs Epochs")
                plt.grid(True)
                plt.legend()
                
                plt.subplot(133)
                df_epochs = [i for i, d in enumerate(history["deepface_disruption"], 1) if d >= 0]
                df_clean = [d for d in history["deepface_disruption"] if d >= 0]
                if df_clean:
                    plt.plot(df_epochs, df_clean, label="Disruption %", color="purple")
                    plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
                    plt.title("DeepFace Disruption Rate vs Epochs")
                    plt.grid(True)
                    plt.legend()
                    
                plt.tight_layout()
                plt.savefig(os.path.join(args.checkpoint_dir, "training_curves_epoch.png"))
                plt.close()
            except Exception as e:
                print(f"  [Plot Warning] Failed to generate graph: {e}")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), final_path)
    np.save(os.path.join(args.checkpoint_dir, "history.npy"), history)
    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Training complete.")
    print(f"  Best val loss  : {best_val:.4f}")
    print(f"  Final SSIM     : {history['val_ssim'][-1]:.4f}")
    if df_history:
        print(f"  DeepFace disruption (overall): {np.mean([r for _,r in df_history]):.1%}")
    print(f"  Checkpoints    : {args.checkpoint_dir}")
    print("=" * 70)
    return model, history


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train NoiseNet — Optimised")
    p.add_argument("--data",            type=str,   default=None)
    p.add_argument("--epochs",          type=int,   default=25)
    p.add_argument("--batch",           type=int,   default=16,
                   help="Use 16 or 32 — smaller batch = slower, not faster")
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--epsilon",         type=float, default=0.012)
    p.add_argument("--l2-weight",       type=float, default=0.01,  dest="l2_weight")
    p.add_argument("--disrupt-weight",  type=float, default=0.15,  dest="disrupt_weight")
    p.add_argument("--freq-weight",     type=float, default=0.05,  dest="freq_weight")
    p.add_argument("--checkpoint-dir",  type=str,   default="saved_models/noise_net",
                   dest="checkpoint_dir")
    p.add_argument("--save-every",      type=int,   default=50,    dest="save_every")
    p.add_argument("--resume",          action="store_true",
                   help="Auto-resume from latest checkpoint")
    p.add_argument("--weights",         type=str,   default=None)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--deepface-every",  type=int,   default=50,    dest="deepface_every",
                   help="DeepFace eval every N batches (0=disable)")
    p.add_argument("--deepface-model",  type=str,   default="Facenet",
                   choices=["Facenet", "VGG-Face", "ArcFace", "Facenet512"],
                   dest="deepface_model")
    train(p.parse_args())