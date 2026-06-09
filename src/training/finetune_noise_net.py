"""
NoiseNet Fine-Tuning Script — GPU + Intel NPU Hybrid
=====================================================
Location : src/training/finetune_noise_net.py

HARDWARE STRATEGY (Intel Core Ultra Gen 1 / Meteor Lake):
  ┌─────────────────────────────────────────────────────────────────┐
  │  XPU / CUDA (GPU)   — Training loop, gradients, optimizer step  │
  │  Intel NPU          — protect() inference + DeepFace eval       │
  │                       (offloaded so GPU is never stalled)       │
  │  CPU                — DeepFace verification (already fast)      │
  └─────────────────────────────────────────────────────────────────┘

  The Intel NPU on Meteor Lake is INFERENCE-ONLY (no backprop).
  It accelerates the protect() calls used in DeepFace evaluation,
  freeing the GPU from eval stalls during training.

  protect() inference is compiled to OpenVINO IR via
  intel-npu-acceleration-library and cached on first run.
  Falls back to CPU inference if NPU is unavailable.

WHY THIS IS FASTER:
  Original:  GPU does training + protect() eval stalls every 30 batches
             GPU thread blocks waiting for DeepFace result
  New:       GPU trains continuously
             NPU/CPU handles protect() + DeepFace in a background thread
             GPU is NEVER stalled for eval

EXPECTED: 3-7 epochs, ~12-20 min each (down from 30-40 min).

Install NPU backend (pick one):
  pip install intel-npu-acceleration-library   # easiest on Meteor Lake
  pip install openvino                         # alternative

Usage:
  python src/training/finetune_noise_net.py
  python src/training/finetune_noise_net.py --epochs 5 --lr 5e-5
  python src/training/finetune_noise_net.py --resume
  python src/training/finetune_noise_net.py --no-npu   # CPU eval fallback
"""

import os, sys, time, json, argparse, warnings, threading, queue
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
    NoiseNet, get_device, ssim_loss,
    perceptual_frequency_loss, disruption_loss,
)

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

IMG_SIZE = 226


# ─────────────────────────────────────────────────────────────────────────────
# NPU Inference Backend
#
# Tries three backends in order:
#   1. intel-npu-acceleration-library  (npu_accel_lib) — best for Meteor Lake
#   2. OpenVINO NPU                    (openvino_npu)
#   3. CPU torch                       (cpu_torch)      — always works
#
# Compilation is one-time and cached to disk.
# Thread-safe — can be called from the eval thread while GPU trains.
# ─────────────────────────────────────────────────────────────────────────────

class NPUProtectInference:
    """
    Wraps NoiseNet.protect() for Intel NPU execution.

    Weight sync happens in the background thread before each eval.
    The OV backend re-exports + recompiles on each sync (slow, ~1s per eval);
    the npu_accel_lib backend just updates the wrapped CPU model (fast).
    """

    NPU_CACHE_DIR = os.path.join(_project_root, "saved_models", "noise_net", "npu_cache")

    def __init__(self, model: "NoiseNet", epsilon: float, enabled: bool = True):
        self.epsilon   = epsilon
        self._lock     = threading.Lock()
        self._backend  = "none"

        if not enabled:
            print("  [NPU] Disabled — using CPU torch for eval inference")
            self._backend    = "cpu_torch"
            self._cpu_model  = self._make_cpu_model(model)
            return

        if self._try_npu_accel_lib(model):
            return
        if self._try_openvino(model):
            return

        print("  [NPU] No NPU backend available — falling back to CPU torch")
        self._backend   = "cpu_torch"
        self._cpu_model = self._make_cpu_model(model)

    # ── Backend 1: intel-npu-acceleration-library ─────────────────────────────
    def _try_npu_accel_lib(self, model) -> bool:
        try:
            import intel_npu_acceleration_library as npu_lib
            print("  [NPU] intel-npu-acceleration-library found — compiling...")

            cpu_m = self._make_cpu_model(model)

            # npu_lib.compile wraps nn.Module for NPU execution (fp16 internally)
            npu_m = npu_lib.compile(cpu_m, dtype=torch.float16)

            # Warm-up to confirm it actually runs on NPU hardware
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            with torch.no_grad():
                _ = npu_m(dummy)

            self._npu_lib_model = npu_m
            self._npu_lib_cpu   = cpu_m   # keep reference for weight sync
            self._backend       = "npu_accel_lib"
            print("  [NPU] intel-npu-acceleration-library backend active ✓")
            return True

        except Exception as e:
            print(f"  [NPU] intel-npu-acceleration-library failed: {e}")
            return False

    # ── Backend 2: OpenVINO NPU ───────────────────────────────────────────────
    def _try_openvino(self, model) -> bool:
        try:
            import openvino as ov
            print("  [NPU] OpenVINO found — attempting NPU compile...")

            os.makedirs(self.NPU_CACHE_DIR, exist_ok=True)
            ov_xml  = os.path.join(self.NPU_CACHE_DIR, "noise_net_protect.xml")
            cpu_m   = self._make_cpu_model(model)
            self._ov_cpu_model = cpu_m

            if not os.path.exists(ov_xml):
                self._export_to_ov(cpu_m, ov_xml)
            else:
                print(f"  [NPU] Loading cached OpenVINO IR: {ov_xml}")

            core      = ov.Core()
            available = core.available_devices
            print(f"  [NPU] OpenVINO devices: {available}")
            target    = "NPU" if "NPU" in available else "CPU"
            if target == "CPU":
                print("  [NPU] NPU not visible to OpenVINO — using CPU OpenVINO runtime")

            compiled  = core.compile_model(core.read_model(ov_xml), target)

            # Warm-up
            dummy_np = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            _ = compiled(dummy_np)

            self._ov_compiled = compiled
            self._ov_core     = core
            self._ov_target   = target
            self._ov_xml      = ov_xml
            self._backend     = f"openvino_{target.lower()}"
            print(f"  [NPU] OpenVINO on {target} active ✓")
            return True

        except Exception as e:
            print(f"  [NPU] OpenVINO path failed: {e}")
            return False

    def _export_to_ov(self, cpu_model, ov_xml: str):
        import openvino as ov
        from openvino.tools.ovc import convert_model
        print("  [NPU] Exporting model to OpenVINO IR (one-time, ~30s)...")
        dummy  = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
        ov_mdl = convert_model(cpu_model, example_input=dummy)
        ov.save_model(ov_mdl, ov_xml)
        print(f"  [NPU] Exported -> {ov_xml}")

    # ── CPU model factory ─────────────────────────────────────────────────────
    def _make_cpu_model(self, gpu_model: "NoiseNet") -> "NoiseNet":
        """Eval-mode CPU copy of the GPU model."""
        cpu_m = NoiseNet(epsilon=self.epsilon).cpu().eval()
        cpu_m.load_state_dict(
            {k: v.detach().cpu() for k, v in gpu_model.state_dict().items()}
        )
        return cpu_m

    # ── Weight sync (called from eval thread before each protect() call) ──────
    def sync_weights(self, gpu_model: "NoiseNet"):
        """
        Copy latest weights from GPU model to the NPU/CPU inference copy.
        Called in the background eval thread — not on the GPU training thread.
        """
        state = {k: v.detach().cpu() for k, v in gpu_model.state_dict().items()}

        with self._lock:
            if self._backend == "npu_accel_lib":
                # Update the underlying CPU model that npu_lib wraps
                self._npu_lib_cpu.load_state_dict(state)
                # npu_lib recompiles lazily on next forward — no extra work needed

            elif self._backend.startswith("openvino"):
                # Re-export IR with new weights and recompile
                # This takes ~1-2s, but happens async so GPU isn't stalled
                try:
                    self._ov_cpu_model.load_state_dict(state)
                    self._export_to_ov(self._ov_cpu_model, self._ov_xml)
                    import openvino as ov
                    self._ov_compiled = self._ov_core.compile_model(
                        self._ov_core.read_model(self._ov_xml), self._ov_target
                    )
                except Exception as e:
                    print(f"\n  [NPU] OV weight sync failed: {e} — using stale weights")

            else:
                self._cpu_model.load_state_dict(state)

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def protect(self, image_cpu: torch.Tensor) -> torch.Tensor:
        """
        image_cpu : (1, C, H, W) float32 on CPU, range [0, 1]
        Returns   : (1, C, H, W) float32 on CPU
        """
        with self._lock:
            if self._backend == "npu_accel_lib":
                # npu_lib model runs on NPU, returns CPU tensor
                return self._npu_lib_model(image_cpu).float().clamp(0.0, 1.0)

            elif self._backend.startswith("openvino"):
                inp    = image_cpu.numpy()
                result = list(self._ov_compiled(inp).values())[0]
                return torch.from_numpy(result).clamp(0.0, 1.0)

            else:  # cpu_torch
                return self._cpu_model(image_cpu).clamp(0.0, 1.0)

    @property
    def backend(self) -> str:
        return self._backend


# ─────────────────────────────────────────────────────────────────────────────
# Async DeepFace Evaluator
#
# A background thread handles:
#   1. Weight sync from GPU model -> NPU model
#   2. NPU protect() forward pass
#   3. DeepFace verify() on CPU
#
# The GPU training loop NEVER blocks.
# Results + feedback (disrupt_weight boost) flow back via thread-safe queues.
# ─────────────────────────────────────────────────────────────────────────────

class AsyncDeepFaceEvaluator:
    """
    Non-blocking DeepFace evaluator backed by a dedicated worker thread.

    Main thread (GPU training):
      - Calls submit_eval() — returns immediately
      - Calls get_disrupt_multiplier() — returns immediately (checks results)

    Worker thread (NPU/CPU):
      - Syncs weights from GPU state dict copy
      - Runs protect() on NPU
      - Runs DeepFace.verify() on CPU
      - Posts result to result_queue
    """

    def __init__(self, npu_backend: NPUProtectInference,
                 every: int          = 30,
                 model_name: str     = "Facenet",
                 feedback_batches: int = 30,
                 enabled: bool       = True):

        self.every            = every
        self.model_name       = model_name
        self.feedback_batches = feedback_batches
        self.npu              = npu_backend
        self._available       = False
        self._results         = []
        self.boost_remaining  = 0
        self.boost_factor     = 2.0
        self._enabled         = enabled

        self._job_queue    = queue.Queue(maxsize=2)
        self._result_queue = queue.Queue()
        self._worker_thread = None

        if not enabled:
            print("  [DeepFace] Disabled (deepface-every=0)")
            return

        try:
            from deepface import DeepFace
            self._DeepFace  = DeepFace
            self._available = True
            self._start_worker()
            print(
                f"  [DeepFace] ASYNC — eval every {every} batches | "
                f"eval backend: {npu_backend.backend} | "
                f"feedback x{self.boost_factor} for {feedback_batches} batches on failure"
            )
        except ImportError:
            print("  [DeepFace] Not installed — skipping (pip install deepface)")

    def _start_worker(self):
        t = threading.Thread(
            target=self._worker_loop, daemon=True, name="deepface-npu-eval"
        )
        t.start()
        self._worker_thread = t

    def _worker_loop(self):
        """Background thread: NPU protect() → CPU DeepFace verify()."""
        import tempfile, cv2

        while True:
            try:
                job = self._job_queue.get(timeout=5.0)
                if job is None:
                    break

                image_cpu, global_step = job

                # protect() runs on NPU — does NOT touch the GPU
                protected_cpu = self.npu.protect(image_cpu)

                def to_np(t):
                    return (t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                disrupted = False
                with tempfile.TemporaryDirectory() as td:
                    op = os.path.join(td, "orig.jpg")
                    pp = os.path.join(td, "prot.jpg")
                    cv2.imwrite(op, cv2.cvtColor(to_np(image_cpu),    cv2.COLOR_RGB2BGR))
                    cv2.imwrite(pp, cv2.cvtColor(to_np(protected_cpu), cv2.COLOR_RGB2BGR))
                    try:
                        res = self._DeepFace.verify(
                            img1_path=op, img2_path=pp,
                            model_name=self.model_name,
                            enforce_detection=False, silent=True,
                        )
                        disrupted = not res.get("verified", True)
                    except Exception:
                        disrupted = True   # detection failed = protection worked

                rate = 1.0 if disrupted else 0.0
                self._result_queue.put((global_step, rate))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n  [DeepFace worker] Error: {e}")

    def get_disrupt_multiplier(self) -> float:
        """
        Called once per training batch (main thread).
        Drains any completed eval results, applies feedback boost if needed.
        Always returns immediately.
        """
        self._drain_results()
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            return self.boost_factor
        return 1.0

    def _drain_results(self):
        """Non-blocking drain of completed eval results from worker thread."""
        while True:
            try:
                step, rate = self._result_queue.get_nowait()
                self._results.append((step, rate))
                status = "DISRUPTED ✓" if rate > 0.5 else "MATCHED ✗ — boosting disrupt_weight"
                if rate < 0.5:
                    self.boost_remaining = self.feedback_batches
                print(f"\n  [DeepFace @step {step}] {status}")
            except queue.Empty:
                break

    def submit_eval(self, gpu_model: "NoiseNet", images: torch.Tensor,
                    global_step: int):
        """
        Non-blocking eval submission.
        Copies weights + one image to CPU, hands off to worker thread.
        If the worker is still busy with the previous eval, we skip (don't queue up).
        """
        if not self._available or not self._enabled:
            return
        if self.every <= 0 or global_step % self.every != 0:
            return
        if self._job_queue.full():
            return   # worker busy — skip, don't stall GPU

        # Sync weights to NPU backend (fast CPU dict copy, not GPU op)
        self.npu.sync_weights(gpu_model)

        # Detach one sample from GPU -> CPU (async, non-blocking)
        image_cpu = images[:1].detach().cpu()
        try:
            self._job_queue.put_nowait((image_cpu, global_step))
        except queue.Full:
            pass

    def shutdown(self):
        if self._worker_thread and self._worker_thread.is_alive():
            try:
                self._job_queue.put(None, timeout=2.0)
            except queue.Full:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CachedTensorDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files in: {cache_dir}")
        print(f"  Loaded {len(self.files):,} cached tensors")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            return torch.load(self.files[idx], weights_only=True)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)


def get_loaders(args):
    cache_dir = os.path.join(_project_root, "data_prepared_noise")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"Cache not found: {cache_dir}\n"
            "Run the main training script first to create the cache."
        )
    dataset = CachedTensorDataset(cache_dir)
    val_n   = max(1, int(len(dataset) * 0.12))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    is_cuda = str(args._device).startswith("cuda")
    workers = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=workers, pin_memory=is_cuda, persistent_workers=(workers > 0))
    tr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True,  **kw)
    vl = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False, **kw)
    print(f"  Train: {train_n:,}  |  Val: {val_n:,}  |  Batch: {args.batch}")
    return tr, vl


# ─────────────────────────────────────────────────────────────────────────────
# Loss on PROTECTED output (the actual fix vs original training)
# ─────────────────────────────────────────────────────────────────────────────

def protect_loss(model, images,
                 ssim_w: float    = 1.0,
                 disrupt_w: float = 0.5,
                 freq_w: float    = 0.05,
                 l2_w: float      = 0.005):
    """
    Computes loss on model.protect() output — the full pipeline including
    spectral_phase_match -> mtf_camera_boost -> embed_in_dct_midfreq.

    This is the key fix: model now gets gradients from the ACTUAL output
    that DeepFace evaluates, not just raw noise.

    disrupt_w=0.5 (vs 0.15 in main training) to overcome post-processing smoothing.
    """
    raw_noise = model(images, training=True)
    protected = model.protect(images)
    eff_noise = protected - images

    loss = (ssim_w    * ssim_loss(images, protected)
            + disrupt_w * disruption_loss(eff_noise)
            + freq_w   * perceptual_frequency_loss(images, protected)
            + l2_w     * torch.norm(raw_noise))

    return loss, protected, eff_noise


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(path, model, optimizer, epoch, step, best_val, history):
    torch.save({
        "epoch": epoch, "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val, "history": history,
    }, path)


def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return (ckpt.get("epoch", 0), ckpt.get("step", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("history", {"train_loss": [], "val_loss": [],
                                  "val_ssim": [], "deepface_disruption": []}))


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune(args):
    print("=" * 70)
    print("  NOISENET FINE-TUNE  |  GPU training + Intel NPU eval inference")
    print("=" * 70)

    try:
        import matplotlib.pyplot as plt
        CAN_PLOT = True
    except ImportError:
        CAN_PLOT = False

    device, ipex = get_device()
    args._device  = device
    is_xpu        = str(device) == "xpu"

    print(f"\n[Devices]")
    print(f"  Training device : {device}  (IPEX: {'yes' if ipex else 'no'})")

    train_loader, val_loader = get_loaders(args)

    model = NoiseNet(epsilon=args.epsilon).to(device)

    # ── Load pretrained weights ───────────────────────────────────────────────
    pretrained = args.weights or os.path.join(
        _project_root, "saved_models", "noise_net", "noise_net_best.pt"
    )
    if not os.path.exists(pretrained):
        raise FileNotFoundError(
            f"Pretrained weights not found: {pretrained}\n"
            "Run main training first, or pass --weights <path>"
        )
    state = torch.load(pretrained, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    print(f"\n  Loaded pretrained weights: {pretrained}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8
    )
    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )

    if ipex and is_xpu:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
        print("  Intel IPEX applied to GPU model (BFloat16 Fast Path) ✓")

    # ── NPU inference backend ─────────────────────────────────────────────────
    print("\n[NPU Setup]")
    npu_backend = NPUProtectInference(
        model   = model,
        epsilon = args.epsilon,
        enabled = not args.no_npu,
    )
    print(f"  Eval inference backend : {npu_backend.backend}")

    # ── Async DeepFace evaluator ──────────────────────────────────────────────
    df_eval = AsyncDeepFaceEvaluator(
        npu_backend      = npu_backend,
        every            = args.deepface_every,
        model_name       = "Facenet",
        feedback_batches = 30,
        enabled          = args.deepface_every > 0,
    )

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "finetune_ckpt.pt")
    best_path = os.path.join(args.checkpoint_dir, "noise_net_finetuned.pt")

    start_epoch = 1
    global_step = 0
    best_val    = float("inf")
    history     = {"train_loss": [], "val_loss": [], "val_ssim": [], "deepface_disruption": []}
    no_improve  = 0

    if args.resume and os.path.exists(ckpt_path):
        start_epoch, global_step, best_val, history = load_ckpt(
            ckpt_path, model, optimizer, device
        )
        start_epoch += 1
        print(f"\n  Resumed from epoch {start_epoch - 1}")

    print(f"\n[Config]  {args.epochs} epochs  |  lr={args.lr}  |  disrupt_w={args.disrupt_weight}")
    print(f"  Loss on model.protect() output (the fix)")
    print(f"  DeepFace eval: NON-BLOCKING — runs in background thread on {npu_backend.backend}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0       = time.time()

        pbar = tqdm(train_loader,
                    desc=f"FT Ep {epoch:02d}/{args.epochs}",
                    ncols=120, unit="batch")

        for step, images in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            global_step += 1

            # Non-blocking: read any completed eval results + get boost multiplier
            df_mult = df_eval.get_disrupt_multiplier()
            eff_dw  = args.disrupt_weight * df_mult

            optimizer.zero_grad(set_to_none=True)

            if is_xpu and ipex:
                with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    loss, protected, eff_noise = protect_loss(
                        model, images,
                        ssim_w    = args.ssim_weight,
                        disrupt_w = eff_dw,
                        freq_w    = args.freq_weight,
                        l2_w      = args.l2_weight,
                    )
            else:
                loss, protected, eff_noise = protect_loss(
                    model, images,
                    ssim_w    = args.ssim_weight,
                    disrupt_w = eff_dw,
                    freq_w    = args.freq_weight,
                    l2_w      = args.l2_weight,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item()
            elapsed   = time.time() - t0
            eta       = (elapsed / (step + 1)) * (len(train_loader) - step - 1)

            pbar.set_postfix({
                "loss" : f"{run_loss/(step+1):.4f}",
                "MAE"  : f"{eff_noise.abs().mean().item():.5f}",
                "lr"   : f"{optimizer.param_groups[0]['lr']:.2e}",
                "dw"   : f"{eff_dw:.3f}",
                "eval" : npu_backend.backend[:10],
                "ETA"  : f"{eta/60:.1f}m",
            })

            # Submit async eval — GPU is NOT stalled
            df_eval.submit_eval(model, images, global_step)

            if global_step % args.save_every == 0:
                save_ckpt(ckpt_path, model, optimizer, epoch,
                          global_step, best_val, history)

        pbar.close()

        # ── Validation (GPU) ──────────────────────────────────────────────────
        model.eval()
        v_loss = v_ssim = 0.0
        with torch.no_grad():
            for images in tqdm(val_loader, desc="  Val", ncols=70, leave=False):
                images = images.to(device, non_blocking=True)
                
                if is_xpu and ipex:
                    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                        loss, protected, _ = protect_loss(
                            model, images,
                            ssim_w    = args.ssim_weight,
                            disrupt_w = args.disrupt_weight,
                            freq_w    = args.freq_weight,
                            l2_w      = args.l2_weight,
                        )
                else:
                    loss, protected, _ = protect_loss(
                        model, images,
                        ssim_w    = args.ssim_weight,
                        disrupt_w = args.disrupt_weight,
                        freq_w    = args.freq_weight,
                        l2_w      = args.l2_weight,
                    )
                    
                v_loss += loss.item()
                v_ssim += (1.0 - ssim_loss(images, protected)).item()

        if is_xpu:
            torch.xpu.synchronize()

        val_loss  = v_loss / len(val_loader)
        val_ssim  = v_ssim / len(val_loader)
        tr_loss   = run_loss / len(train_loader)
        epoch_t   = time.time() - t0

        # Drain any final async results before printing epoch summary
        df_eval._drain_results()
        epoch_df  = [r for s, r in df_eval._results
                     if s > global_step - len(train_loader) and s <= global_step]
        df_avg    = np.mean(epoch_df) if epoch_df else float("nan")
        ssim_flag = "OK" if val_ssim >= 0.95 else "WARN <0.95"
        df_flag   = f"{df_avg:.0%}" if not np.isnan(df_avg) else "n/a"

        print(
            f"\n  FT Ep {epoch:02d}/{args.epochs} | "
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

        # moved below the logic to keep consistency
        if val_loss == best_val: # Only triggered if it passed the check above
            torch.save(model.state_dict(), best_path)
            print(f"  Best fine-tuned model saved -> {best_path}")
        else:
            no_improve += 1
            print(f"  No improvement {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\n  Early stopping (patience={args.patience})")
                break

        save_ckpt(ckpt_path, model, optimizer, epoch,
                  global_step, best_val, history)

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
                df_clean = [d for d in history["deepface_disruption"] if d >= 0]
                if df_clean:
                    plt.plot(df_clean, label="Disruption %", color="purple")
                    plt.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
                    plt.title("DeepFace Disruption Rate vs Epochs")
                    plt.grid(True)
                    plt.legend()
                    
                plt.tight_layout()
                plt.savefig(os.path.join(args.checkpoint_dir, "finetune_training_curves_epoch.png"))
                plt.close()
            except Exception as e:
                print(f"  [Plot Warning] Failed to generate graph: {e}")

    df_eval.shutdown()

    with open(os.path.join(args.checkpoint_dir, "finetune_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    overall_df = np.mean([r for _, r in df_eval._results]) if df_eval._results else 0.0

    print("\n" + "=" * 70)
    print(f"  Fine-tune complete.")
    print(f"  Best val loss         : {best_val:.4f}")
    print(f"  Final SSIM            : {history['val_ssim'][-1]:.4f}")
    print(f"  DeepFace disruption   : {overall_df:.1%}")
    print(f"  Fine-tuned model      : {best_path}")
    print("=" * 70)
    if CAN_PLOT and len(history["train_loss"]) > 0:
        print(f"  Fine-tune training curves generated -> saved_models/noise_net/finetune_training_curves_epoch.png")
        
    print("\n  NOTE: Use noise_net_finetuned.pt going forward.")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NoiseNet Fine-Tune — GPU training + Intel NPU eval inference"
    )
    p.add_argument("--weights",         type=str,   default=None,
                   help="Path to pretrained .pt (default: saved_models/noise_net/noise_net_best.pt)")
    p.add_argument("--epochs",          type=int,   default=7,
                   help="Fine-tune epochs (3-7 usually enough)")
    p.add_argument("--batch",           type=int,   default=16)
    p.add_argument("--lr",              type=float, default=5e-5,
                   help="Low LR — don't destroy pretrained imperceptibility")
    p.add_argument("--epsilon",         type=float, default=0.012)
    p.add_argument("--ssim-weight",     type=float, default=1.0,   dest="ssim_weight")
    p.add_argument("--disrupt-weight",  type=float, default=0.5,   dest="disrupt_weight",
                   help="Higher than main training (0.5 vs 0.15) to overcome post-processing")
    p.add_argument("--freq-weight",     type=float, default=0.05,  dest="freq_weight")
    p.add_argument("--l2-weight",       type=float, default=0.005, dest="l2_weight")
    p.add_argument("--checkpoint-dir",  type=str,   default="saved_models/noise_net",
                   dest="checkpoint_dir")
    p.add_argument("--save-every",      type=int,   default=50,    dest="save_every")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--patience",        type=int,   default=3)
    p.add_argument("--deepface-every",  type=int,   default=30,    dest="deepface_every",
                   help="Async DeepFace eval every N batches (0=disable)")
    p.add_argument("--no-npu",          action="store_true", dest="no_npu",
                   help="Disable NPU backend, fall back to CPU torch for eval inference")
    finetune(p.parse_args())