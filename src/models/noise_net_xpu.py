"""
NoiseNet - PyTorch + Intel XPU  (OPTIMISED)
============================================
Location : src/models/noise_net_xpu.py

ALL operations run on-device (XPU / CUDA / CPU).
Zero scipy. Zero CPU loops. Zero .numpy() during training.

Innovation layers:
  1. DCT mid-frequency embedding    -> JPEG / screenshot survival
  2. EoT augmentation (vectorized)  -> resize / blur / stretch / color survival
  3. Spectral phase-matching        -> invisible under RGB / XOR / plane forensics
  4. MTF-matched frequency boost    -> partial phone-recapture robustness
  5. Full-image, no face coupling   -> protects any content type
  6. Perceptual frequency loss      -> forces noise into mid-freq only (no color blotches)
  7. Cached SSIM kernel             -> no kernel rebuild per step
  8. Skip connections (U-Net style) -> better gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as tfft
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Device: Intel XPU > CUDA > CPU
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            print("[Device] Intel XPU detected - oneAPI acceleration active")
            return torch.device("xpu"), ipex
    except ImportError:
        pass
    if torch.cuda.is_available():
        print(f"[Device] CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda"), None
    print("[Device] CPU fallback")
    return torch.device("cpu"), None


# ─────────────────────────────────────────────────────────────────────────────
# EoT Augmentation — vectorized, zero .cpu() calls
# Simulates: JPEG, resize/stretch, Gaussian blur, color jitter
# ─────────────────────────────────────────────────────────────────────────────
class EoTAugment(nn.Module):
    def __init__(self,
                 jpeg_range=(60, 95),
                 scale_range=(0.70, 1.30),
                 blur_sigma_max=1.5,
                 color_jitter_strength=0.05):
        super().__init__()
        self.jpeg_range            = jpeg_range
        self.scale_range           = scale_range
        self.blur_sigma_max        = blur_sigma_max
        self.color_jitter_strength = color_jitter_strength

    def _gaussian_kernel(self, sigma: float, device: torch.device) -> torch.Tensor:
        ks = max(3, int(2 * round(2 * sigma) + 1))
        if ks % 2 == 0:
            ks += 1
        coords = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)   # (1,1,ks,ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1. JPEG approximation
        q   = float(np.random.randint(*self.jpeg_range))
        amp = (100.0 - q) / 100.0 * 0.03
        x   = x + torch.randn_like(x) * amp

        # 2. Random resize-back
        scale = float(np.random.uniform(*self.scale_range))
        nh    = max(16, int(H * scale))
        nw    = max(16, int(W * scale))
        x     = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        x     = F.interpolate(x, size=(H,  W),  mode='bilinear', align_corners=False)

        # 3. Gaussian blur
        sigma = float(np.random.uniform(0.0, self.blur_sigma_max))
        if sigma > 0.1:
            k   = self._gaussian_kernel(sigma, device).expand(C, 1, -1, -1)
            pad = k.shape[2] // 2
            x   = F.conv2d(x, k, padding=pad, groups=C)

        # 4. Color jitter
        if self.color_jitter_strength > 0:
            gain = 1.0 + float(np.random.uniform(-self.color_jitter_strength,
                                                   self.color_jitter_strength))
            bias = float(np.random.uniform(-self.color_jitter_strength * 0.5,
                                            self.color_jitter_strength * 0.5))
            x = x * gain + bias

        return x.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Phase Matching
# Noise adopts image FFT phase -> invisible under RGB/XOR/plane analysis
# ─────────────────────────────────────────────────────────────────────────────
def spectral_phase_match(image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    epsilon     = noise.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    img_fft     = tfft.fft2(image)
    noise_fft   = tfft.fft2(noise)
    matched_fft = torch.polar(torch.abs(noise_fft), torch.angle(img_fft))
    matched     = tfft.ifft2(matched_fft).real
    peak        = matched.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    return (matched * (epsilon / peak)).to(dtype=noise.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# MTF Camera Boost
# Boosts noise at 0.04-0.28 cycles/px (survives smartphone optical capture)
# ─────────────────────────────────────────────────────────────────────────────
def mtf_camera_boost(noise: torch.Tensor,
                     low: float = 0.04, high: float = 0.28,
                     boost: float = 3.0, suppress: float = 0.2) -> torch.Tensor:
    device  = noise.device
    epsilon = noise.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    H, W    = noise.shape[-2], noise.shape[-1]

    fy       = tfft.fftfreq(H, device=device)
    fx       = tfft.fftfreq(W, device=device)
    freq_mag = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # IPEX workaround for Intel XPU True/False inversion
    mask = ((freq_mag >= low) & (freq_mag <= high)).to(noise.dtype)
    mult = mask * boost + (1.0 - mask) * suppress
    mult = mult.unsqueeze(0).unsqueeze(0)

    boosted = tfft.ifft2(tfft.fft2(noise) * mult).real
    peak    = boosted.abs().amax(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
    return (boosted * (epsilon / peak)).to(dtype=noise.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# DCT Mid-Frequency Embedding
# Embeds noise at JPEG zigzag indices 6-20 — survives screenshot/compression
# Fully batched via torch.fft, zero scipy, zero block loops
# ─────────────────────────────────────────────────────────────────────────────
_DCT_MAT = None

def _get_dct_matrix(device: torch.device, N: int = 8) -> torch.Tensor:
    global _DCT_MAT
    if _DCT_MAT is None or _DCT_MAT.device != device:
        n = torch.arange(N, device=device, dtype=torch.float32)
        k = n.unsqueeze(-1)
        ans = torch.cos(torch.pi / N * (n + 0.5) * k)
        ans[0] = ans[0] / torch.sqrt(torch.tensor(2.0, device=device))
        _DCT_MAT = ans * torch.sqrt(torch.tensor(2.0 / N, device=device))
    return _DCT_MAT

def _dct2_blocks(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    B, C, H, W = x.shape
    D = _get_dct_matrix(x.device, block).to(x.dtype)
    # Reshape and permute to (B, C, H//8, W//8, 8, 8)
    xb = x.reshape(B, C, H // block, block, W // block, block).permute(0, 1, 2, 4, 3, 5)
    # DCT 2D = D @ xb @ D.T
    dct = D @ xb @ D.transpose(-1, -2)
    return dct.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

def _idct2_blocks(X: torch.Tensor, block: int = 8) -> torch.Tensor:
    B, C, H, W = X.shape
    D = _get_dct_matrix(X.device, block).to(X.dtype)
    Xb = X.reshape(B, C, H // block, block, W // block, block).permute(0, 1, 2, 4, 3, 5)
    # IDCT 2D = D.T @ Xb @ D
    idct = D.transpose(-1, -2) @ Xb @ D
    return idct.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)


_ZIGZAG = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]
_MIDFREQ_MASK_8 = torch.zeros(8, 8)
for _i, (_r, _c) in enumerate(_ZIGZAG):
    if 6 <= _i <= 20:
        _MIDFREQ_MASK_8[_r, _c] = 1.0


def embed_in_dct_midfreq(image: torch.Tensor,
                          noise: torch.Tensor,
                          block_size: int = 8) -> torch.Tensor:
    B, C, H, W = image.shape
    device = image.device

    ph = (block_size - H % block_size) % block_size
    pw = (block_size - W % block_size) % block_size
    img_p   = F.pad(image, (0, pw, 0, ph))
    noise_p = F.pad(noise, (0, pw, 0, ph))

    Hp, Wp    = img_p.shape[2], img_p.shape[3]
    mask_full = _MIDFREQ_MASK_8.to(device)\
                .repeat(Hp // block_size, Wp // block_size)\
                .unsqueeze(0).unsqueeze(0)   # (1,1,Hp,Wp)

    dct_protected = _dct2_blocks(img_p, block_size) + \
                    _dct2_blocks(noise_p, block_size) * mask_full
    return _idct2_blocks(dct_protected, block_size)[:, :, :H, :W].clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# NoiseNet — lean U-Net, NO bottleneck block (bottleneck caused XPU OOM)
# ─────────────────────────────────────────────────────────────────────────────
class NoiseNet(nn.Module):
    """
    Lean U-Net encoder-decoder with 2 skip connections.
    Bottleneck removed — was causing UR_RESULT_ERROR_OUT_OF_RESOURCES on XPU.
    Output: tanh * epsilon  in [-epsilon, +epsilon].
    """

    def __init__(self, epsilon: float = 0.012):
        super().__init__()
        self.epsilon = epsilon

        def _block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.enc1 = _block(3,   64)
        self.enc2 = _block(64,  128)
        self.enc3 = _block(128, 256)

        # Skip: enc2 -> dec4, enc1 -> dec5
        self.dec4 = _block(256 + 128, 128)
        self.dec5 = _block(128 + 64,  64)
        self.out  = nn.Conv2d(64, 3, 3, padding=1)

        self.eot  = EoTAugment()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        inp = self.eot(x) if training else x
        e1  = self.enc1(inp)
        e2  = self.enc2(e1)
        e3  = self.enc3(e2)
        d4  = self.dec4(torch.cat([e3, e2], dim=1))
        d5  = self.dec5(torch.cat([d4, e1], dim=1))
        return torch.tanh(self.out(d5)) * self.epsilon

    @torch.no_grad()
    def protect(self, image: torch.Tensor,
                use_dct: bool = True,
                use_spectral: bool = True,
                use_mtf: bool = True) -> torch.Tensor:
        """Full inference pipeline. image: (B,C,H,W) or (C,H,W) in [0,1]."""
        was_3d = image.dim() == 3
        if was_3d:
            image = image.unsqueeze(0)
        noise = self(image, training=False)
        if use_spectral:
            noise = spectral_phase_match(image, noise)
        if use_mtf:
            noise = mtf_camera_boost(noise)
        out = embed_in_dct_midfreq(image, noise) if use_dct \
              else (image + noise).clamp(0.0, 1.0)
        return out.squeeze(0) if was_3d else out


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions — all on-device
# ─────────────────────────────────────────────────────────────────────────────
class SSIMLoss(nn.Module):
    """Cached SSIM — kernel built once, stored as buffer."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        self.register_buffer("kernel", g.outer(g).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C, pad = x.shape[1], self.window_size // 2
        k = self.kernel.expand(C, 1, self.window_size, self.window_size)
        mu_x  = F.conv2d(x, k, padding=pad, groups=C)
        mu_y  = F.conv2d(y, k, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
        sig_x2 = F.conv2d(x*x, k, padding=pad, groups=C) - mu_x2
        sig_y2 = F.conv2d(y*y, k, padding=pad, groups=C) - mu_y2
        sig_xy = F.conv2d(x*y, k, padding=pad, groups=C) - mu_xy
        num = (2*mu_xy + self.C1) * (2*sig_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sig_x2 + sig_y2 + self.C2)
        return 1.0 - (num / den).mean()


_ssim_module: SSIMLoss = None

def ssim_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _ssim_module
    if _ssim_module is None or _ssim_module.kernel.device != x.device:
        _ssim_module = SSIMLoss().to(x.device)
    return _ssim_module(x, y)


def perceptual_frequency_loss(original: torch.Tensor,
                               perturbed: torch.Tensor) -> torch.Tensor:
    """
    Penalises noise in LOW frequencies (< 0.05 cycles/px).

    Low-freq noise = visible color blotches / smooth yellowish stains on faces.
    This loss forces ALL noise into mid/high frequencies where the human eye
    is effectively blind — the Disney magic: 100% of the adversarial signal
    hidden in microscopic high-frequency textures that look like natural grain.

    Under RGB channel split, XOR, plane shifting — it's indistinguishable
    from sensor noise because it occupies the same frequency bands.

    Single FFT pair computed on XPU — negligible overhead vs SSIM.
    """
    diff_fft = torch.abs(tfft.fft2(original) - tfft.fft2(perturbed))

    H, W   = original.shape[-2], original.shape[-1]
    device = original.device
    fy     = tfft.fftfreq(H, device=device)
    fx     = tfft.fftfreq(W, device=device)
    freq_mag = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Workaround for IPEX comparison operator issues on boolean tensors
    # Explicit float conversion and multiplication handles it properly
    zero_clamp = torch.clamp(freq_mag - 0.05, max=0.0)
    low_mask = (zero_clamp < 0.0).to(original.dtype).unsqueeze(0).unsqueeze(0)
    
    return (diff_fft * low_mask).mean()


def disruption_loss(noise: torch.Tensor) -> torch.Tensor:
    """Maximise spatial gradient -> disrupts GAN texture statistics."""
    dx = noise[:, :, :, 1:] - noise[:, :, :, :-1]
    dy = noise[:, :, 1:, :] - noise[:, :, :-1, :]
    return -(dx.abs().mean() + dy.abs().mean())


def combined_loss(original: torch.Tensor,
                  perturbed: torch.Tensor,
                  noise: torch.Tensor,
                  ssim_w: float    = 1.0,
                  l2_w: float      = 0.01,
                  disrupt_w: float = 0.15,
                  freq_w: float    = 0.05) -> torch.Tensor:
    """
    SSIM       — structural imperceptibility  (primary)
    L2 norm    — keeps perturbation magnitude small
    Disruption — maximises GAN texture confusion
    Freq       — penalises low-freq (visible) noise, forces mid-freq embedding
    """
    return (ssim_w    * ssim_loss(original, perturbed)
            + l2_w    * torch.norm(noise)
            + disrupt_w * disruption_loss(noise)
            + freq_w  * perceptual_frequency_loss(original, perturbed))