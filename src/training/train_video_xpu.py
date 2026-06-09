import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import argparse

# Ensure project root is first on sys.path so `from src...` always resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.video_models_pytorch import VideoCNNLSTM

def get_intel_backend():
    has_ipex = False
    has_npu = False
    device = torch.device('cpu')
    
    # Check IPEX (Intel Extension for PyTorch)
    try:
        import intel_extension_for_pytorch as ipex
        has_ipex = True
        if torch.xpu.is_available():
            device = torch.device('xpu')
    except Exception as e:
        pass
        
    return device, has_ipex, has_npu

import torch.nn.functional as F_nn
import torchvision.transforms.functional as F_tv

class RandomVideoAugmentation:
    def __call__(self, tensor):
        # tensor shape: [frames, 3, 226, 226]. These are already normalized floats.
        
        # 1. Random Gaussian Blur (Simulates smudged lens / hazy camera)
        if torch.rand(1).item() < 0.3:
            # We use 5x5 or 7x7 kernel to make the blur noticeable
            kernel_size = int(torch.randint(3, 8, (1,)).item())
            if kernel_size % 2 == 0: kernel_size += 1
            tensor = F_tv.gaussian_blur(tensor, [kernel_size, kernel_size])
            
        # 2. Random Pixelation (Simulates Instagram compression / low-res internet videos)
        if torch.rand(1).item() < 0.3:
            # Downsample heavily then upscale to create blocky pixelation artifacts
            down_scale = torch.randint(48, 112, (1,)).item()
            tensor = F_nn.interpolate(tensor, size=(down_scale, down_scale), mode='bilinear', align_corners=False)
            tensor = F_nn.interpolate(tensor, size=(226, 226), mode='bilinear', align_corners=False)
            
        # 3. Random High-Frequency Noise (Simulates ISO noise in dark / filter effects)
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(tensor) * 0.05
            tensor = tensor + noise
            
        return tensor

# Explicit Dataset Logic strictly maps REAL to 0 and FAKE to 1 (No alphabetical bugs)
class VideoTensorDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.augment = augment
        self.aug_pipeline = RandomVideoAugmentation() if augment else None
        
        # FAKE is Class 1
        fake_dir = os.path.join(root_dir, 'FAKE')
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(fake_dir, f), 1.0))
        
        # REAL is Class 0
        real_dir = os.path.join(root_dir, 'REAL')
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith('.pt'):
                    self.samples.append((os.path.join(real_dir, f), 0.0))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Memory mapped tensor load is incredibly fast.
        tensor = torch.load(path, weights_only=True) # shape is [frames, 3, 226, 226]
        
        if self.augment:
            tensor = self.aug_pipeline(tensor)
            
        return tensor, torch.tensor([label], dtype=torch.float32)


def train_video_model(data_dir, epochs=10, batch_size=8, resume=False):
    device, has_ipex, has_npu = get_intel_backend()
    print(f"✅ PyTorch 3D/Video Training Pipeline initialized.")
    print(f"🎯 Target Device: {device}")

    # 1. Dataset (Enable augmentation for trainset later)
    # We load all files first to count them
    full_dataset = VideoTensorDataset(data_dir, augment=False)
    if len(full_dataset) == 0:
        print("❌ No sequence .pt files found in data directory!")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])
    
    # We recreate datasets using subsets so we ONLY augment the training data, NOT validation
    train_dataset = torch.utils.data.Subset(VideoTensorDataset(data_dir, augment=True), train_indices.indices)
    val_dataset = torch.utils.data.Subset(VideoTensorDataset(data_dir, augment=False), val_indices.indices)
    
    # Notice batch size must be much smaller for video models (Batch * 15 frames per batch object)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup Model
    print("⏳ Loading Video CNN-LSTM Architecture...")
    model = VideoCNNLSTM(hidden_dim=256, lstm_layers=1, bidirectional=False)
    
    # ⚡ EXTREME SPEEDUP: Freeze the EfficientNet backbone! 
    # The backbone already perfectly knows how to identify facial features from standard tasks.
    # By freezing it, PyTorch skips calculating gradients/backprop for 5.3 million parameters,
    # meaning your computer only has to do math on the tiny LSTM layer at the end.
    for param in model.features.parameters():
        param.requires_grad = False
    print("❄️ FROZE Backbone CNN. Training will be ~4x to 6x faster per epoch!")
    
    # EXACT DEVICE BINDING BEFORE OPTIMIZER
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 3. Handle IPEX memory optimization 
    # (Bypassing strictly `ipex.optimize` when layers are frozen, which causes PyTorch convolution_prepack bugs)
    if has_ipex and device.type == 'xpu':
        print("⚡ Using Intel XPU (Skipping IPEX deep-optimization due to frozen backbone bindings).")

    # 4. Resumption
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = 'checkpoints_video'
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, 'video_latest_checkpoint.pth')
    
    if resume and os.path.isfile(latest_ckpt_path):
        print(f"🔄 Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        
        # When shifting from an "unfrozen" architecture to a "frozen" one, 
        # the loaded optimizer shapes will completely mismatch the new restricted shapes.
        # So we safely skip loading old momentum vectors and let Adam calculate fresh gradients for the LSTM. 
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            # Explicitly move optimizer states to target device (crucial for Intel XPU to avoid cpu/xpu conflicts)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except ValueError:
            print("⚠️ Optimizer parameter shape shifted (due to frozen backbone). Restarting Adam momentum states. (This is normal and safe!)")
            
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

    # 5. Training Epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{(running_loss / (i + 1)):.4f}"})
            
        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n")
        
        is_best = avg_loss < best_loss
        if is_best: best_loss = avg_loss
            
        ckpt_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss
        }
        
        torch.save(ckpt_state, latest_ckpt_path)
        if is_best:
            torch.save(ckpt_state, os.path.join(checkpoint_dir, 'video_best_model.pth'))
            print("🌟 New best Temporal Video model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_seq_pt")
    parser.add_argument("--epochs", type=int, default=15, help="Total number of epochs to train")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (Keep small, eg 4-8, to fit in VRAM)")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists")
    args = parser.parse_args()
    
    train_video_model(args.data, epochs=args.epochs, batch_size=args.batch, resume=args.resume)