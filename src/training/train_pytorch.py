import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        print(f"Skipping IPEX: {e}")
        
    # Check NPU Library
    try:
        import intel_npu_acceleration_library
        has_npu = True
    except Exception as e:
        print(f"Skipping NPU: {e}")
        
    return device, has_ipex, has_npu

def train_model(data_dir, epochs=10, batch_size=32, resume=False):
    device, has_ipex, has_npu = get_intel_backend()
    print(f"✅ PyTorch Training Pipeline initialized.")
    print(f"🎯 Target Device: {device}")
    print(f"🔧 IPEX Available: {has_ipex} | NPU Library Available: {has_npu}")

    # 1. Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(dataset) == 0:
        print("❌ No data found!")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model Initialization
    print("⏳ Loading EfficientNet-B0 backbone...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. Apply Intel Optimizations mutually exclusively so NPU doesn't hijack IPEX memory buffers
    if has_ipex and device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        print("⚡ Applied IPEX optimizations.")
    elif has_npu:
        try:
            import intel_npu_acceleration_library
            from intel_npu_acceleration_library.compiler import CompilerConfig
            config = CompilerConfig()
            print("🧠 Compiling model for Intel NPU Acceleration...")
            model = intel_npu_acceleration_library.compile(model, config)
        except Exception as e:
            print(f"⚠️ NPU compilation failed (falling back to CPU): {e}")

    # 4. Checkpoint Resumption
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if resume and os.path.isfile(latest_ckpt_path):
        print(f"🔄 Resuming from checkpoint: {latest_ckpt_path}")
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"✅ Resumed successfully from epoch {start_epoch}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")

    # 5. Training Loop
    print("\n🚀 Starting Training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar setup
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar with loss
            pbar.set_postfix({'Loss': f"{(running_loss / (i + 1)):.4f}"})
            
        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}\n")
        
        # Save Checkpoints
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        ckpt_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss
        }
        
        # Save Latest
        torch.save(ckpt_state, latest_ckpt_path)
        # Save Best
        if is_best:
            best_ckpt_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(ckpt_state, best_ckpt_path)
            print("🌟 New best model saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_prepared_pt")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs to train")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists")
    args = parser.parse_args()
    
    train_model(args.data, epochs=args.epochs, batch_size=args.batch, resume=args.resume)