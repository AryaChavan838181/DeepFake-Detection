import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.audio_models import get_audio_model

class AudioDeepfakeDataset(Dataset):
    def __init__(self, data_dir, is_train=True, max_time_steps=128):
        self.data_dir = data_dir
        self.is_train = is_train
        self.max_time_steps = max_time_steps
        self.files = []
        self.labels = []
        
        # Mapping mapping real -> 1, fake -> 0
        classes = {'real': 1, 'fake': 0}
        
        for cls_name, cls_label in classes.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Directory not found: {cls_dir}")
                continue
                
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.files.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_label)
                    
        print(f"Loaded {len(self.files)} files from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio using librosa
        try:
            # Load at 16kHz
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Data augmentation for training
            if self.is_train:
                # Add slight noise
                if np.random.rand() < 0.5:
                    noise = np.random.randn(*audio.shape) * 0.005
                    audio = audio + noise
                    
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            spec_min = log_mel_spec.min()
            spec_max = log_mel_spec.max()
            if spec_max - spec_min > 0:
                normalized = (log_mel_spec - spec_min) / (spec_max - spec_min)
            else:
                normalized = np.zeros_like(log_mel_spec)
                
            # Pad or truncate to max_time_steps
            if normalized.shape[1] < self.max_time_steps:
                pad_width = self.max_time_steps - normalized.shape[1]
                normalized = np.pad(normalized, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Random crop if training, else center crop
                if self.is_train:
                    start = np.random.randint(0, normalized.shape[1] - self.max_time_steps + 1)
                else:
                    start = (normalized.shape[1] - self.max_time_steps) // 2
                normalized = normalized[:, start:start + self.max_time_steps]
                
            # Add channel dimension: [1, 128, 128]
            tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            target = torch.tensor([label], dtype=torch.float32)
            
            return tensor, target
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # return dummy data on failure to keep DataLoader going
            return torch.zeros((1, 128, self.max_time_steps)), torch.tensor([0.0])

def get_intel_backend():
    has_ipex = False
    device = torch.device('cpu')
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = torch.device('xpu')
            has_ipex = True
            print("🚀 Intel XPU successfully detected!")
    except ImportError:
        pass
    print(f"🎯 Execution Target: {device}")
    return device, has_ipex

def train_audio_model(data_dir, num_epochs=20, batch_size=32):
    device, has_ipex = get_intel_backend()
    
    print("\n📦 Loading Dataset...")
    full_dataset = AudioDeepfakeDataset(data_dir, is_train=True)
    
    # Simple split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Override is_train for val dataset (it's slightly hacky but works for RandomSplit)
    val_dataset.dataset.is_train = False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_audio_model(device=device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    
    print("\n🔥 Starting Audio Model Training on", device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # For BCEWithLogits, apply sigmoid for predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct_train/total_train:.2f}%'})
            
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        
        print(f"📊 Epoch {epoch+1} Results: Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join('checkpoints', 'best_audio_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saving new best model to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="release_in_the_wild", help="Path to the audio dataset")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    train_audio_model(args.data_dir, num_epochs=args.epochs)