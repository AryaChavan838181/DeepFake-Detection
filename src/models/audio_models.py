import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSpectrogramCNN(nn.Module):
    """
    Deepfake audio detector using 2D CNN on mel-spectrogram representations.
    Input: (batch_size, channels=1, mel_bins=128, time_steps=128)
    Output: (batch_size, 1) - binary prediction
    """
    def __init__(self):
        super(AudioSpectrogramCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.3)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.3)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.4)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(256, 512)
        self.drop_fc1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.drop_fc2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.drop_fc3 = nn.Dropout(0.2)
        
        # Logits output (No sigmoid so we can use BCEWithLogitsLoss for numerical stability)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: [B, 1, 128, 128]
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool4(F.relu(self.bn4(self.conv4(x)))))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        x = self.drop_fc1(F.relu(self.fc1(x)))
        x = self.drop_fc2(F.relu(self.fc2(x)))
        x = self.drop_fc3(F.relu(self.fc3(x)))
        
        x = self.out(x)
        return x

def get_audio_model(device=None):
    model = AudioSpectrogramCNN()
    if device is not None:
        model = model.to(device)
    return model

if __name__ == "__main__":
    print("Testing PyTorch Audio Model...")
    model = AudioSpectrogramCNN()
    # Dummy input: Batch=2, Channel=1, Mels=128, TimeFrames=128
    dummy_input = torch.randn(2, 1, 128, 128)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")  # should be [2, 1]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
