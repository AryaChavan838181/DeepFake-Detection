import torch
import torch.nn as nn
from torchvision import models

class VideoCNNLSTM(nn.Module):
    """
    Multimodal Temporal 3D/Video implementation using Intel XPU backend.
    Passes a sequence of face frames into EffNet -> sequence into LSTM -> Fake/Real prediction.
    """
    def __init__(self, hidden_dim=256, lstm_layers=1, bidirectional=False, dropout=0.3):
        super(VideoCNNLSTM, self).__init__()
        
        # 1. Base Image feature extractor (EfficientNet-B0 for High Speed on Intel XPU)
        print("⏳ Constructing CNN-LSTM Architecture...")
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # We slice off the final fully connected classifier network
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Ensure modules are sent to the correct device immediately to prevent XPU binding errors
        self.features.to(torch.device('xpu') if torch.xpu.is_available() else torch.device('cpu'))
        self.pool.to(torch.device('xpu') if torch.xpu.is_available() else torch.device('cpu'))
        
        cnn_out_dim = 1280  # EfficientNet-B0 generates a 1280-dimension vector per frame
        
        # 2. Temporal Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,    # Input format: (Batch, Seq_Len, Features)
            bidirectional=bidirectional
        )
        
        # 3. Final Frame Classifier
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)    # Final sigmoid output (1 = FAKE, 0 = REAL)
        )

    def forward(self, x):
        # Input 'x' shape: [Batch, Sequence_Length, Channels, Height, Width]
        b, s, c, h, w = x.size()
        
        # Step 1: Flatten batch and sequence into a single flat batch to massively speed up XPU computation
        x = x.view(b * s, c, h, w)
        
        # Determine device dynamic from x to prevent FloatTensor/XPUFloatType mismatch crashes
        device = x.device
        self.features.to(device)
        self.pool.to(device)
        self.lstm.to(device)
        self.classifier.to(device)

        # Step 2: Extract spatial features frame-by-frame
        # Because we froze the features in training, we wrap this in no_grad() 
        # to ensure zero RAM is wasted on calculating backbone derivatives.
        # Check if the first layer's weights require grad.
        if not next(self.features.parameters()).requires_grad:
            with torch.no_grad():
                x = self.features(x)
                x = self.pool(x)
        else:
            x = self.features(x)
            x = self.pool(x)
        
        # Step 3: Reshape back into [Batch, Sequence, Feature_Dims] for the LSTM
        x = x.view(b, s, -1)
        
        # Step 4: Temporal Processing
        lstm_out, _ = self.lstm(x)
        
        # Step 5: Take the output of the very end of the sequence block 
        # (meaning the LSTM has seen the "entire" video and understands the temporal relationship)
        last_frame_features = lstm_out[:, -1, :] 
        
        # Predict
        out = self.classifier(last_frame_features)
        return out
