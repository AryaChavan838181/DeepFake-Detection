import os
import torch
import librosa
import numpy as np
import argparse
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.audio_models import get_audio_model

def get_intel_backend():
    device = torch.device('cpu')
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = torch.device('xpu')
    except:
        pass
    return device

def preprocess_audio(file_path, max_time_steps=128):
    """
    Extract perfectly dimensioned 128x128 mel-spectrograms from raw wav files.
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"❌ Failed to load audio {file_path}: {e}")
        return None
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    spec_min = log_mel_spec.min()
    spec_max = log_mel_spec.max()
    if spec_max - spec_min > 0:
        normalized = (log_mel_spec - spec_min) / (spec_max - spec_min)
    else:
        normalized = np.zeros_like(log_mel_spec)
        
    if normalized.shape[1] < max_time_steps:
        pad_width = max_time_steps - normalized.shape[1]
        normalized = np.pad(normalized, ((0, 0), (0, pad_width)), mode='constant')
    else:
        start = (normalized.shape[1] - max_time_steps) // 2
        normalized = normalized[:, start:start + max_time_steps]
        
    # [Batch, Channels, Mels, Time] -> [1, 1, 128, 128]
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def predict_audio(file_path, ckpt_path="checkpoints/best_audio_model.pth"):
    device = get_intel_backend()
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at {ckpt_path}. Have you trained the audio model yet?")
        return
        
    model = get_audio_model(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    print(f"\n🎵 Analyzing Audio: {file_path}")
    print(f"⚙️ Execution Environment: {device}")
    
    tensor = preprocess_audio(file_path)
    if tensor is None:
        return
        
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        
    # Training mapping: real = 1, fake = 0
    predicted_fake = prob < 0.5
    
    print("\n" + "="*40)
    if predicted_fake:
        confidence = (1 - prob) * 100
        print(f"🛑 VERDICT: FAKE AUDIO! (Confidence: {confidence:.2f}%)")
    else:
        confidence = prob * 100
        print(f"✅ VERDICT: REAL AUDIO. (Confidence: {confidence:.2f}%)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to the WAV file to test")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_audio_model.pth", help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"❌ Error: Audio file '{args.audio_path}' not found!")
        exit(1)
        
    predict_audio(args.audio_path, args.ckpt)