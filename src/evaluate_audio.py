import os
import sys
import time
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.audio_models import get_audio_model

def get_intel_backend():
    device = torch.device('cpu')
    has_ipex = False
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = torch.device('xpu')
            has_ipex = True
    except:
        pass
    return device

def preprocess_audio(file_path, max_time_steps=128):
    """
    Extract perfectly dimensioned 128x128 mel-spectrograms from raw wav/mp3 files.
    """
    import warnings
    # Suppress librosa PySoundFile warning for mp3s if ffmpeg is fallback
    warnings.filterwarnings("ignore", category=UserWarning)
    
    try:
        audio, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"\n❌ Failed to load audio {file_path}: {e}")
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
        
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def evaluate_folder(data_dir, ckpt_path):
    device = get_intel_backend()
    print(f"⚙️ Execution Environment: {device}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at {ckpt_path}.")
        return

    print(f"📥 Loading best audio model from {ckpt_path}...")
    model = get_audio_model(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    allowed_extensions = ('.wav', '.mp3')
    files_to_eval = []
    
    # Map classes based on folder name: fake = 0, real = 1
    for cls_name, cls_label in {'fake': 0, 'real': 1}.items():
        folder_path = os.path.join(data_dir, cls_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder not found: {folder_path}")
            continue
            
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(allowed_extensions):
                files_to_eval.append((os.path.join(folder_path, fname), cls_label))
                
    if len(files_to_eval) == 0:
        print(f"❌ No .wav or .mp3 files found in {data_dir}/fake or {data_dir}/real")
        return

    print(f"🔍 Found {len(files_to_eval)} audio files to evaluate in '{data_dir}'.")
    
    y_true = []
    y_pred = []
    total_time = 0.0
    
    with torch.no_grad():
        for file_path, true_label in tqdm(files_to_eval, desc="Evaluating Audio Files"):
            start_time = time.time()
            
            tensor = preprocess_audio(file_path)
            if tensor is None:
                continue
                
            tensor = tensor.to(device)
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            
            # Prediction: < 0.5 is Fake (0), >= 0.5 is Real (1)
            pred_label = 1 if prob >= 0.5 else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            total_time += (time.time() - start_time)

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    avg_inf_time = total_time / len(y_true) if len(y_true) > 0 else 0

    print("\n" + "="*50)
    print(" 📊 AUDIO MODEL EVALUATION REPORT ")
    print("="*50)
    print(f"Dataset             : {data_dir}")
    print(f"Total Files Tested  : {len(y_true)}")
    print(f"Global Accuracy     : {accuracy*100:.2f}%")
    print(f"Precision (Real)    : {precision*100:.2f}%")
    print(f"Recall (Real)       : {recall*100:.2f}%")
    print(f"F1-Score            : {f1*100:.2f}%")
    print(f"Avg Inference Time  : {avg_inf_time:.3f} seconds/audio")
    
    print("\n[Confusion Matrix] ")
    print(f"                 Predicted Fake(0)   Predicted Real(1)")
    print(f"Actual Fake(0): {cm[0][0]:^17} | {cm[0][1]:^17}")
    print(f"Actual Real(1): {cm[1][0]:^17} | {cm[1][1]:^17}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Audio Model on an entire directory of Real/Fake wavs and mp3s")
    parser.add_argument("--data_dir", type=str, default="release_in_the_wild", help="Path to evaluation dataset (must contain 'real' and 'fake' subfolders)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_audio_model.pth", help="Path to the trained model checkpoint")
    
    args = parser.parse_args()
    evaluate_folder(args.data_dir, args.ckpt)
