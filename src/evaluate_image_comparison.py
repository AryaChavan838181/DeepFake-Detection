import os
import torch
import random
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import argparse
import sys

# Append project root just in case
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Helper function to load the legacy PyTorch 2D Model (EfficientNet)
def load_legacy_2d_model(ckpt_path, device):
    print("⏳ Loading Legacy 2D EfficientNet PyTorch Model...")
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

# Helper function to load the new optimized 3D OpenVINO Model
def load_openvino_model(xml_path):
    print("⏳ Loading Modern OpenVINO Video CNN-LSTM Graph...")
    try:
        import openvino as ov
        core = ov.Core()
        # Grab best available Intel silicon
        preferred_device = 'CPU'
        devices = core.available_devices
        for d in ['NPU', 'GPU']:
            if d in devices:
                preferred_device = d
                break
        
        model_ir = core.read_model(xml_path)
        compiled_model = core.compile_model(model_ir, preferred_device)
        print(f"✅ OpenVINO active on [{preferred_device}]")
        return compiled_model
    except ImportError:
        print("❌ OpenVINO is not installed. Please run: pip install openvino")
        exit(1)

def evaluate_models(data_dir, model_2d, model_3d_ov, device, num_frames=8, max_samples=500):
    print(f"\n🚀 Starting Dual-Model Accuracy Race")
    print("="*60)
    
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    # 1. Grab Images
    real_dir = os.path.join(data_dir, "REAL")
    fake_dir = os.path.join(data_dir, "FAKE")
    
    reals = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    fakes = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    sample_size = min(len(reals), len(fakes), max_samples // 2)
    random.shuffle(reals)
    random.shuffle(fakes)
    
    test_suite = []
    # Using String labels to avoid confusion
    for r in reals[:sample_size]: test_suite.append((r, "REAL"))
    for f in fakes[:sample_size]: test_suite.append((f, "FAKE"))
    random.shuffle(test_suite)
    
    total = len(test_suite)
    print(f"📊 Dataset Loaded: {total} extracted frames (50% REAL, 50% FAKE) directly from {data_dir}\n")
    
    # Trackers
    correct_2d = 0
    correct_3d = 0
    
    pbar = tqdm(test_suite, desc="Race in Progress", unit="img")
    for img_path, true_label in pbar:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor_2d = transform(img).unsqueeze(0).to(device)  # Shape: [1, 3, 226, 226]
        except:
            total -= 1
            continue
            
        # ==========================================
        # 🏎️ RUN MODEL 1: LEGACY 2D EFFICIENTNET
        # ==========================================
        with torch.no_grad():
            output_2d = model_2d(tensor_2d)
            prob_2d = torch.sigmoid(output_2d).item()
        
        # In PyTorch ImageFolder, FAKE was folder 0, REAL was folder 1
        predicted_fake_2d = prob_2d < 0.5 
        if (predicted_fake_2d and true_label == "FAKE") or (not predicted_fake_2d and true_label == "REAL"):
            correct_2d += 1
            
        # ==========================================
        # 🏎️ RUN MODEL 2: OPENVINO 3D CNN-LSTM
        # ==========================================
        # Simulate sequence by repeating the same frame
        tensor_3d = tensor_2d.repeat(1, num_frames, 1, 1, 1).cpu().numpy() # [1, 8, 3, 226, 226]
        
        output_3d = model_3d_ov([tensor_3d])[0]
        prob_3d = torch.sigmoid(torch.tensor(output_3d)).item()
        
        # In Video Sequences, we mapped REAL=0, FAKE=1
        predicted_fake_3d = prob_3d >= 0.5
        if (predicted_fake_3d and true_label == "FAKE") or (not predicted_fake_3d and true_label == "REAL"):
            correct_3d += 1
            
        # Update progress bar
        acc_2d = (correct_2d / (pbar.n + 1)) * 100
        acc_3d = (correct_3d / (pbar.n + 1)) * 100
        pbar.set_postfix({"2D_Acc": f"{acc_2d:.1f}%", "3D_Acc": f"{acc_3d:.1f}%"})

    if total == 0:
        print("❌ No images could be evaluated.")
        return

    acc_2d_final = (correct_2d / total) * 100
    acc_3d_final = (correct_3d / total) * 100
    
    print("\n" + "="*60)
    print("🏆 FINAL CHAMPIONSHIP RESULTS")
    print("="*60)
    print(f"Total Test Images:\t{total} cropped faces")
    print("-" * 60)
    print(f"Legacy 2D PyTorch Model:\t{acc_2d_final:.2f}% Accuracy ({correct_2d}/{total} correct)")
    print(f"Modern OpenVINO 3D Model:\t{acc_3d_final:.2f}% Accuracy ({correct_3d}/{total} correct)")
    print("="*60)
    
    if acc_3d_final > acc_2d_final:
        print("🎉 The OpenVINO / Video-Sequence Model is fundamentally superior even for static images!")
    elif acc_2d_final > acc_3d_final:
        print("⚠️ The PyTorch 2D Model retained stronger spatial boundaries for single frames.")
    else:
        print("🤝 It's a dead tie on static topological analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_prepared_pt", help="Path to cropped JPG face dataset")
    parser.add_argument("--ckpt_2d", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--ckpt_3d", type=str, default="openvino_model/video_model.xml")
    parser.add_argument("--max_samples", type=int, default=1000, help="Evaluate up to N random images total")
    args = parser.parse_args()

    # Intel Device Fetch for PyTorch model
    device = torch.device('cpu')
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available(): device = torch.device('xpu')
    except: pass
    
    print("Booting Evaluation Matrix...")
    m_2d = load_legacy_2d_model(args.ckpt_2d, device)
    m_3d = load_openvino_model(args.ckpt_3d)
    
    evaluate_models(args.data, m_2d, m_3d, device, max_samples=args.max_samples)