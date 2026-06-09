import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from facenet_pytorch import MTCNN
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.video_models_pytorch import VideoCNNLSTM

def get_intel_backend():
    has_ipex = False
    device = torch.device('cpu')
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            device = torch.device('xpu')
            has_ipex = True
    except:
        pass
    return device, has_ipex

def load_video_model(checkpoint_path, device, has_ipex, use_openvino=False, num_frames=8):
    if use_openvino:
        try:
            import openvino as ov
            print(f"📥 Loading OpenVINO IR Model instead of PyTorch...")
            core = ov.Core()
            # If available, use 'GPU' or 'NPU', fallback to 'CPU'
            devices = core.available_devices
            preferred_device = 'CPU'
            for d in ['NPU', 'GPU']:
                if d in devices:
                    preferred_device = d
                    break
            
            print(f"🎯 OpenVINO Target Selected: {preferred_device} (Available: {devices})")
            model_xml = checkpoint_path.replace(".pth", ".xml")
            if not os.path.exists(model_xml):
                print(f"❌ Error: OpenVINO model not found at {model_xml}")
                print("⚠️ Please run export_openvino.py first!")
                exit(1)
                
            model = core.read_model(model_xml)
            # Compile model dynamically
            compiled_model = core.compile_model(model, preferred_device)
            print("🚀 OpenVINO Execution Graph Compiled successfully!")
            return {'openvino': True, 'compiled': compiled_model}
        except ImportError:
            print("❌ OpenVINO not installed. Falling back to PyTorch.")
            
    print("⏳ Loading Video CNN-LSTM Architecture in PyTorch...")
    model = VideoCNNLSTM(hidden_dim=256, lstm_layers=1, bidirectional=False)
    
    print(f"📥 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    if has_ipex and device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)
        print("⚡ Applied IPEX optimizations for inference.")
        
    return model

def predict_video_sequence(video_path, model, device, num_frames=15):
    print(f"\n🎬 Analyzing video sequence: {video_path}")
    
    # 1. Initialize Face Detector
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # 2. Open Video
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if v_len == 0:
        print("❌ Could not read video!")
        return None
        
    step = max(1, v_len // num_frames)
    
    frames_to_process = []
    curr_frame = 0
    
    print("🔍 Extracting face sequences...")
    # Fast read: Use grab() to completely skip decoding useless frames natively in C++
    while len(frames_to_process) < num_frames:
        success = v_cap.grab()
        if not success: 
            break
        
        if curr_frame % step == 0:
            success, frame = v_cap.retrieve()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_to_process.append(Image.fromarray(frame_rgb))
        
        curr_frame += 1
        
    v_cap.release()
    
    if len(frames_to_process) == 0:
        print("⚠️ No frames could be extracted.")
        return None
        
    print("🚀 Running Rapid-Scan MTCNN Detection...")
    
    # ⚡ EXTREME INFERENCE FIX: MTCNN speed is quadratic to resolution.
    # High-Movement videos need detection per frame (Trackers fail on skipped frames).
    # Solution: We severely downscale the frame *only* for the MTCNN calculation,
    # then multiply the coordinate results back to crop from the original 1080p/4K frame!
    fast_detect_frames = []
    scale_factors = []
    MAX_FAST_RES = 360  # Ultra-fast resolution
    
    for img in frames_to_process:
        w, h = img.size
        scale = min(MAX_FAST_RES/w, MAX_FAST_RES/h)
        if scale < 1.0:
            # Drop resolution down to calculate box lightning fast
            new_w, new_h = int(w * scale), int(h * scale)
            fast_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            scale_factors.append(1.0 / scale)  # Restore factor
        else:
            fast_img = img
            scale_factors.append(1.0)
        fast_detect_frames.append(fast_img)
        
    # Fast XPU batch detection on downscaled frames (Should take < 1 second instead of 30)
    boxes_batch, _ = mtcnn.detect(fast_detect_frames)
        
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    face_tensors = []
    if boxes_batch is not None:
        for i, boxes in enumerate(boxes_batch):
            if boxes is not None and len(boxes) > 0:
                # Get the box from the low-res scan and multiply by scale_factors to map to high-res image
                box = boxes[0].tolist()
                scale = scale_factors[i]
                
                # Expand box slightly (10%) to catch any motion blur
                x1, y1 = max(0, box[0]*scale), max(0, box[1]*scale)
                x2, y2 = min(frames_to_process[i].size[0], box[2]*scale), min(frames_to_process[i].size[1], box[3]*scale)
                
                w, h = x2 - x1, y2 - y1
                x1, y1 = max(0, x1 - 0.1*w), max(0, y1 - 0.1*h)
                x2, y2 = min(frames_to_process[i].size[0], x2 + 0.1*w), min(frames_to_process[i].size[1], y2 + 0.1*h)

                face = frames_to_process[i].crop((x1, y1, x2, y2))
                face_tensors.append(transform(face))
                
    if len(face_tensors) == 0:
        print("⚠️ No faces detected in the video sequence!")
        return None
        
    # Pad to required sequence length (pad backwards with last known face)
    while len(face_tensors) < num_frames:
        face_tensors.append(face_tensors[-1])
        
    # Stack into a sequence tensor: Shape [1, Sequence, Channels, Height, Width]
    # We add unsqueeze(0) to simulate a batch size of 1
    sequence_tensor = torch.stack(face_tensors).unsqueeze(0).to(device)
    
    print(f"🤖 Running Temporal LSTM inference on consecutive face sequence...")
    # Checking for OpenVINO model bypass
    if isinstance(model, dict) and 'openvino' in model:
        print("⚡ Executing explicitly via Intel OpenVINO NPU/GPU/CPU Runtime...")
        ov_compiled = model['compiled']
        # Convert PyTorch tensor to numpy for OpenVINO
        ov_input = sequence_tensor.cpu().numpy()
        ov_output = ov_compiled([ov_input])[0]
        prob = torch.sigmoid(torch.tensor(ov_output)).item()
    else:
        with torch.no_grad():
            output = model(sequence_tensor)
            # Convert raw logits to probability using Sigmoid
            prob = torch.sigmoid(output).item()
        
    print("\n" + "="*40)
    # Binary logic: FAKE is 1, REAL is 0 (since we mapped FAKE=1 in our PyTorch dataset)
    if prob >= 0.5:
        print(f"🛑 VERDICT: FAKE! (Confidence: {prob * 100:.2f}%)")
    else:
        print(f"✅ VERDICT: REAL. (Confidence: {(1 - prob) * 100:.2f}%)")
    print("="*40 + "\n")
    
    return prob

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the MP4 video to test")
    parser.add_argument("--ckpt", type=str, default="checkpoints_video/video_best_model.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames tracking in sequence (default 8 since we trained on 8)")
    parser.add_argument("--openvino", action="store_true", help="Execute using OpenVINO for maximum Intel performance")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video '{args.video_path}' not found!")
        exit(1)
        
    if not os.path.exists(args.ckpt) and not args.openvino:
        print(f"❌ Error: Checkpoint '{args.ckpt}' not found!")
        exit(1)
        
    device, has_ipex = get_intel_backend()
    model = load_video_model(args.ckpt, device, has_ipex, use_openvino=args.openvino, num_frames=args.frames)
    
    predict_video_sequence(args.video_path, model, device, num_frames=args.frames)
