import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image

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

def load_model(checkpoint_path, device, has_ipex):
    print("⏳ Loading EfficientNet-B0 model architecture...")
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    print(f"📥 Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Handle the fact that model might have been saved in IPEX format
    # state_dict is inside 'model_state' key based on our train script
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

def predict_video(video_path, model, device, num_frames=10):
    print(f"\n🎬 Analyzing video: {video_path}")
    
    # 1. Initialize Face Detector
    # MTCNN handles face detection
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # 2. Open Video
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if v_len == 0:
        print("❌ Could not read video!")
        return None
        
    # Calculate which frames to pick uniformly across the video
    step = max(1, v_len // num_frames)
    frames_to_process = [i * step for i in range(num_frames) if i * step < v_len]
    
    # 3. Model Preprocessing Transform
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    face_tensors = []
    
    print("🔍 Extracting faces...")
    for frame_idx in frames_to_process:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = v_cap.read()
        if not success:
            continue
            
        # Convert BGR to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Detect Face
        # MTCNN returns a cropped face tensor if return_prob=False, but here we'll just get the box
        boxes, _ = mtcnn.detect(pil_img)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0].tolist() # Take the dominant face
            face = pil_img.crop((box[0], box[1], box[2], box[3]))
            
            # Apply our exact training transforms
            face_tensor = transform(face)
            face_tensors.append(face_tensor)

    v_cap.release()
    
    if len(face_tensors) == 0:
        print("⚠️ No faces detected in the video!")
        return None
        
    # 4. Predict
    # Stack faces into a batch: Shape [N, 3, 226, 226]
    batch_tensors = torch.stack(face_tensors).to(device)
    
    print(f"🤖 Running AI inference on {len(batch_tensors)} faces...")
    with torch.no_grad():
        outputs = model(batch_tensors)
        # Convert raw logits to probabilities using Sigmoid
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # Handle single face vs multiple faces shape discrepancy
    if probs.ndim == 0:
        probs = [probs.item()]
        
    avg_prob = sum(probs) / len(probs)
    
    print("\n" + "="*40)
    # PyTorch's ImageFolder mapped FAKE to class 0 and REAL to class 1 alphabetically based on subfolder names.
    # Therefore, close to 0 = Fake, close to 1 = Real.
    if avg_prob < 0.5:
        print(f"🛑 VERDICT: FAKE! (Confidence: {(1 - avg_prob) * 100:.2f}%)")
    else:
        print(f"✅ VERDICT: REAL. (Confidence: {avg_prob * 100:.2f}%)")
    print("="*40 + "\n")
    
    return avg_prob

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the MP4 video to test")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to extract for analysis")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video '{args.video_path}' not found!")
        exit(1)
        
    if not os.path.exists(args.ckpt):
        print(f"❌ Error: Checkpoint '{args.ckpt}' not found!")
        exit(1)
        
    device, has_ipex = get_intel_backend()
    model = load_model(args.ckpt, device, has_ipex)
    
    predict_video(args.video_path, model, device, num_frames=args.frames)
