import os
import torch
import argparse
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms, models
import torch.nn as nn
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

def load_image_model(ckpt_path, device, use_openvino=False):
    if use_openvino:
        try:
            import openvino as ov
            print(f"📥 Loading OpenVINO Image IR Model instead of PyTorch...")
            core = ov.Core()
            # If available, use 'NPU' or 'GPU', fallback to 'CPU'
            devices = core.available_devices
            preferred_device = 'CPU'
            for d in ['NPU', 'GPU']:
                if d in devices:
                    preferred_device = d
                    break
            
            print(f"🎯 OpenVINO Target Selected: {preferred_device} (Available: {devices})")
            
            # Switch pure .pth extension to OpenVINO XML path layout
            if ckpt_path.endswith('.pth'):
                model_xml = os.path.join("openvino_model", "image_model.xml")
            else:
                model_xml = ckpt_path
                
            if not os.path.exists(model_xml):
                print(f"❌ Error: OpenVINO image model not found at {model_xml}")
                print("⚠️ Please run export_openvino_image.py first!")
                exit(1)
                
            model = core.read_model(model_xml)
            compiled_model = core.compile_model(model, preferred_device)
            print("🚀 OpenVINO 2D Image Graph Compiled successfully!")
            return {'openvino': True, 'compiled': compiled_model}
        except ImportError:
            print("❌ OpenVINO not installed. Falling back to PyTorch.")
            
    print("⏳ Loading Legacy 2D EfficientNet PyTorch Model...")
    
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    print(f"📥 Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    return model

def predict_single_image(image_path, model, device):
    print(f"\n🖼️ Analyzing Image: {image_path}")
    
    # 1. Initialize Face Detector
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # 2. Load Image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Could not read image: {e}")
        return None
        
    print("🔍 Extracting face via MTCNN...")
    # Fast detection
    boxes, _ = mtcnn.detect(img)
    
    if boxes is None or len(boxes) == 0:
        print("⚠️ No faces detected in the image!")
        return None
        
    # Grab the first face box and expand it slightly (10%)
    box = boxes[0].tolist()
    w, h = box[2] - box[0], box[3] - box[1]
    
    x1, y1 = max(0, box[0] - 0.1*w), max(0, box[1] - 0.1*h)
    x2, y2 = min(img.size[0], box[2] + 0.1*w), min(img.size[1], box[3] + 0.1*h)
    
    face = img.crop((x1, y1, x2, y2))
    
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    
    face_tensor = transform(face)
    
    # Add batch dimension: [Batch, Channels, Height, Width]
    tensor_2d = face_tensor.unsqueeze(0).to(device)
    
    print(f"🤖 Running EfficientNet-B0 Inference on extracted face...")
    
    # Checking for OpenVINO model bypass
    if isinstance(model, dict) and 'openvino' in model:
        print("⚡ Executing explicitly via Intel OpenVINO...")
        ov_compiled = model['compiled']
        ov_input = tensor_2d.cpu().numpy()
        ov_output = ov_compiled([ov_input])[0]
        prob = torch.sigmoid(torch.tensor(ov_output)).item()
    else:
        with torch.no_grad():
            output = model(tensor_2d)
            prob = torch.sigmoid(output).item()
            
    # For PyTorch ImageFolder, Fake=0 and Real=1 based on folder names
    # So close to 0 is FAKE, close to 1 is REAL
    predicted_fake = prob < 0.5
    
    print("\n" + "="*40)
    if predicted_fake:
        confidence = (1 - prob) * 100
        print(f"🛑 VERDICT: FAKE! (Confidence: {confidence:.2f}%)")
    else:
        confidence = prob * 100
        print(f"✅ VERDICT: REAL. (Confidence: {confidence:.2f}%)")
    print("="*40 + "\n")
    
    return prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the JPG/PNG image to test")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--openvino", action="store_true", help="Execute using OpenVINO for maximum Intel performance")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image '{args.image_path}' not found!")
        exit(1)
        
    device, has_ipex = get_intel_backend()
    model = load_image_model(args.ckpt, device, use_openvino=args.openvino)
    
    predict_single_image(args.image_path, model, device)