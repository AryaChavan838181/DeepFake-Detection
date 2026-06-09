import os
import json
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

def get_intel_device():
    # 1. Try IPEX for XPU (Intel GPU/Integrated GPU)
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return torch.device('xpu')
    except Exception as e:
        print(f"Skipping IPEX: {e}")
        
    # 2. Try Intel NPU Acceleration Library
    try:
        import intel_npu_acceleration_library
        # Usually NPU library handles the compilation graph, 
        # but if device mapping is exposed, we catch it.
    except Exception as e:
        print(f"Skipping NPU: {e}")
        
    return torch.device('cpu')

def extract_faces_from_videos_pytorch(metadata_path, output_dir, max_videos_per_class=10, frames_per_video=3):
    """
    Extracts face frames from the provided DFDC chunk using PyTorch and accelerates it via Intel XPU/NPU.
    """
    device = get_intel_device()
    print(f"✅ PyTorch Preprocessing Pipeline initialized. Using device: {device}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    video_dir = os.path.dirname(metadata_path)
    os.makedirs(os.path.join(output_dir, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "FAKE"), exist_ok=True)
    
    # Load MTCNN in PyTorch onto the Intel IPEX target device
    mtcnn = MTCNN(keep_all=False, device=device)
    
    counts = {"REAL": 0, "FAKE": 0}
    
    for filename, data in tqdm(metadata.items(), desc="Processing Videos (PyTorch/Intel)"):
        label = data["label"]
        if counts[label] >= max_videos_per_class:
            continue
            
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300
            
        frame_interval = max(1, total_frames // frames_per_video)
        frames_saved = 0
        
        for i in range(frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # MTCNN from facenet-pytorch returning bounding boxes
                boxes, probs = mtcnn.detect(frame_rgb)
                
                if boxes is not None:
                    x1, y1, x2, y2 = boxes[0]
                    y1, y2 = max(0, int(y1)), max(0, int(y2))
                    x1, x2 = max(0, int(x1)), max(0, int(x2))
                    
                    face = frame[y1:y2, x1:x2]
                    
                    if face.size > 0:
                        face_resized = cv2.resize(face, (226, 226))
                        out_path = os.path.join(output_dir, label, f"{filename.split('.')[0]}_frame{i}.jpg")
                        cv2.imwrite(out_path, face_resized)
                        frames_saved += 1
            except Exception as e:
                pass
        
        if frames_saved > 0:
            counts[label] += 1
            
        cap.release()
        
        if counts["REAL"] >= max_videos_per_class and counts["FAKE"] >= max_videos_per_class:
            break

    print(f"Extraction complete! Saved {counts['REAL']} REAL videos and {counts['FAKE']} FAKE videos.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from DFDC dataset.")
    parser.add_argument("--metadata", type=str, default=r"C:\Users\ritik\Desktop\Projects\sem4_edi\dfdc_train_part_00\dfdc_train_part_0\metadata.json")
    parser.add_argument("--output", type=str, default="data_prepared_pt")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos per class (None for all)")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to extract per video")
    
    args = parser.parse_args()
    
    max_vids = args.max_videos if args.max_videos is not None else float('inf')
    print(f"Starting Intel PyTorch face extraction (Max videos per class: {'All' if args.max_videos is None else args.max_videos})...")
    extract_faces_from_videos_pytorch(args.metadata, args.output, max_videos_per_class=max_vids, frames_per_video=args.frames)
