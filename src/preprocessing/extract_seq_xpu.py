import os
import cv2
import json
import torch
from glob import glob
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision import transforms
import argparse

def get_intel_device():
    try:
        import intel_extension_for_pytorch as ipex
        if torch.xpu.is_available():
            return torch.device('xpu')
    except Exception:
        pass
    return torch.device('cpu')

def find_metadata_files(base_dirs):
    paths = []
    for d in base_dirs:
        for root, dirs, files in os.walk(d):
            if 'metadata.json' in files:
                paths.append(os.path.join(root, 'metadata.json'))
    return paths

def extract_video_sequences(metadata_files, output_dir, max_videos=None, frames_per_video=15):
    """
    Extracts sequences of faces instead of random frames to support the CNN-LSTM video architecture!
    """
    device = get_intel_device()
    print(f"✅ Fast Video Sequence Extractor initialized. Using device: {device}")
    
    os.makedirs(os.path.join(output_dir, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "FAKE"), exist_ok=True)

    mtcnn = MTCNN(keep_all=False, device=device)
    transform = transforms.Compose([
        transforms.Resize((226, 226)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    video_entries = []
    for m in metadata_files:
        with open(m, 'r') as f:
            data = json.load(f)
            folder_path = os.path.dirname(m)
            for file_name, info in data.items():
                video_entries.append({
                    'path': os.path.join(folder_path, file_name),
                    'label': info['label'],
                    'name': file_name
                })
                
    if max_videos:
        video_entries = video_entries[:max_videos]

    print(f"Found {len(video_entries)} candidate videos across dataset partitions.")
    
    saved_reals = 0
    saved_fakes = 0

    import sys
    import io
    
    # Need to disable stdout sometimes to ignore cv2 spam
    devnull = io.TextIOWrapper(open(os.devnull, 'wb'), encoding='utf-8')
    
    for entry in tqdm(video_entries, desc="Processing Video Sequences"):
        if not os.path.exists(entry['path']):
            continue

        label_folder = entry['label']
        save_path = os.path.join(output_dir, label_folder, f"{entry['name']}.pt")
        
        # Skip if already extracted
        if os.path.exists(save_path):
            if entry['label'] == 'REAL':
                saved_reals += 1
            else:
                saved_fakes += 1
            continue

        # Suppress MTCNN internal print spam
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            v_cap = cv2.VideoCapture(entry['path'])
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if v_len < 2:
                raise Exception("Corrupt Video")
                
            step = max(1, v_len // frames_per_video)
            
            frames_to_process = []
            curr_frame = 0
            
            # ⚡ FAST READ: Use grab() to completely skip decoding useless frames natively in C++
            while len(frames_to_process) < frames_per_video:
                success = v_cap.grab()
                if not success: 
                    break
                
                if curr_frame % step == 0:
                    success, frame = v_cap.retrieve() # Only fully render the frame if we need it
                    if success:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames_to_process.append(Image.fromarray(frame_rgb))
                
                curr_frame += 1
                
            v_cap.release()
            
            # ⚡ FAST XPU BATCHING: Send all 15 frames to Intel XPU at the exactly same time!
            if len(frames_to_process) > 0:
                boxes_batch, _ = mtcnn.detect(frames_to_process)
            else:
                boxes_batch = None
                
            face_tensors = []
            if boxes_batch is not None:
                # boxes_batch returns a list of detections for each frame
                for i, boxes in enumerate(boxes_batch):
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0].tolist()
                        face = frames_to_process[i].crop((box[0], box[1], box[2], box[3]))
                        face_tensors.append(transform(face))
            
            # Pad the tensor if the detector missed certain frames but found at least 3 valid faces
            if len(face_tensors) >= 3:
                while len(face_tensors) < frames_per_video:
                    face_tensors.append(face_tensors[-1]) # Repeat the last known face backwards
                
                final_tensor = torch.stack(face_tensors) # Shape: [15, 3, 226, 226]
                torch.save(final_tensor, save_path)      # DIRECTLY save the mapped tensor so training is immediate!
                
                if entry['label'] == 'REAL':
                    saved_reals += 1
                else:
                    saved_fakes += 1

        except Exception as e:
            pass
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    print(f"\n✅ Sequential Extraction Complete! Saved: {saved_reals} REAL Tensors, {saved_fakes} FAKE Tensors")
    print("These `.pt` files contain the temporal sequence memory and load 10x faster than JPEGs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs='+', default=["dfdc_train_part_00", "dfdc_train_part_01"], help="List of dataset folder paths")
    parser.add_argument("--out", type=str, default="data_seq_pt", help="Where to output the sequence tensors")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of total videos to process")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames per sequence track")
    args = parser.parse_args()
    
    meta_files = find_metadata_files(args.dirs)
    if not meta_files:
        print("❌ Could not find 'metadata.json' in the provided directories!")
    else:
        extract_video_sequences(meta_files, args.out, max_videos=args.max, frames_per_video=args.frames)
