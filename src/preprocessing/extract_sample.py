import os
import json
import cv2
import glob
from pathlib import Path
from mtcnn import MTCNN
from tqdm import tqdm

def extract_faces_from_videos(metadata_path, output_dir, max_videos_per_class=20, frames_per_video=3):
    """
    Extracts face frames from the provided DFDC chunk.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    video_dir = os.path.dirname(metadata_path)
    os.makedirs(os.path.join(output_dir, "REAL"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "FAKE"), exist_ok=True)
    
    detector = MTCNN()
    
    counts = {"REAL": 0, "FAKE": 0}
    
    # Iterate over items in metadata
    for filename, data in tqdm(metadata.items(), desc="Processing Videos"):
        label = data["label"]
        if counts[label] >= max_videos_per_class:
            continue
            
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
            
        # Get total frames to sample uniformly
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300 # Fallback
            
        frame_interval = max(1, total_frames // frames_per_video)
        frames_saved = 0
        
        for i in range(frames_per_video):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            
            if faces:
                # Use the first face found
                x, y, w, h = faces[0]['box']
                # Add some margin
                y = max(0, y)
                x = max(0, x)
                face = frame[y:y+h, x:x+w]
                
                # Resize
                try:
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
    metadata = r"C:\Users\ritik\Desktop\Projects\sem4_edi\dfdc_train_part_00\dfdc_train_part_0\metadata.json"
    output = "data_prepared"
    print("Starting face extraction...")
    extract_faces_from_videos(metadata, output, max_videos_per_class=50, frames_per_video=3)
