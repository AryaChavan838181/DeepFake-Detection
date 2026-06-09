import os
import sys
import tempfile
import cv2
import torch
import librosa
import numpy as np
import argparse
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

try:
    # Try moviepy v2 API first
    from moviepy import VideoFileClip
except ImportError:
    try:
        # Fallback to moviepy v1 API
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("❌ ERROR: moviepy not installed. Please run: pip install moviepy")
        exit(1)

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.audio_models import get_audio_model

class DeepfakeEnsemblePredictor:
    def __init__(self, 
                 audio_ckpt="checkpoints/best_audio_model.pth", 
                 video_ov_xml="openvino_model/video_model.xml",
                 image_ov_xml="openvino_model/image_model.xml"):
        
        self.device = self._get_intel_backend()
        print("\n" + "="*50)
        print("🚀 INITIALIZING MULTI-MODAL ENSEMBLE PIPELINE")
        print("="*50)
        
        # 1. Initialize Face Tracking 
        print("👁️ Loading Face Tracker (MTCNN)...")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        
        # 2. Load PyTorch Audio Model (Phase 2C)
        print(f"🎵 Loading Audio Model (PyTorch XPU) from {audio_ckpt}...")
        self.audio_model = get_audio_model(device=self.device)
        if os.path.exists(audio_ckpt):
            self.audio_model.load_state_dict(torch.load(audio_ckpt, map_location=self.device))
        else:
            print(f"⚠️ warning: Audio checkpoint missing at {audio_ckpt}")
        self.audio_model.eval()
        
        # 3. Load OpenVINO Models
        try:
            import openvino as ov
            self.ov_core = ov.Core()
            
            # Select OpenVINO Device (prioritize NPU/GPU)
            ov_devices = self.ov_core.available_devices
            self.ov_target = 'CPU'
            for d in ['NPU', 'GPU']:
                if d in ov_devices:
                    self.ov_target = d
                    break
            print(f"⚙️ OpenVINO Execution Device: {self.ov_target}")
            
            # Load Video Spatial-Temporal Model (3D)
            print(f"🎥 Loading 3D Video OpenVINO Model from {video_ov_xml}...")
            if os.path.exists(video_ov_xml):
                v_model = self.ov_core.read_model(video_ov_xml)
                self.video_ov = self.ov_core.compile_model(v_model, self.ov_target)
            else:
                self.video_ov = None
                print(f"⚠️ warning: Video OpenVINO IR not found at {video_ov_xml}")
            
            # Load Image Spatial Model (2D)
            print(f"🖼️ Loading 2D Image OpenVINO Model from {image_ov_xml}...")
            if os.path.exists(image_ov_xml):
                i_model = self.ov_core.read_model(image_ov_xml)
                self.image_ov = self.ov_core.compile_model(i_model, self.ov_target)
            else:
                self.image_ov = None
                print(f"⚠️ warning: Image OpenVINO IR not found at {image_ov_xml}")
                
        except ImportError:
            print("❌ OpenVINO not installed! Cannot load video/image models.")
            self.video_ov = None
            self.image_ov = None
            
        # Transforms corresponding to respective models
        self.video_transform = transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.image_transform = transforms.Compose([
            transforms.Resize((226, 226)),
            transforms.ToTensor()
        ])

    def _get_intel_backend(self):
        device = torch.device('cpu')
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device('xpu')
        except:
            pass
        return device

    def extract_faces_from_video(self, video_path, num_frames=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return [], None
        
        step = max(1, total_frames // num_frames)
        face_frames = []
        best_face_image = None
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret: continue
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Scale down for faster MTCNN
            w, h = pil_img.size
            scale = min(1.0, 640.0 / max(w, h))
            if scale < 1.0:
                small_img = pil_img.resize((int(w * scale), int(h * scale)))
            else:
                small_img = pil_img
                
            boxes, _ = self.mtcnn.detect(small_img)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0].tolist()
                box = [b / scale for b in box] # rescale back
                
                w_box, h_box = box[2] - box[0], box[3] - box[1]
                x1, y1 = max(0, box[0] - 0.1 * w_box), max(0, box[1] - 0.1 * h_box)
                x2, y2 = min(w, box[2] + 0.1 * w_box), min(h, box[3] + 0.1 * h_box)
                
                face = pil_img.crop((x1, y1, x2, y2))
                face_frames.append(face)
                
                # Save the middle-ish frame as the static face for Image Model
                if best_face_image is None or i == num_frames // 2:
                    best_face_image = face

        cap.release()
        return face_frames, best_face_image
        
    def predict_audio(self, video_path):
        temp_wav = tempfile.mktemp(suffix=".wav")
        try:
            # Extract audio
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                return 1.0 # Default to real if no audio
            # Remove unsupported kwargs for moviepy v2
            clip.audio.write_audiofile(temp_wav, codec='pcm_s16le')
            clip.close()
            
            # Preprocess
            audio, sr = librosa.load(temp_wav, sr=16000)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            spec_min, spec_max = log_mel_spec.min(), log_mel_spec.max()
            normalized = (log_mel_spec - spec_min) / (spec_max - spec_min) if spec_max - spec_min > 0 else np.zeros_like(log_mel_spec)
            
            max_time_steps = 128
            if normalized.shape[1] < max_time_steps:
                normalized = np.pad(normalized, ((0, 0), (0, max_time_steps - normalized.shape[1])))
            else:
                start = (normalized.shape[1] - max_time_steps) // 2
                normalized = normalized[:, start:start + max_time_steps]
                
            tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.audio_model(tensor)
                prob = torch.sigmoid(output).item()
            return prob
        except Exception as e:
            print(f"⚠️ Audio extraction failed: {e}")
            return 1.0
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

    def predict_video(self, face_frames):
        if not self.video_ov or len(face_frames) == 0: return 1.0
        
        # We need exactly 8 frames for 3D Video
        seq = face_frames[:8]
        while len(seq) < 8:
            seq.append(seq[-1] if len(seq) > 0 else Image.new('RGB', (226, 226)))
            
        tensors = [self.video_transform(img) for img in seq]
        tensor_stack = torch.stack(tensors) # [8, 3, 226, 226]
        
        # Original model expects: [1, 8, 3, 226, 226] (Batch, Frames, Channels, Height, Width)
        tensor_stack = tensor_stack.unsqueeze(0) 
        
        ov_input = tensor_stack.numpy()
        ov_output = self.video_ov([ov_input])[0]
        return torch.sigmoid(torch.tensor(ov_output)).item()

    def predict_image(self, best_face):
        if not self.image_ov or best_face is None: return 1.0
        
        tensor_2d = self.image_transform(best_face).unsqueeze(0) # [1, 3, 226, 226]
        ov_input = tensor_2d.numpy()
        ov_output = self.image_ov([ov_input])[0]
        return torch.sigmoid(torch.tensor(ov_output)).item()

    def run_ensemble(self, video_path):
        print(f"\n🎥 Analysing Input Video: {video_path}")
        
        print("1️⃣ Decoding frames and tracking faces...")
        faces, best_face = self.extract_faces_from_video(video_path)
        if len(faces) == 0:
            print("❌ No faces found! Cannot proceed with ensemble.")
            return

        print("2️⃣ Running 3D Context Model (OpenVINO NPU)...")
        raw_prob_video = self.predict_video(faces)
        # ⚠️ CRITICAL FIX: The Video CNN-LSTM model expects FAKE=1 and REAL=0 based on its training loop.
        # We must invert it so that it aligns with the Image/Audio models where FAKE=0 and REAL=1.
        prob_video = 1.0 - raw_prob_video

        print("3️⃣ Running 2D Spatial Image Model (OpenVINO NPU)...")
        prob_image = self.predict_image(best_face)

        print("4️⃣ Extracting Audio and Running Spectrogram CNN (PyTorch XPU)...")
        prob_audio = self.predict_audio(video_path)

        # FUSION LOGIC (Fake = Close to 0.0, Real = Close to 1.0)
        w_video = 0.45
        w_image = 0.35
        w_audio = 0.20
        
        final_score = (prob_video * w_video) + (prob_image * w_image) + (prob_audio * w_audio)
        is_fake = final_score < 0.5
        
        print("\n" + "="*50)
        print(" 🧠 ENSEMBLE VERDICT ")
        print("="*50)
        print(f"🔸 3D Video Net Score : {prob_video:.4f}")
        print(f"🔸 2D Image Net Score : {prob_image:.4f}")
        print(f"🔸 2D Audio Net Score : {prob_audio:.4f}  (Weighted at {w_audio*100}%)")
        print("-" * 50)
        print(f"🔹 FINAL FUSED SCORE  : {final_score:.4f}  (Threshold < 0.5 is FAKE)")
        print("-" * 50)
        
        if is_fake:
            print(f"🚨🚨 CONCLUSION: FAKE / MANIPULATED VIDEO 🚨🚨")
            print(f"Confidence: {(1 - final_score)*100:.2f}%")
        else:
            print(f"✅✅ CONCLUSION: REAL / AUTHENTIC VIDEO ✅✅")
            print(f"Confidence: {final_score*100:.2f}%")
        print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the video to predict via Ensemble")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Target video '{args.video_path}' not found!")
        exit(1)
        
    predictor = DeepfakeEnsemblePredictor()
    predictor.run_ensemble(args.video_path)