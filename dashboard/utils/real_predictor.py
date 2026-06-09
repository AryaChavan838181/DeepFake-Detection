import os
import io
import cv2
import librosa
import librosa.display
import tempfile
import torch
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

# Ensure project root is in python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predict_ensemble import DeepfakeEnsemblePredictor

def generate_ela(img_path, quality=90):
    """ Error Level Analysis to detect manipulations based on jpeg compression ratios. """
    original = Image.open(img_path).convert('RGB')
    temp_filename = 'temp_ela.jpg'
    original.save(temp_filename, 'JPEG', quality=quality)
    temporary = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_filename)
    return ela_image

def generate_spectrogram(audio_path):
    """ Generate a real Mel-Spectrogram """
    y, sr = librosa.load(audio_path, sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-Spectrogram Spatial Artifacts')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_frequency_heatmap(image):
    """ Generate FFT-based frequency spectrum heatmap (reveals GAN artifacts) """
    # Convert PIL Image to OpenCV format
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    f = np.fft.fft2(cv_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(magnitude_spectrum, cmap='inferno')
    ax.set_title('High-Frequency Artifact Heatmap')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_temporal_plot(scores_per_frame):
    """ Generate temporal difference probabilities over time """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scores_per_frame, marker='o', linestyle='-', color='r' if np.mean(scores_per_frame) > 0.5 else 'g')
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Frame Index / Block')
    ax.set_ylabel('Deepfake Probability')
    ax.set_title('Video Temporal Consistency (3D-CNN+LSTM Profile)')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

class RealPredictor:
    def __init__(self):
        try:
            self.model = DeepfakeEnsemblePredictor()
            self.model_loaded = True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model_loaded = False
            
    def analyze_media(self, file_path, file_type):
        """ Run analysis: extract parameters, run model, build visual interpretations. """
        results = {
            'confidence': 0.0,
            'is_deepfake': False,
            'scores': {},
            'plots': [],
            'flag_reason': ''
        }
        
        is_video = file_type in ['mp4', 'avi', 'mov']
        is_audio = file_type in ['mp3', 'wav']
        is_image = file_type in ['jpg', 'jpeg', 'png']

        if self.model_loaded:
            if is_video:
                faces, best_face = self.model.extract_faces_from_video(file_path, num_frames=8)
                if best_face is not None:
                    # frequency diff for best face
                    fft_buf = generate_frequency_heatmap(best_face)
                    results['plots'].append({'title': 'Frequency Spectrum Analysis (Image Spatial)', 'stream': fft_buf})
                    
                    # temporal scores
                    scores = []
                    for face in faces:
                        p = self.model.predict_image(face)
                        scores.append(1.0 - p) # Deepfake Prob
                    
                    temporal_buf = generate_temporal_plot(scores)
                    results['plots'].append({'title': 'Temporal Confidence Over Time', 'stream': temporal_buf})
                    
                    # ELA
                    best_face.save("temp_face.jpg")
                    ela_img = generate_ela("temp_face.jpg")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(ela_img)
                    ax.set_title('Error Level Analysis (Compression Artifacts)')
                    ax.axis('off')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    plt.close(fig)
                    results['plots'].append({'title': 'Error Level Analysis', 'stream': buf})
                    
                # audio predictions
                try:
                    spec_buf = generate_spectrogram(file_path)
                    results['plots'].append({'title': 'Audio Mel-Spectrogram Extract', 'stream': spec_buf})
                except: pass
                
                # real predictor logic
                # For demonstration inside Streamlit, we mock the final values slightly if OpenVINO is missing
                # but we show the real plots we just calculated above.
                prob_video = 1.0 - self.model.predict_video(faces) if faces else 0.5
                prob_image = self.model.predict_image(best_face) if best_face else 0.5
                prob_audio = self.model.predict_audio(file_path)
                
                final_score = (prob_video * 0.45) + (prob_image * 0.35) + (prob_audio * 0.20)
                is_df = final_score < 0.5
                
                results['confidence'] = float(1 - final_score if is_df else final_score)
                results['is_deepfake'] = is_df
                results['scores']['Video 3D-CNN (Temporal Jitter)'] = 1 - prob_video
                results['scores']['Image 2D (Spatial Artifacts)'] = 1 - prob_image
                results['scores']['Audio (Spectrogram Frequency)'] = 1 - prob_audio
                
                reasons = []
                if prob_video < 0.5: reasons.append("high temporal inconsistency identified (frame-by-frame jitter)")
                if prob_image < 0.5: reasons.append("spatial artifacts/GAN frequency patterns detected in frames")
                if prob_audio < 0.5: reasons.append("vocal spectrogram features match synthetic generation")
                
                results['flag_reason'] = " and ".join(reasons) if reasons else "No significant artifacts detected."

            elif is_audio:
                prob_audio = self.model.predict_audio(file_path)
                results['confidence'] = float(1 - prob_audio if prob_audio < 0.5 else prob_audio)
                results['is_deepfake'] = prob_audio < 0.5
                results['scores']['Audio (Spectrogram Frequency)'] = 1 - prob_audio
                
                spec_buf = generate_spectrogram(file_path)
                results['plots'].append({'title': 'Audio Mel-Spectrogram Extract', 'stream': spec_buf})
                
                results['flag_reason'] = "synthetic vocal generation traits detected in spectrogram" if prob_audio < 0.5 else "No artificial acoustic signatures found."

            elif is_image:
                img = Image.open(file_path).convert('RGB')
                
                # ELA Plot
                ela_img = generate_ela(file_path)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(ela_img)
                ax.set_title('Error Level Analysis (Compression Artifacts)')
                ax.axis('off')
                ela_buf = io.BytesIO()
                plt.savefig(ela_buf, format='png', bbox_inches='tight', dpi=150)
                ela_buf.seek(0)
                plt.close(fig)
                results['plots'].append({'title': 'Error Level Analysis', 'stream': ela_buf})
                
                # FFT Plot
                fft_buf = generate_frequency_heatmap(img)
                results['plots'].append({'title': 'Frequency Spectrum Analysis (Image Spatial)', 'stream': fft_buf})
                
                # We need mtcnn to grab face
                try:
                    boxes, _ = self.model.mtcnn.detect(img)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0]
                        img = img.crop((box[0], box[1], box[2], box[3]))
                except:
                    pass
                
                prob_image = self.model.predict_image(img)
                results['confidence'] = float(1 - prob_image if prob_image < 0.5 else prob_image)
                results['is_deepfake'] = prob_image < 0.5
                results['scores']['Image 2D (Spatial Artifacts)'] = 1 - prob_image
                
                results['flag_reason'] = "GAN-based synthesis traces identified in spatial frequency arrays or pixel boundaries" if prob_image < 0.5 else "Clean pixel structures."
            
        return results

