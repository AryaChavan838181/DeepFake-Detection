import sys
sys.path.append('.')

import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048):
        #sample_rate: Sample rate of audio, n_mels: Number of mel frequency bins, n_fft: FFT window size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft

    def get_mel_spectogram(self, audio, sr):
        #generate mel-scale spectogram from audio

        #convert audio to mel-scale spectogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )

        #convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def normalize_spectogram(self, spectogram):
        spec_min = spectogram.min()
        spec_max = spectogram.max()

        if spec_max - spec_min == 0:
            return np.zeros_like(spectogram)
        
        normalized = (spectogram-spec_min)/(spec_max-spec_min)
        return normalized
    
    def preprocess_audio(self, audio, sr):
        mel_spec = self.get_mel_spectogram(audio, sr)
        normalized = self.normalize_spectogram(mel_spec)
        return normalized
    
# if __name__ == "__main__":
#     from src.data.audio_loader import AudioDataLoader
    
#     print("=" * 50)
#     print("AUDIO PREPROCESSING TEST")
#     print("=" * 50)
    
#     loader = AudioDataLoader(sample_rate=16000)
    
#     # Load audio from video
#     print("\n📻 Loading audio from video...")
#     audio, sr = loader.load_from_video("C:\\Users\\alesh\\OneDrive\\Documents\\coding\\college\\edi sem4\\sem4_edi\\test_video.mp4")
    
#     if audio is None:
#         print("❌ Failed to load audio")
#     else:
#         print(f"✅ Loaded audio")
#         print(f"   Sample rate: {sr} Hz")
#         print(f"   Duration: {len(audio) / sr:.2f} seconds")
        
#         # Preprocess
#         print("\n🔄 Generating spectrogram...")
#         preprocessor = AudioPreprocessor(sample_rate=sr)
#         spectrogram = preprocessor.preprocess_audio(audio, sr)
        
#         print(f"✅ Spectrogram generated")
#         print(f"   Shape: {spectrogram.shape}")
#         print(f"   Frequency bins: {spectrogram.shape[0]}")
#         print(f"   Time steps: {spectrogram.shape[1]}")
#         print(f"   Value range: [{spectrogram.min():.2f}, {spectrogram.max():.2f}]")
        
#         # Visualize
#         print("\n📊 Generating visualization...")
#         plt.figure(figsize=(14, 6))
#         plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
#         plt.colorbar(label='Normalized Amplitude (0-1)')
#         plt.title('Mel-Scale Spectrogram')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Frequency Bins (Mel Scale)')
#         plt.tight_layout()
#         plt.savefig('spectrogram_visualization.png')
#         print("✅ Saved visualization to 'spectrogram_visualization.png'")
#         plt.show()
#         print("📈 A window should pop up with the spectrogram visualization")