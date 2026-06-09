import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DataExplorer:
    @staticmethod
    def visualize_image(image, title="Image"):
        #visualize a single image
        plt.figure(figsize=(6,6))

        #denormalize for visualization
        img_display = (image*0.229 + 0.485)*255
        img_display = np.clip(img_display, 0, 255).astype(np.uint8)
        plt.imshow(img_display)
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def visualize_spectogram(spectogram, title="Spectogram"):
        #visualize audio spect
        plt.figure(figsize=(12,4))
        plt.imshow(spectogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='dB')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Frequency Bins')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_batch_stats(batch):
        #get stats about a batch of data
        if isinstance(batch, np.ndarray):
            return{
                'shape':batch.shape,
                'dtype': str(batch.dtype),
                'min': batch.min(),
                'max': batch.max(),
                'mean': batch.mean(),
                'std':batch.std()
            }
        return None