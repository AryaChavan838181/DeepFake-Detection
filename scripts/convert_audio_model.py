import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import onnx
import onnxruntime
from src.models.audio_models import AudioSpectrogramCNN

def export_audio_model_to_onnx():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "../src/models/weights/audio_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(base_dir, "../checkpoints/best_audio_model.pth")
            
        onnx_path = os.path.join(base_dir, "../android_app/app/src/main/assets/audio_model.onnx")
        
        # Initialize model
        model = AudioSpectrogramCNN()
        
        # Load weights
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Dummy input: Batch=1, Channel=1, Mel-Bins=128, Time-Steps=128
        dummy_input = torch.randn(1, 1, 128, 128)

        print("Exporting to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"Success! Model exported to {onnx_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed.")

    except Exception as e:
        print(f"Failed to export model: {e}")

if __name__ == "__main__":
    export_audio_model_to_onnx()
