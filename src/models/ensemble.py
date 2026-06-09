import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from src.models.image_models import EfficientNetDetector, ResNetFrequencyDetector, ViTDetector
from src.models.video_models import VideoCNNLSTM
from src.models.audio_models import AudioSpectrogramCNN


# LATE FUSION ENSEMBLE (Independent Modality Predictions → Weighted Average)
class EnsembleDetectorLateFusion(keras.Model):
    """
    Late fusion ensemble combining predictions from image, video, and audio modalities.
    
    Each modality is trained independently, their predictions are concatenated,
    and a final dense layer learns optimal weighted fusion.
    
    Inputs:
        image: (batch_size, 226, 226, 3)
        video: (batch_size, num_frames, 226, 226, 3)
        audio: (batch_size, 128, 128, 1)
    
    Output: (batch_size, 1) - ensemble deepfake probability
    """
    
    def __init__(
        self, 
        image_model='efficientnet',
        video_freeze_early=True,
        audio_freeze_early=True
    ):
        super(EnsembleDetectorLateFusion, self).__init__()
        
        # Image modality
        if image_model == 'efficientnet':
            self.image_model = EfficientNetDetector()
        elif image_model == 'resnet_freq':
            self.image_model = ResNetFrequencyDetector()
        elif image_model == 'vit':
            self.image_model = ViTDetector()
        else:
            raise ValueError(f"image_model must be 'efficientnet', 'resnet_freq', or 'vit'")
        
        # Video modality
        self.video_model = VideoCNNLSTM()
        
        # Audio modality
        self.audio_model = AudioSpectrogramCNN()
        
        # Optionally freeze early layers for faster training
        if video_freeze_early:
            self.video_model.unfreeze_conv(layer_count=1)
        if audio_freeze_early:
            self.audio_model.unfreeze_conv(layer_count=1)
        
        # Fusion layers: concatenate 3 predictions → learn weights
        self.fusion_dense1 = layers.Dense(64, activation='relu')
        self.fusion_dropout1 = layers.Dropout(0.3)
        self.fusion_dense2 = layers.Dense(32, activation='relu')
        self.fusion_dropout2 = layers.Dropout(0.2)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: Dict with keys 'image', 'video', 'audio'
                - image: (batch_size, 226, 226, 3)
                - video: (batch_size, num_frames, 226, 226, 3)
                - audio: (batch_size, 128, 128, 1)
            training: Boolean flag for dropout/BN
        
        Returns:
            output: (batch_size, 1) - ensemble deepfake probability
        """
        # Get per-modality predictions
        image_pred = self.image_model(inputs['image'], training=training)
        video_pred = self.video_model(inputs['video'], training=training)
        audio_pred = self.audio_model(inputs['audio'], training=training)
        
        # Concatenate predictions
        fused = tf.concat([image_pred, video_pred, audio_pred], axis=1)
        
        # Learn optimal fusion weights
        fused = self.fusion_dense1(fused)
        fused = self.fusion_dropout1(fused, training=training)
        fused = self.fusion_dense2(fused)
        fused = self.fusion_dropout2(fused, training=training)
        output = self.output_layer(fused)
        
        return output
    
    def unfreeze_modalities(self, modality_layers: dict):
        """
        Selectively unfreeze specific modalities for fine-tuning.
        
        Args:
            modality_layers: Dict with keys 'image', 'video', 'audio'
                Each value is number of layers to unfreeze (or None to skip)
        
        Example:
            ensemble.unfreeze_modalities({
                'image': 20,
                'video': 2,
                'audio': 2
            })
        """
        if modality_layers.get('image'):
            if hasattr(self.image_model, 'unfreeze_base'):
                self.image_model.unfreeze_base(layer_count=modality_layers['image'])
            elif hasattr(self.image_model, 'unfreeze_bias'):
                self.image_model.unfreeze_bias(layer_count=modality_layers['image'])
        
        if modality_layers.get('video'):
            self.video_model.unfreeze_conv(layer_count=modality_layers['video'])
        
        if modality_layers.get('audio'):
            self.audio_model.unfreeze_conv(layer_count=modality_layers['audio'])


# ATTENTION FUSION ENSEMBLE (Learned Fusion Weights per Modality)
class EnsembleDetectorAttentionFusion(keras.Model):
    """
    Attention-based ensemble with learned modality importance weights.
    
    Instead of simple averaging, learns a attention mechanism to weight
    which modality is more important for each sample.
    
    Useful when one modality is noisier in certain conditions.
    """
    
    def __init__(
        self, 
        image_model='efficientnet',
        video_freeze_early=True,
        audio_freeze_early=True
    ):
        super(EnsembleDetectorAttentionFusion, self).__init__()
        
        # Image modality
        if image_model == 'efficientnet':
            self.image_model = EfficientNetDetector()
        elif image_model == 'resnet_freq':
            self.image_model = ResNetFrequencyDetector()
        elif image_model == 'vit':
            self.image_model = ViTDetector()
        else:
            raise ValueError(f"image_model must be 'efficientnet', 'resnet_freq', or 'vit'")
        
        # Video modality
        self.video_model = VideoCNNLSTM()
        
        # Audio modality
        self.audio_model = AudioSpectrogramCNN()
        
        # Optionally freeze early layers
        if video_freeze_early:
            self.video_model.unfreeze_conv(layer_count=1)
        if audio_freeze_early:
            self.audio_model.unfreeze_conv(layer_count=1)
        
        # Attention weights for each modality
        self.attention_layer = layers.Dense(3, activation='softmax')  # 3 modalities
        
        # Fusion classification layers
        self.fusion_dense1 = layers.Dense(64, activation='relu')
        self.fusion_dropout1 = layers.Dropout(0.3)
        self.fusion_dense2 = layers.Dense(32, activation='relu')
        self.fusion_dropout2 = layers.Dropout(0.2)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: Dict with keys 'image', 'video', 'audio'
            training: Boolean flag for dropout/BN
        
        Returns:
            output: (batch_size, 1) - ensemble deepfake probability
        """
        # Get per-modality predictions
        image_pred = self.image_model(inputs['image'], training=training)
        video_pred = self.video_model(inputs['video'], training=training)
        audio_pred = self.audio_model(inputs['audio'], training=training)
        
        # Concatenate predictions
        combined = tf.concat([image_pred, video_pred, audio_pred], axis=1)
        
        # Learn modality-specific attention weights
        attention_weights = self.attention_layer(combined)  # (batch_size, 3)
        
        # Apply attention weights to each modality prediction
        weighted_image = image_pred * attention_weights[:, 0:1]
        weighted_video = video_pred * attention_weights[:, 1:2]
        weighted_audio = audio_pred * attention_weights[:, 2:3]
        
        # Sum weighted predictions
        fused = weighted_image + weighted_video + weighted_audio
        
        # Classification head
        fused = self.fusion_dense1(fused)
        fused = self.fusion_dropout1(fused, training=training)
        fused = self.fusion_dense2(fused)
        fused = self.fusion_dropout2(fused, training=training)
        output = self.output_layer(fused)
        
        return output
    
    def unfreeze_modalities(self, modality_layers: dict):
        """
        Selectively unfreeze specific modalities for fine-tuning.
        """
        if modality_layers.get('image'):
            if hasattr(self.image_model, 'unfreeze_base'):
                self.image_model.unfreeze_base(layer_count=modality_layers['image'])
            elif hasattr(self.image_model, 'unfreeze_bias'):
                self.image_model.unfreeze_bias(layer_count=modality_layers['image'])
        
        if modality_layers.get('video'):
            self.video_model.unfreeze_conv(layer_count=modality_layers['video'])
        
        if modality_layers.get('audio'):
            self.audio_model.unfreeze_conv(layer_count=modality_layers['audio'])


# SIMPLE WEIGHTED ENSEMBLE (Fixed Weights)
class EnsembleDetectorWeighted(keras.Model):
    """
    Simple weighted average ensemble with fixed fusion weights.
    
    Useful for quick baseline when you know modality importance ratios.
    
    Example weights:
        - Image heavy: [0.5, 0.3, 0.2]
        - Balanced: [0.33, 0.33, 0.34]
        - Audio heavy: [0.25, 0.25, 0.5]
    """
    
    def __init__(
        self, 
        image_model='efficientnet',
        fusion_weights=(0.33, 0.33, 0.34)
    ):
        super(EnsembleDetectorWeighted, self).__init__()
        
        # Image modality
        if image_model == 'efficientnet':
            self.image_model = EfficientNetDetector()
        elif image_model == 'resnet_freq':
            self.image_model = ResNetFrequencyDetector()
        elif image_model == 'vit':
            self.image_model = ViTDetector()
        else:
            raise ValueError(f"image_model must be 'efficientnet', 'resnet_freq', or 'vit'")
        
        # Video modality
        self.video_model = VideoCNNLSTM()
        
        # Audio modality
        self.audio_model = AudioSpectrogramCNN()
        
        # Store fusion weights
        self.fusion_weights = fusion_weights
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: Dict with keys 'image', 'video', 'audio'
            training: Boolean flag for dropout/BN
        
        Returns:
            output: (batch_size, 1) - weighted ensemble prediction
        """
        # Get per-modality predictions
        image_pred = self.image_model(inputs['image'], training=training)
        video_pred = self.video_model(inputs['video'], training=training)
        audio_pred = self.audio_model(inputs['audio'], training=training)
        
        # Apply fixed fusion weights
        output = (
            self.fusion_weights[0] * image_pred +
            self.fusion_weights[1] * video_pred +
            self.fusion_weights[2] * audio_pred
        )
        
        return output


# UTILITY FUNCTIONS
def load_ensemble_model(
    fusion_type: str = 'late_fusion',
    image_model: str = 'efficientnet',
    weights_path: str = None
) -> keras.Model:
    """
    Load ensemble detection model.
    
    Args:
        fusion_type: 'late_fusion', 'attention_fusion', or 'weighted'
        image_model: 'efficientnet', 'resnet_freq', or 'vit'
        weights_path: Path to pre-trained weights (.h5 file)
    
    Returns:
        Initialized Keras model
    """
    models = {
        'late_fusion': EnsembleDetectorLateFusion,
        'attention_fusion': EnsembleDetectorAttentionFusion,
        'weighted': EnsembleDetectorWeighted,
    }
    
    if fusion_type not in models:
        raise ValueError(f"fusion_type must be one of {list(models.keys())}")
    
    model = models[fusion_type](image_model=image_model)
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model


if __name__ == "__main__":
    print("Testing ensemble models...")
    
    # Create dummy inputs
    dummy_image = np.random.randn(2, 226, 226, 3).astype(np.float32) / 255.0
    dummy_video = np.random.randn(2, 16, 226, 226, 3).astype(np.float32) / 255.0
    dummy_audio = np.random.randn(2, 128, 128, 1).astype(np.float32)
    
    inputs = {
        'image': dummy_image,
        'video': dummy_video,
        'audio': dummy_audio
    }
    
    # Test Late Fusion
    print("\n1. Ensemble - Late Fusion:")
    model1 = EnsembleDetectorLateFusion(image_model='efficientnet')
    output1 = model1(inputs)
    print(f"   ✅ Output shape: {output1.shape}")
    print(f"   ✅ Trainable params: {model1.count_params():,}")
    
    # Test Attention Fusion
    print("\n2. Ensemble - Attention Fusion:")
    model2 = EnsembleDetectorAttentionFusion(image_model='efficientnet')
    output2 = model2(inputs)
    print(f"   ✅ Output shape: {output2.shape}")
    print(f"   ✅ Trainable params: {model2.count_params():,}")
    
    # Test Weighted Ensemble
    print("\n3. Ensemble - Weighted (Fixed Weights):")
    model3 = EnsembleDetectorWeighted(image_model='efficientnet')
    output3 = model3(inputs)
    print(f"   ✅ Output shape: {output3.shape}")
    print(f"   ✅ Trainable params: {model3.count_params():,}")
    
    print("\n✅ All ensemble models initialized successfully!")
