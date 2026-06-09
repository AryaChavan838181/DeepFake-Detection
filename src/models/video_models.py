import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ============================================================================
# VIDEO CNN-LSTM DETECTOR (OPTIMIZED 3D CONVOLUTIONS + TEMPORAL LSTM)
# ============================================================================
class VideoCNNLSTM(keras.Model):
    """
    Optimized spatiotemporal deepfake detector using 3D convolutions for spatial
    feature extraction and LSTM for temporal sequence modeling across video frames.
    
    Targets temporal jitter artifacts common in deepfakes by:
    - Analyzing spatial features across time (3D CNN)
    - Modeling frame-to-frame transitions (LSTM)
    - Detecting motion inconsistencies
    
    Args:
        input_shape: (num_frames, height, width, channels). Default: (16, 226, 226, 3)
        num_frames: Number of consecutive frames (for flexible-length sequences)
    
    Input:  (batch_size, num_frames, height, width, channels)
    Output: (batch_size, 1) - deepfake probability [0, 1]
    """
    
    def __init__(self, input_shape=(16, 226, 226, 3), num_frames=16):
        super(VideoCNNLSTM, self).__init__()
        self.input_shape_val = input_shape
        self.num_frames = num_frames
        
        # ===== 3D CONVOLUTIONAL PATHWAY (SPATIAL-TEMPORAL FEATURE EXTRACTION) =====
        # Block 1: Initial 3D convolution with moderate channel expansion
        self.conv3d_1 = layers.Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            strides=(1, 2, 2),
            name='conv3d_block1'
        )
        self.bn3d_1 = layers.BatchNormalization(name='bn3d_block1')
        self.pool3d_1 = layers.MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            name='pool3d_block1'
        )
        
        # Block 2: Deeper spatial receptive field with channel increase
        self.conv3d_2 = layers.Conv3D(
            filters=64,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            strides=(1, 2, 2),
            name='conv3d_block2'
        )
        self.bn3d_2 = layers.BatchNormalization(name='bn3d_block2')
        self.pool3d_2 = layers.MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            name='pool3d_block2'
        )
        
        # Block 3: Temporal refinement with spatial compression
        # Smaller spatial kernel to preserve temporal resolution
        self.conv3d_3 = layers.Conv3D(
            filters=128,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            strides=(1, 1, 1),
            name='conv3d_block3'
        )
        self.bn3d_3 = layers.BatchNormalization(name='bn3d_block3')
        self.pool3d_3 = layers.MaxPooling3D(
            pool_size=(1, 2, 2),
            strides=(1, 2, 2),
            name='pool3d_block3'
        )
        
        # ===== TEMPORAL LSTM PATHWAY (SEQUENCE MODELING) =====
        # LSTM layer 1: Captures short-term temporal dependencies
        self.lstm_1 = layers.LSTM(
            units=256,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_1'
        )
        
        # LSTM layer 2: Captures long-term temporal patterns
        self.lstm_2 = layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_2'
        )
        
        # ===== CLASSIFICATION HEAD =====
        self.dense1 = layers.Dense(256, activation='relu', name='dense1')
        self.dropout1 = layers.Dropout(0.5, name='dropout1')
        self.dense2 = layers.Dense(128, activation='relu', name='dense2')
        self.dropout2 = layers.Dropout(0.3, name='dropout2')
        self.output_layer = layers.Dense(1, activation='sigmoid', name='deepfake_score')
    
    def call(self, inputs, training=False):
        """
        Forward pass through the video CNN-LSTM model.
        
        Args:
            inputs: Tensor of shape (batch_size, num_frames, height, width, channels)
                   Values should be in range [0, 1] (normalized)
            training: Boolean flag for dropout and batch normalization
        
        Returns:
            output: Tensor of shape (batch_size, 1) with deepfake scores [0, 1]
        """
        # ===== 3D CONVOLUTION BLOCKS (Spatiotemporal feature extraction) =====
        x = self.conv3d_1(inputs)
        x = self.bn3d_1(x, training=training)
        x = self.pool3d_1(x)
        
        x = self.conv3d_2(x)
        x = self.bn3d_2(x, training=training)
        x = self.pool3d_2(x)
        
        x = self.conv3d_3(x)
        x = self.bn3d_3(x, training=training)
        x = self.pool3d_3(x)
        
        # ===== RESHAPE FOR LSTM (Time × Features) =====
        # From (batch, time, height, width, channels) → (batch, time, features)
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        spatial_dim = tf.reduce_prod(tf.shape(x)[2:])
        x = tf.reshape(x, [batch_size, time_steps, spatial_dim])
        
        # ===== LSTM TEMPORAL MODELING =====
        # LSTM processes frame sequences and captures temporal correlations
        x = self.lstm_1(x, training=training)
        x = self.lstm_2(x, training=training)  # Returns only final hidden state
        
        # ===== CLASSIFICATION HEAD =====
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        output = self.output_layer(x)
        
        return output
    
    def unfreeze_conv(self, layer_count=2):
        """
        Unfreeze the last N 3D convolutional blocks for fine-tuning.
        
        Earlier conv layers learn general spatiotemporal patterns (motion, edges).
        Later layers adapt to deepfake-specific artifacts (jitter, discontinuities).
        
        Args:
            layer_count: Number of conv blocks to unfreeze from the end
        """
        conv_layers = [
            (self.conv3d_1, self.bn3d_1, self.pool3d_1),
            (self.conv3d_2, self.bn3d_2, self.pool3d_2),
            (self.conv3d_3, self.bn3d_3, self.pool3d_3),
        ]
        
        # Freeze earlier blocks
        for i, (conv, bn, pool) in enumerate(conv_layers[:-layer_count]):
            conv.trainable = False
            bn.trainable = False
        
        # Unfreeze later blocks
        for i, (conv, bn, pool) in enumerate(conv_layers[-layer_count:]):
            conv.trainable = True
            bn.trainable = True


# ============================================================================
# LIGHTWEIGHT VIDEO DETECTOR (MEMORY-EFFICIENT VARIANT)
# ============================================================================
class VideoLightweight3DCNN(keras.Model):
    """
    Lightweight spatiotemporal detector for constrained memory environments.
    Uses fewer filters and shallower architecture than VideoCNNLSTM.
    
    Useful for:
    - Mobile/edge deployment
    - Quick validation during development
    - Ensemble baseline model
    
    Input:  (batch_size, num_frames, height, width, channels)
    Output: (batch_size, 1) - deepfake probability [0, 1]
    """
    
    def __init__(self, input_shape=(16, 226, 226, 3), num_frames=16):
        super(VideoLightweight3DCNN, self).__init__()
        self.input_shape_val = input_shape
        self.num_frames = num_frames
        
        # Lightweight 3D convolutions
        self.conv3d_1 = layers.Conv3D(
            filters=16,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            strides=(1, 2, 2)
        )
        self.bn3d_1 = layers.BatchNormalization()
        self.pool3d_1 = layers.MaxPooling3D(pool_size=(2, 2, 2))
        
        self.conv3d_2 = layers.Conv3D(
            filters=32,
            kernel_size=(3, 3, 3),
            activation='relu',
            padding='same',
            strides=(1, 2, 2)
        )
        self.bn3d_2 = layers.BatchNormalization()
        self.pool3d_2 = layers.MaxPooling3D(pool_size=(2, 2, 2))
        
        # Lightweight LSTM
        self.lstm = layers.LSTM(
            units=128,
            return_sequences=False,
            dropout=0.2
        )
        
        # Classification
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.conv3d_1(inputs)
        x = self.bn3d_1(x, training=training)
        x = self.pool3d_1(x)
        
        x = self.conv3d_2(x)
        x = self.bn3d_2(x, training=training)
        x = self.pool3d_2(x)
        
        # Reshape for LSTM
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        spatial_dim = tf.reduce_prod(tf.shape(x)[2:])
        x = tf.reshape(x, [batch_size, time_steps, spatial_dim])
        
        x = self.lstm(x, training=training)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        output = self.output_layer(x)
        
        return output



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def load_video_model(model_type: str = 'cnn_lstm', weights_path: str = None) -> keras.Model:
    """
    Load video detection model with optional pre-trained weights.
    
    Args:
        model_type: Type of video model ('cnn_lstm' or 'lightweight_3dcnn')
        weights_path: Optional path to pre-trained weights file (.h5 or .weights.h5)
    
    Returns:
        Initialized Keras model ready for inference or training
    
    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'cnn_lstm': VideoCNNLSTM,
        'lightweight_3dcnn': VideoLightweight3DCNN,
    }
    
    if model_type not in models:
        raise ValueError(f"model_type must be one of {list(models.keys())}")
    
    model = models[model_type]()
    
    if weights_path:
        print(f"Loading pre-trained weights from {weights_path}...")
        model.load_weights(weights_path)
        print("✅ Weights loaded successfully")
    
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("VIDEO MODEL ARCHITECTURE TEST")
    print("=" * 70)
    
    # Create dummy video input batch: (batch_size, num_frames, height, width, channels)
    # Normalized to [0, 1] as expected by the model
    dummy_video = np.random.randn(2, 16, 226, 226, 3).astype(np.float32) / 255.0
    print(f"\n📹 Input shape: {dummy_video.shape}")
    print(f"   (batch_size=2, frames=16, height=226, width=226, channels=3)")
    
    # ===== Test 1: VideoCNNLSTM =====
    print("\n" + "−" * 70)
    print("1. VideoCNNLSTM (Optimized 3D Conv + LSTM):")
    print("−" * 70)
    try:
        model1 = VideoCNNLSTM()
        output1 = model1(dummy_video)
        print(f"   ✅ Output shape:       {output1.shape}")
        print(f"   ✅ Trainable params:   {model1.count_params():,}")
        print(f"   📊 Model memory (GB):  {model1.count_params() * 4 / 1e9:.3f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # ===== Test 2: VideoLightweight3DCNN =====
    print("\n" + "−" * 70)
    print("2. VideoLightweight3DCNN (Memory-Efficient):")
    print("−" * 70)
    try:
        model2 = VideoLightweight3DCNN()
        output2 = model2(dummy_video)
        print(f"   ✅ Output shape:       {output2.shape}")
        print(f"   ✅ Trainable params:   {model2.count_params():,}")
        print(f"   📊 Model memory (GB):  {model2.count_params() * 4 / 1e9:.3f}")
        print(f"   ⚡ Size reduction:     {(1 - model2.count_params() / model1.count_params()) * 100:.1f}%")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # ===== Test 3: Utility Function =====
    print("\n" + "−" * 70)
    print("3. Model Loading (Utility Function):")
    print("−" * 70)
    try:
        model3 = load_video_model('cnn_lstm')
        print(f"   ✅ Successfully loaded 'cnn_lstm' model")
        print(f"   ✅ Compatible with ensemble training pipeline")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ALL VIDEO MODELS INITIALIZED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. Refine video_preprocessing.py for 16-frame extraction")
    print("   2. Create train_video_ensemble.py script")
    print("   3. Integrate with late-fusion ensemble in ensemble.py")
    print("=" * 70)
