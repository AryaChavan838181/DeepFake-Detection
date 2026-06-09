"""
VIDEO ENSEMBLE TRAINING SCRIPT
===============================
Trains the 3D-CNN + LSTM video deepfake detector on the DFDC dataset.

Supports:
- Streaming video-to-sequence conversion for memory efficiency
- Multi-model ensemble (primary CNN-LSTM + lightweight variant)
- Mini-batch training from 10GB DFDC chunks
- Temporal jitter artifact capture
- Resume from checkpoints

Usage:
    python src/training/train_video_ensemble.py --data /path/to/videos --epochs 5 --batch 2
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks as keras_callbacks
import numpy as np
import yaml
import argparse
from typing import Tuple, Generator, List

sys.path.append('.')
from src.models.video_models import VideoCNNLSTM, VideoLightweight3DCNN, load_video_model
from src.preprocessing.video_preprocessing import VideoPreprocessor


# ============================================================================
# VIDEO ENSEMBLE ARCHITECTURE
# ============================================================================
class VideoEnsemble(keras.Model):
    """
    Fuses the video-based architectures: Primary 3D-CNN + LSTM (main detector)
    and Lightweight variant (fast secondary detector for ensemble voting).
    
    This is the VIDEO component that will be fused with IMAGE + AUDIO
    in the final late-fusion ensemble.
    
    Strategy:
    - Primary model (VideoCNNLSTM): Full feature extraction
    - Lightweight model: Quick validation / ensemble diversity
    - Fusion layer: Learns to weight both predictions
    
    Input shape: (batch_size, num_frames, height, width, channels)
    """
    
    def __init__(self, input_shape=(16, 226, 226, 3), use_lightweight=False):
        super(VideoEnsemble, self).__init__()
        
        self.input_shape_val = input_shape
        self.num_frames = input_shape[0]
        self.use_lightweight = use_lightweight
        
        # Primary 3D-CNN + LSTM model
        self.primary_model = VideoCNNLSTM(input_shape=input_shape)
        
        # Optional: Lightweight model for diversity
        if use_lightweight:
            self.lightweight_model = VideoLightweight3DCNN(input_shape=input_shape)
        
        # Fusion layers: Learn to combine predictions
        if use_lightweight:
            # Two predictions to fuse
            self.fusion_dense1 = layers.Dense(32, activation='relu')
            self.fusion_dropout = layers.Dropout(0.3)
            self.fusion_dense2 = layers.Dense(16, activation='relu')
        else:
            # Single prediction, just refinement
            self.fusion_dense1 = layers.Dense(32, activation='relu')
            self.fusion_dropout = layers.Dropout(0.2)
            self.fusion_dense2 = layers.Dense(16, activation='relu')
        
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """
        Forward pass through video ensemble.
        
        Args:
            inputs: (batch_size, num_frames, height, width, channels)
            training: Boolean flag for dropout/batch norm
        
        Returns:
            output: (batch_size, 1) deepfake probability
        """
        # Primary model prediction
        primary_out = self.primary_model(inputs, training=training)
        
        # Optionally combine with lightweight model
        if self.use_lightweight:
            lightweight_out = self.lightweight_model(inputs, training=training)
            combined = tf.concat([primary_out, lightweight_out], axis=1)
        else:
            combined = primary_out
        
        # Fusion
        x = self.fusion_dense1(combined)
        x = self.fusion_dropout(x, training=training)
        x = self.fusion_dense2(x)
        output = self.output_layer(x)
        
        return output


# ============================================================================
# VIDEO SEQUENCE GENERATOR (For Memory-Safe Mini-Batch Training)
# ============================================================================
class VideoSequenceGenerator:
    """
    Generates batches of video sequences for training.
    
    Memory strategy:
    - Stream videos from disk on-the-fly
    - Extract 16-frame sequences with sliding window
    - Batch multiple sequences together
    - Only keep current batch in memory
    
    This allows training on 10GB DFDC chunks without loading entire videos.
    """
    
    def __init__(
        self,
        video_dir: str,
        batch_size: int = 2,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (226, 226),
        stride: int = 8
    ):
        """
        Initialize video sequence generator.
        
        Args:
            video_dir: Directory or list of video paths
            batch_size: Sequences per batch
            num_frames: Frames per sequence
            target_size: Target frame resolution
            stride: Frame stride between sequences (8 = 50% overlap)
        """
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.target_size = target_size
        self.stride = stride
        self.preprocessor = VideoPreprocessor(
            target_size=target_size,
            num_frames=num_frames
        )
        
        # Placeholder: Will be replaced with actual video paths
        self.video_paths = []
        self.labels = []
        
    def load_video_paths(self, real_dir: str, fake_dir: str):
        """
        Load video file paths from Real/Fake folders.
        
        Args:
            real_dir: Directory with real videos (label=0)
            fake_dir: Directory with deepfake videos (label=1)
        """
        import glob
        
        real_videos = glob.glob(os.path.join(real_dir, "*.mp4"))
        fake_videos = glob.glob(os.path.join(fake_dir, "*.mp4"))
        
        self.video_paths = real_videos + fake_videos
        self.labels = [0] * len(real_videos) + [1] * len(fake_videos)
        
        print(f"Loaded {len(real_videos)} real and {len(fake_videos)} fake videos")
        return len(self.video_paths)
    
    def extract_frames_from_video(self, video_path: str, max_frames: int = 32):
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (memory safe)
        
        Returns:
            List of frame arrays or None if failed
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            frames = []
            frame_count = 0
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to target size
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if len(frames) >= self.num_frames:
                return frames
            else:
                return None
                
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return None
    
    def generate_batches(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of video sequences for training.
        
        Yields:
            (X_batch, y_batch): Sequences and labels
                X_batch shape: (batch_size, num_frames, height, width, 3)
                y_batch shape: (batch_size,)
        """
        batch_sequences = []
        batch_labels = []
        
        for video_path, label in zip(self.video_paths, self.labels):
            # Load and preprocess frames
            frames = self.extract_frames_from_video(video_path)
            if frames is None:
                continue
            
            # Preprocess frames
            preprocessed = self.preprocessor.preprocess_frames(frames)
            if len(preprocessed) == 0:
                continue
            
            # Extract sequences with sliding window
            sequences = self.preprocessor.extract_frame_sequences(
                list(preprocessed),
                stride=self.stride,
                allow_padding=False
            )
            
            # Add to batch
            for seq in sequences:
                batch_sequences.append(seq)
                batch_labels.append(label)
                
                # Yield when batch is full
                if len(batch_sequences) >= self.batch_size:
                    X_batch = np.array(batch_sequences)
                    y_batch = np.array(batch_labels)
                    
                    yield X_batch, y_batch
                    
                    batch_sequences = []
                    batch_labels = []
        
        # Yield remaining batch if any
        if batch_sequences:
            X_batch = np.array(batch_sequences)
            y_batch = np.array(batch_labels)
            yield X_batch, y_batch


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_video_ensemble(
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 2,
    weights_path: str = None,
    model_type: str = 'lightweight',
    learning_rate: float = 1e-4
):
    """
    Train video deepfake detector on DFDC video data.
    
    Args:
        data_dir: Path to data with 'Real' and 'Fake' subdirectories
        epochs: Number of training epochs per batch
        batch_size: Sequences per batch (memory constrained on Windows)
        weights_path: Optional path to resume from checkpoint
        model_type: Model to train ('lightweight', 'cnn_lstm', or 'ensemble')
                   - lightweight: VideoLightweight3DCNN only (DEFAULT - fast)
                   - cnn_lstm: VideoCNNLSTM only (full accuracy)
                   - ensemble: Both VideoCNNLSTM + VideoLightweight (voting)
        learning_rate: Adam optimizer learning rate
    """
    
    print("=" * 70)
    print("VIDEO DEEPFAKE DETECTOR TRAINING")
    print("=" * 70)
    print(f"\n📊 Training Mode: {model_type.upper()}")
    
    # ===== 1. SETUP MODEL =====
    print("\n[1/5] Initializing Model...")
    
    # Determine which models to train
    use_ensemble = (model_type == 'ensemble')
    use_cnn_lstm = (model_type in ['cnn_lstm', 'ensemble'])
    use_lightweight = (model_type in ['lightweight', 'ensemble'])
    
    if model_type == 'lightweight':
        # Train lightweight model only - faster, simpler
        print("   Mode: LIGHTWEIGHT MODEL (Fast Training)")
        model = VideoLightweight3DCNN()
        print(f"   ✅ Model: VideoLightweight3DCNN")
        
    elif model_type == 'cnn_lstm':
        # Train full CNN-LSTM model only - maximum accuracy
        print("   Mode: FULL CNN-LSTM (Maximum Accuracy)")
        model = VideoCNNLSTM()
        print(f"   ✅ Model: VideoCNNLSTM")
        
    elif model_type == 'ensemble':
        # Train both models with learned fusion - best of both
        print("   Mode: ENSEMBLE (Voting + Fusion)")
        model = VideoEnsemble(use_lightweight=True)
        print(f"   ✅ Model: VideoEnsemble (VideoCNNLSTM + VideoLightweight)")
        
    else:
        raise ValueError(f"model_type must be 'lightweight', 'cnn_lstm', or 'ensemble', got '{model_type}'")
    
    # Load weights if continuing
    if weights_path and os.path.exists(weights_path):
        print(f"     Loading weights from: {weights_path}")
        model.load_weights(weights_path)
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 16, 226, 226, 3))
    _ = model(dummy_input)
    
    print(f"     ✅ Model initialized")
    print(f"     Total parameters: {model.count_params():,}")
    
    # ===== 2. COMPILE MODEL =====
    print("\n[2/5] Compiling Model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    print(f"     ✅ Model compiled with Adam (lr={learning_rate})")
    
    # ===== 3. SETUP DATA GENERATOR =====
    print("\n[3/5] Setting up Video Sequence Generator...")
    
    # Check if data_dir has Real/Fake structure
    real_dir = os.path.join(data_dir, 'Real')
    fake_dir = os.path.join(data_dir, 'Fake')
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"     ⚠️  Expected structure: {data_dir}/Real and {data_dir}/Fake")
        print(f"     Creating dummy generator for testing...")
        # For testing without actual video data
        generator = VideoSequenceGenerator(data_dir, batch_size=batch_size)
        # Create synthetic data for demo
        def dummy_generator():
            for _ in range(10):
                X = np.random.rand(batch_size, 16, 226, 226, 3).astype(np.float32)
                y = np.random.randint(0, 2, batch_size).astype(np.float32)
                yield X, y
        train_gen = dummy_generator()
    else:
        generator = VideoSequenceGenerator(data_dir, batch_size=batch_size)
        num_videos = generator.load_video_paths(real_dir, fake_dir)
        print(f"     ✅ Loaded {num_videos} videos")
        train_gen = generator.generate_batches()
    
    # ===== 4. SETUP CALLBACKS =====
    print("\n[4/5] Setting up Callbacks...")
    os.makedirs('saved_models/checkpoints', exist_ok=True)
    
    checkpoint_callback = keras_callbacks.ModelCheckpoint(
        filepath='saved_models/checkpoints/video_ensemble_best.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stop_callback = keras_callbacks.EarlyStopping(
        monitor='val_auc',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard_callback = keras_callbacks.TensorBoard(
        log_dir='./logs/video_ensemble',
        histogram_freq=1
    )
    
    print(f"     ✅ Callbacks configured")
    print(f"        - Checkpoint: saved_models/checkpoints/video_ensemble_best.h5")
    print(f"        - Early stopping: patience=3")
    print(f"        - TensorBoard: ./logs/video_ensemble")
    
    # ===== 5. TRAIN =====
    print("\n[5/5] Training on mini-batch...")
    print(f"     Mode: {model_type.upper()}")
    print(f"     Epochs: {epochs}, Batch Size: {batch_size} sequences")
    
    # Memory estimation by model type
    if model_type == 'lightweight':
        mem_per_batch = batch_size * 9.35 / 2  # Lightweight uses ~half memory
    elif model_type == 'cnn_lstm':
        mem_per_batch = batch_size * 9.35
    else:  # ensemble
        mem_per_batch = batch_size * 9.35 * 1.5  # Ensemble uses more
    
    print(f"     Expected batch memory: ~{mem_per_batch:.1f} MB per step")
    print("-" * 70)
    
    try:
        # For mini-batch training with generator
        # Note: validation_split not supported with generators, use separate validation data
        history = model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=10,  # Limited steps per batch for demonstration
            validation_steps=2,  # Use same generator for validation
            callbacks=[checkpoint_callback, early_stop_callback, tensorboard_callback],
            verbose=1
        )
        
        print("-" * 70)
        print("\n✅ TRAINING COMPLETE")
        
        # Save final model
        final_path = 'saved_models/video_ensemble_final.h5'
        model.save_weights(final_path)
        print(f"   Final weights saved to: {final_path}")
        
        # Save training history
        history_path = 'saved_models/video_training_history.yaml'
        with open(history_path, 'w') as f:
            yaml.dump({
                'epochs': epochs,
                'batch_size': batch_size,
                'use_lightweight': use_lightweight,
                'final_loss': float(history.history['loss'][-1]) if 'loss' in history.history else None,
                'final_auc': float(history.history['auc'][-1]) if 'auc' in history.history else None
            }, f)
        print(f"   Training history saved to: {history_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    print("\n📝 Next Steps:")
    print("   1. Create video_loader.py for efficient video batching")
    print("   2. Create audio preprocessing and models (audio_models.py)")
    print("   3. Integrate Image + Video + Audio in ensemble.py (late-fusion)")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train video deepfake detector on DFDC mini-batch"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to video data folder (with 'Real' and 'Fake' subfolders)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per mini-batch"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size in sequences (default=4 for lightweight)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['lightweight', 'cnn_lstm', 'ensemble'],
        default='lightweight',
        help="Model to train: lightweight (DEFAULT - fast), cnn_lstm (full), or ensemble (voting)"
    )
    parser.add_argument(
        "--ensemble",
        action='store_true',
        help="Shortcut for --model ensemble (train both models together)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to weights from previous batch (resume training)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate"
    )
    
    args = parser.parse_args()
    
    # Handle --ensemble shortcut
    model_type = 'ensemble' if args.ensemble else args.model
    
    # Adjust default batch size based on model type
    if args.batch == 4 and model_type == 'ensemble':
        batch_size = 2  # Ensemble needs more memory
    else:
        batch_size = args.batch
    
    # Ensure output directory exists
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Train
    train_video_ensemble(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=batch_size,
        weights_path=args.weights,
        model_type=model_type,
        learning_rate=args.lr
    )
