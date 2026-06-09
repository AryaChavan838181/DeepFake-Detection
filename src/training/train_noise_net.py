"""
Noise Net Training Script (Slim)
================================

Trains NoiseNet on DFDC/FaceForensics++ images.

Usage:
    python src/training/train_noise_net.py --data ./data/dfdc_sample --epochs 20 --batch 32
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
from src.models.noise_net import NoiseNet, combined_loss


def load_image_dataset(data_dir, batch_size=32, target_size=(226, 226), val_split=0.15):
    """Load images from Real/ and Fake/ subdirectories."""
    
    print(f"Loading images from {data_dir}...")
    
    all_images = []
    
    # Load from Real and Fake folders
    for folder in ['Real', 'Fake']:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(folder_path, img_file))
    
    if not all_images:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Found {len(all_images)} images")
    
    # Split into train/val
    np.random.shuffle(all_images)
    val_count = int(len(all_images) * val_split)
    train_images = all_images[val_count:]
    val_images = all_images[:val_count]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    def load_and_preprocess(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, target_size)
        return image
    
    train_ds = tf.data.Dataset.from_tensor_slices(train_images) \
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .shuffle(len(train_images)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices(val_images) \
        .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


def train_noise_net(data_dir, epochs=20, batch_size=32, learning_rate=1e-4,
                    checkpoint_dir='saved_models/noise_net', weights_path=None):
    """
    Train NoiseNet model.
    
    Args:
        data_dir: Directory with Real/ and Fake/ subdirectories
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Where to save checkpoints
        weights_path: Optional path to resume from
    """
    
    print("=" * 70)
    print("NOISE NET TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading dataset...")
    train_ds, val_ds = load_image_dataset(data_dir, batch_size=batch_size)
    
    # Initialize model
    print("\n[2/3] Initializing NoiseNet...")
    model = NoiseNet(input_shape=(226, 226, 3), epsilon=0.01)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=['mae']
    )
    
    # Callbacks
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'noise_net_best.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
    
    # Train
    print("\n[3/3] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'noise_net_final.h5')
    model.save_weights(final_path)
    print(f"\n✓ Model saved: {final_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NoiseNet")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data directory (Real/ and Fake/ subdirs)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default='saved_models/noise_net')
    parser.add_argument("--weights", type=str, default=None)
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_noise_net(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        weights_path=args.weights
    )