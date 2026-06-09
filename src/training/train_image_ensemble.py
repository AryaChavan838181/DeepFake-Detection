import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import yaml
import argparse
from src.models.image_models import EfficientNetDetector, ResNetFrequencyDetector, ViTDetector
from src.data.image_loader import ImageDataLoader

# ENSEMBLE ARCHITECTURE (Specifically for the 3 image models)
class ImageEnsemble(keras.Model):
    """
    Fuses the 3 image-based architectures: EfficientNet, ResNet-Freq, and ViT.
    This is what you'll actually use for the 'Image' part of your project.
    """
    def __init__(self, input_shape=(226, 226, 3)):
        super(ImageEnsemble, self).__init__()
        
        # 1. Initialize the 3 sub-models
        self.eff_net = EfficientNetDetector(input_shape=input_shape)
        self.res_freq = ResNetFrequencyDetector(input_shape=input_shape)
        self.vit_net = ViTDetector(input_shape=input_shape)
        
        # 2. Fusion Layer (Learns which model to trust more)
        self.fusion_dense = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Pass input through all 3 models
        out1 = self.eff_net(inputs, training=training)
        out2 = self.res_freq(inputs, training=training)
        out3 = self.vit_net(inputs, training=training)
        
        # Concatenate their individual scores
        combined = tf.concat([out1, out2, out3], axis=1)
        
        # Final decision
        x = self.fusion_dense(combined)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

def train_ensemble(data_dir, epochs=10, batch_size=16, weights_path=None):
    print("--- Initializing IMAGE ENSEMBLE (3-in-1) ---")
    
    # 1. Setup Model
    model = ImageEnsemble()
    
    # 2. Load Weights if Continuing Training
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from previous batch: {weights_path}")
        model.load_weights(weights_path)
    
    # 3. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # 4. Use the Updated ImageDataLoader (Memory Safe)
    loader = ImageDataLoader(batch_size=batch_size)
    train_gen, val_gen = loader.get_generators(data_dir)
    
    # 5. Callbacks
    os.makedirs('saved_models/checkpoints', exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='saved_models/checkpoints/image_ensemble_best.h5',
            save_best_only=True, monitor='val_auc', mode='max'
        ),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
    
    # 6. Train using Generators
    print(f"Training on 10GB batch in {data_dir}...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # 7. Final Save
    save_path = 'saved_models/image_ensemble_final.h5'
    model.save_weights(save_path)
    print(f"Batch training complete. Final weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Folder containing 'Real' and 'Fake' subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--weights", type=str, help="Path to weights from previous batch")
    
    args = parser.parse_args()
    
    # Ensure saved_models exists
    os.makedirs('saved_models', exist_ok=True)
    
    train_ensemble(args.data, args.epochs, args.batch, args.weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Folder containing 'Real' and 'Fake' subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8) # Lowered for local safety
    parser.add_argument("--weights", type=str, help="Path to weights from previous 10GB batch")
    
    args = parser.parse_args()
    train_ensemble(args.data, args.epochs, args.batch, args.weights)
