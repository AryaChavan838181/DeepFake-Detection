import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import yaml
import argparse
from src.models.image_models import load_image_model
from src.data.image_loader import ImageLoader # Assuming this matches your Phase 1 implementation

def train_image_model(model_type, data_dir, epochs=10, batch_size=32, weights_path=None):
    """
    Local training script for image-based deepfake detection models.
    """
    print(f"--- Starting Local Training for {model_type} ---")
    
    # 1. Load Configurations
    config_path = os.path.join('config', 'training_config.yml')
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # 2. Initialize Model
    # model_type: 'efficientnet', 'resnet_freq', or 'vit'
    model = load_image_model(model_type, weights_path=weights_path)
    
    # 3. Compile Model
    lr = train_config.get('learning_rate', 1e-4)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    # 4. Initialize Data Loader
    # Note: Adjust these parameters based on your src/data/image_loader.py implementation
    loader = ImageLoader(
        base_dir=data_dir,
        batch_size=batch_size,
        target_size=(226, 226)
    )
    
    train_ds, val_ds = loader.get_datasets() # Assuming this returns split tf.data.Dataset
    
    # 5. Callbacks
    checkpoint_dir = 'saved_models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_type}_best.h5'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]
    
    # 6. Train
    print(f"Training {model_type} on local CPU/GPU...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # 7. Final Save
    save_path = f'saved_models/{model_type}_final.h5'
    model.save_weights(save_path)
    print(f"Final weights saved to {save_path}")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Training for Deepfake Image Models")
    parser.add_argument("--model", type=str, default="efficientnet", help="efficientnet, resnet_freq, or vit")
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed faces directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16) # Smaller batch for local testing
    
    args = parser.parse_args()
    
    # Ensure saved_models directory exists
    os.makedirs('saved_models', exist_ok=True)
    
    train_image_model(
        model_type=args.model,
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch
    )
