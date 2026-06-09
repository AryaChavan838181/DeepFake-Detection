import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class NoiseNet(keras.Model):
    """
    Lightweight CNN for generating imperceptible perturbations.
    
    The perturbations are designed to:
    1. Be invisible to human eye (SSIM > 0.95)
    2. Disrupt deepfake generation quality
    3. Be generalizable across generator types
    
    Architecture: 6 convolutional layers with batch normalization
    - No pooling (maintain spatial resolution)
    - Symmetric (down-project then up-project features)
    - Output scaled to [-epsilon, epsilon] for imperceptibility
    """

    # initializing the noisenet model
    # epsilon = max perturbation magnitude (default 0.01 = around 2.5 pixel value)
    def __init__(self, input_shape=(226,226,3), epsilon=0.01, name="NoiseNet"):
        super(NoiseNet, self).__init(name=name)

        self.input_shape_val = input_shape
        self.epsilon = epsilon

        # encoder layers for progressive feature extraction
        self.conv1 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()

        # decoder layers for progressive feature synthesis
        self.conv4 = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.bn5 = layers.BatchNormalization()

        # output layer - generates 3-channel perturbation
        self.conv6 = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')


    # forward pass: generate perturbation from input image
    # returns: erturbation tensor (batch_size, 226, 226, 3) in [-epsilon, epsilon]
    def call(self, inputs, training=False):
        # encoder - extracts features
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)

        x = self.conv2(inputs)
        x = self.bn2(x, training=training)

        x = self.conv3(inputs)
        x = self.bn3(x, training=training)

        # decoder - synthesizes perturbation
        x = self.conv4(inputs)
        x = self.bn4(x, training=training)

        x = self.conv5(inputs)
        x = self.bn5(x, training=training)

        # output - scaled to [-epsilon, epsilon]
        noise = self.conv6(x) # tanh outputs [-1, 1]
        noise = noise * self.epsilon # scaled to epsilon range


    # input img can be (226, 226, 3) or (batch, 226, 226, 3) 
    # returns perturbation in same shape as input
    def generate_perturbation(self, image):
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0) # added batch dimension

        noise = self(image, training=False)
        return noise[0] if noise.shape[0] == 1 else noise


    # perturbed image = image + noise
    # returns img + pert, clipped to [0,1]
    def apply_perturbation(self, image):
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0) # added batch dimension

        noise = self(image, training=False)
        perturbed = image + noise
        perturbed = tf.clip_by_value(perturbed, 0.0, 1.0)

        return perturbed[0] if perturbed.shape[0] == 1 else perturbed
    

    def get_config(self):
        return {
            'input_shape': self.input_shape_val,
            'epsilon': self.epsilon
        }
    

# LOSS FUNCTIONS
def ssim_loss(original, perturbed, max_val=1.0):
    """
    SSIM loss to ensure imperceptibility.
    
    How it works:
    - Compare structural similarity between original and perturbed
    - SSIM > 0.95 means nearly identical images
    - We minimize (1 - SSIM) to maximize similarity
    
    Args:
        original: Original image tensor
        perturbed: Perturbed image tensor
        max_val: Maximum pixel value (1.0 for normalized images)
        
    Returns:
        loss: SSIM loss value (minimize this)
    """
    # Compute SSIM (higher is more similar)
    ssim_value = tf.image.ssim(original, perturbed, max_val=max_val)
    
    # Convert to loss (minimize 1 - SSIM)
    loss = 1.0 - tf.reduce_mean(ssim_value)

    return loss

def l2_regularization(noise, weight=0.01):
    """
    L2 regularization to keep perturbations small.
    
    Args:
        noise: Perturbation tensor
        weight: Regularization weight (applied as multiplier)
        
    Returns:
        l2_loss: Weighted L2 norm of perturbation
    """
    l2_norm = tf.norm(noise)
    return weight * l2_norm

def combined_loss(original, perturbed, noise, ssim_weight=1.0, l2_weight=0.01):
    """
    Combined loss: Imperceptibility + Perturbation magnitude control.
    
    Formula:
    Loss = SSIM_loss + lambda * L2_regularization
    
    Where:
    - SSIM_loss ensures image similarity (imperceptibility)
    - L2_regularization keeps perturbations small and bounded
    - λ balances the two objectives
    
    Args:
        original: Original image batch
        perturbed: Perturbed image batch
        noise: Perturbation tensor
        ssim_weight: Weight for SSIM loss
        l2_weight: Weight for L2 regularization
        
    Returns:
        total_loss: Combined loss value
    """
    ssim = ssim_loss(original, perturbed)
    l2 = l2_regularization(noise, weight=l2_weight)
    
    total_loss = ssim_weight * ssim + l2
    
    return total_loss
    

# utility functions
def create_noise_net(input_shape=(226, 226, 3), epsilon=0.01):
    """
    Factory function to create and compile NoiseNet model.
    
    Args:
        input_shape: Input image shape
        epsilon: Perturbation magnitude bound
        
    Returns:
        model: Compiled NoiseNet model ready for training
    """
    model = NoiseNet(input_shape=input_shape, epsilon=epsilon)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['mae']  # Mean absolute error for monitoring
    )
    
    return model

if __name__ == "__main__":
    # Quick test
    print("Testing NoiseNet...")
    model = NoiseNet()
    
    # Create dummy batch
    dummy_input = tf.random.normal((2, 226, 226, 3))
    
    # Generate noise
    noise = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Noise shape: {noise.shape}")
    print(f"Noise range: [{tf.reduce_min(noise).numpy():.6f}, {tf.reduce_max(noise).numpy():.6f}]")
    
    # Apply perturbation
    perturbed = model.apply_perturbation(dummy_input)
    print(f"Perturbed shape: {perturbed.shape}")
    print(f"Perturbed range: [{tf.reduce_min(perturbed).numpy():.6f}, {tf.reduce_max(perturbed).numpy():.6f}]")