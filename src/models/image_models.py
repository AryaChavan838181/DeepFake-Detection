import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# EFFICIENTNET-B7 (ALREADY PROVEN ON DFDC THEREFORE THE BASELINE)
class EfficientNetDetector(keras.Model):
    def __init__(self, input_shape=(226, 226, 3)):
        super(EfficientNetDetector, self).__init__()
        self.input_shape_val = input_shape

        base_model = keras.applications.EfficientNetB7(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )

        base_model.trainable = False #freeze it for transfer learning, so all the weights of the base model remain the same

        #custom
        self.base_model = base_model
        self.global_avg_pool = layers.GlobalAveragePooling2D() 
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=False)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        output = self.output_layer(x)
        return output
    
    def unfreeze_base(self, layer_count=20):
        #in this funct we will unfreeze all the layers, so that we can fine tune the model
        #earlier layers will be frozen still. so we are only freezing the last 20 layers for training.
        #the earlier layers have general features, we are training the high level features to fit our task
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-layer_count]:
            layer.trainable = False


# RESNET-50 + FREQUENCY DOMAIN
class ResNetFrequencyDetector(keras.Model):
    #using dual path resnet50:
    #path 1 is spatial domain (standard cnn)
    #path 2 is frequency domain (FFT based features)

    def __init__(self, input_shape=(226,226,3)):
        super(ResNetFrequencyDetector, self).__init__()
        self.input_shape_val = input_shape

        #path 1: spatial domain
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )

        base_model.trainable = False
        self.spatial_base = base_model
        self.spatial_pool = layers.GlobalAveragePooling2D()

        #path 2: frequency domain
        self.freq_dense1 = layers.Dense(256, activation='relu')
        self.freq_dropout = layers.Dropout(0.4)
        self.freq_dense2 = layers.Dense(128, activation='relu')

        #fusion
        self.fusion_dense = layers.Dense(256, activation='relu')
        self.fusion_dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def compute_frequency_features(self, images):
        #extracting frequency domain features from imgs
        #will return a frequency features tensor

        #convert to grayscale
        gray = tf.image.rgb_to_grayscale(images)
        gray = tf.squeeze(gray, axis=-1)

        #compute fft
        fft = tf.signal.fft2d(tf.cast(gray, tf.complex64))
        magnitude = tf.abs(fft)
        phase = tf.angle(fft)

        #reduce spatial dimensions
        magnitude_avg = tf.reduce_mean(magnitude, axis=[1,2])
        phase_avg = tf.reduce_mean(phase, axis=[1,2])

        #concatenate and normalize
        freq_features = tf.concat([magnitude_avg, phase_avg], axis=1)
        
        return freq_features

    def call(self, inputs, training=False):
        #path 1
        spatial_features = self.spatial_base(inputs, training=False)
        spatial_features = self.spatial_pool(spatial_features)

        #path 2
        freq_features = self.compute_frequency_features(inputs)
        freq_features = self.freq_dense1(freq_features)
        freq_features = self.freq_dropout(freq_features, training=training)
        freq_features = self.freq_dense2(freq_features)

        #fusion
        fused = tf.concat([spatial_features, freq_features], axis=1)

        #final classification layers
        fused = self.fusion_dense(fused)
        fused = self.fusion_dropout(fused, training=training)
        output = self.output_layer(fused)

        return output
    
    def unfreeze_bias(self, layer_count=20):
        self.spatial_base.trainable = True
        for layer in self.spatial_base.layers[:-layer_count]:
            layer.trainable = False


# VISION TRANSFORMER DETECTOR
class ViTDetector(keras.Model):
    def __init__(self, input_shape=(226, 226, 3), patch_size=16):
        super(ViTDetector, self).__init__()
        self.input_shape_val = input_shape
        self.patch_size = patch_size

        #img dimensions
        image_size = input_shape[0]
        num_patches = (image_size // patch_size)**2
        patch_dim = input_shape[2]*patch_size*patch_size

        #patch embedding
        self.patch_embed = layers.Dense(256)
        self.pos_embed = layers.Embedding(num_patches+1, 256)

        #transformer blocks
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=32)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256)
        ])

        #classification head
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def extract_patches(self, images):
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID'
        )

        batch_size = tf.shape(patches)[0]
        num_patches = tf.shape(patches)[1]*tf.shape(patches)[2]
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dim])

        return patches
    
    def call(self, inputs, training=False):
        # Extract patches
        patches = self.extract_patches(inputs)
        
        # Embed patches
        x = self.patch_embed(patches)
        
        # Add positional embeddings
        positions = tf.range(tf.shape(x)[1])
        positions = tf.expand_dims(positions, 0)
        pos_embeddings = self.pos_embed(positions)
        x = x + pos_embeddings
        
        # Transformer block
        attn_output = self.attention(x, x)
        x = self.norm1(attn_output + x)
        ffn_output = self.ffn(x)
        x = self.norm2(ffn_output + x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Classification
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        output = self.output_layer(x)
        
        return output


# UTILITY FUNCTIONS
def load_image_model(model_type: str, weights_path: str = None) -> keras.Model:
    models = {
        'efficientnet' : EfficientNetDetector,
        'resnet_freq' : ResNetFrequencyDetector,
        'vit' : ViTDetector
    }

    if model_type not in models:
        raise ValueError(f"model_type must be one of {list(models.keys())}")
    
    model = models[model_type]()
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model


if __name__ == "__main__":
    print("Testing image models...")
    
    # Create dummy input
    dummy_input = np.random.randn(2, 226, 226, 3).astype(np.float32) / 255.0
    
    # Test EfficientNet
    print("\n1. EfficientNet-B7:")
    model1 = EfficientNetDetector()
    output1 = model1(dummy_input)
    print(f"   ✅ Output shape: {output1.shape}")
    print(f"   ✅ Trainable params: {model1.count_params():,}")
    
    # Test ResNet + Frequency
    print("\n2. ResNet-50 + Frequency Domain:")
    model2 = ResNetFrequencyDetector()
    output2 = model2(dummy_input)
    print(f"   ✅ Output shape: {output2.shape}")
    print(f"   ✅ Trainable params: {model2.count_params():,}")
    
    # Test ViT
    print("\n3. Vision Transformer:")
    model3 = ViTDetector()
    output3 = model3(dummy_input)
    print(f"   ✅ Output shape: {output3.shape}")
    print(f"   ✅ Trainable params: {model3.count_params():,}")
    
    print("\n✅ All image models initialized successfully!")

