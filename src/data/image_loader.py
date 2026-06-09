import os
import tensorflow as tf

class ImageLoader:
    """
    Loads preprocessed facial images for training the image models.
    Expects data to be organized in folders by class:
    base_dir/
        REAL/
            img1.jpg
            ...
        FAKE/
            img1.jpg
            ...
    """
    def __init__(self, base_dir, batch_size=32, target_size=(226, 226), validation_split=0.2):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.validation_split = validation_split

    def get_datasets(self):
        """
        Returns (train_dataset, validation_dataset)
        """
        print(f"Loading data from {self.base_dir}...")
        
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.base_dir,
            validation_split=self.validation_split,
            subset="training",
            seed=123,
            image_size=self.target_size,
            batch_size=self.batch_size,
            label_mode='binary'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.base_dir,
            validation_split=self.validation_split,
            subset="validation",
            seed=123,
            image_size=self.target_size,
            batch_size=self.batch_size,
            label_mode='binary'
        )

        # Optimize for performance
        def process_images(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            mean = tf.constant([0.485, 0.456, 0.406], tf.float32)
            std = tf.constant([0.229, 0.224, 0.225], tf.float32)
            image = (image - mean) / std
            return image, label

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(process_images, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(process_images, num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds
