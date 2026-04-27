# Import necessary libraries
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import xarray as xr
import zarr

# Specify the path to your zipped Zarr file
zarr_file = 'remap_umfl_s_20220728T000000Z.nc.=.field_umfl_s.=.rank_0.zarr.zip'

# Open the Zarr store
# The 'field_umfl_s' part of the filename suggests this is the data variable.
try:
    # xarray can automatically handle Zarr stores within a zip archive.
    # We specify the engine as 'zarr' and tell it to use the `fsspec` library
    # to open the zip file.
    ds = xr.open_zarr(zarr_file, engine='zarr')

    # Access the 'field_umfl_s' variable
    data_array = ds['field_umfl_s']

    # Convert the xarray DataArray to a NumPy array
    x_train_np = data_array.values

    print("Conversion successful! Here is the shape of the numpy array:")
    print(numpy_array.shape)

except FileNotFoundError:
    print(f"Error: The file '{zarr_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# --- 1. SET UP THE DISTRIBUTED TRAINING ENVIRONMENT ---

# This is a crucial step. In a real-world scenario, each of your 10 nodes
# would have a TF_CONFIG environment variable set. This JSON tells
# TensorFlow about the cluster. We simulate it here for demonstration.
#
# A typical TF_CONFIG would look like this on one of the machines:
# os.environ['TF_CONFIG'] = json.dumps({
#     'cluster': {
#         'worker': ['host1:port', 'host2:port', ..., 'host10:port']
#     },
#     'task': {'type': 'worker', 'index': 0}
# })
# On a different machine, 'index' would be 1, and so on.

# For this example, we will simulate the strategy for a single machine
# with multiple GPUs, but the code scales to multiple nodes.
# MultiWorkerMirroredStrategy handles both cases seamlessly.
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# --- 2. DEFINE THE VAE MODEL AND TRAINING LOGIC ---

# The VAE model is a subclass of tf.keras.Model. This makes it
# easy to train with Keras's built-in `fit` method.
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        # Forward pass through the VAE
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

    def train_step(self, data):
        # A custom training step to handle the VAE's specific loss function
        with tf.GradientTape() as tape:
            # Get outputs from the VAE model
            reconstructed, z_mean, z_log_var = self(data, training=True)
            
            # Calculate the reconstruction loss (e.g., Mean Squared Error)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstructed), axis=(1, 2)
                )
            )
            
            # Calculate the KL divergence loss for regularization
            # It encourages the latent distribution to be close to a normal distribution.
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Total loss is the sum of reconstruction and KL divergence losses
            total_loss = reconstruction_loss + kl_loss
        
        # Compute gradients and apply them to the model weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update the loss trackers for monitoring
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return a dictionary of the updated loss values
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# --- 3. DEFINE THE MODEL ARCHITECTURE INSIDE THE STRATEGY SCOPE ---

# The `strategy.scope()` is where the distributed magic happens.
# All model and optimizer creation must happen within this scope.
with strategy.scope():
    # Define the encoder part of the VAE
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    # A custom sampling layer to generate latent vectors
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Define the decoder part of the VAE
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(decoder_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    # Instantiate the VAE model and compile it with a suitable optimizer
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

# --- 4. PREPARE THE DATASET ---

# Create a tf.data.Dataset from the NumPy array.
# The `batch_size` is per-replica, so the global batch size is `batch_size * num_replicas`.
# This is a key detail for efficient distributed training.
batch_size_per_replica = 128
num_replicas = strategy.num_replicas_in_sync
global_batch_size = batch_size_per_replica * num_replicas

# Create a tf.data.Dataset and set a high buffer size for shuffling.
dataset = tf.data.Dataset.from_tensor_slices(x_train_np).shuffle(1024).batch(global_batch_size)
# The `distribute` method distributes the dataset across all workers.
dist_dataset = strategy.experimental_distribute_dataset(dataset)

# --- 5. TRAIN THE MODEL ---

# Train the VAE using the distributed dataset.
# The training is now automatically handled across all nodes by the strategy.
print("Starting distributed training...")
vae.fit(dist_dataset, epochs=10)

print("Distributed VAE training completed.")
