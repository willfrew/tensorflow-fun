# Based on
#   https://keras.io/examples/generative/dcgan_overriding_train_step/

import numpy as np
import requests
import os
import tarfile
from tempfile import mkdtemp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGES_FILE="/cache/images.tar"
IMAGES_URL="http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
IMAGES_UNPACK_DIR="/cache/doggan_images/"

def get_dataset_from_tarball(tarfile, unpack_dir):
    tarfile.extractall(path=unpack_dir)
    tarfile.close()
    dataset = keras.preprocessing.image_dataset_from_directory(
        unpack_dir,
        labels=None, # No classification required.
        image_size=(64, 64),
        color_mode='rgb',
        shuffle=True,
        batch_size=32,
    )
    dataset = dataset.map(lambda x: x / 255.0)
    return dataset

def get_images_tarball(filename, fallback_url):
    try:
        return tarfile.open(filename)
    except FileNotFoundError:
        req = requests.get(fallback_url)
        with open(filename, 'wb') as fd:
            for chunk in req.iter_content(chunk_size=128):
                fd.write(chunk)
        return tarfile.open(filename)

def discriminator_model():
    return keras.Sequential(
        [
            keras.Input(shape=(64, 64, 3)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

def generator_model(latent_space_dimensions):
    return keras.Sequential(
        [
            keras.Input(shape=(latent_space_dimensions,)),
            layers.Dense(8 * 8 * 128),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_space_size):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_space_size = latent_space_size

    def compile(self, d_optimiser, g_optimiser, loss_fn):
        super(GAN, self).compile()
        self.d_optimiser = d_optimiser
        self.g_optimiser = g_optimiser
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_space_size))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat(
            [generated_images, real_images],
            axis=0
        )

        # Label up the images as real/fake
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))],
            axis=0,
        )
        # Add random noise to the labels
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimiser.apply_gradients(
            zip(grads, self.discriminator.trainable_weights),
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_space_size))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimiser.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_images, latent_space_size):
        self.num_images = num_images
        self.latent_space_size = latent_space_size

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_images, self.latent_space_size))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_images):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("/build/generated_img_%03d_%d.png" % (epoch, i))


def main():
    tarball = get_images_tarball(IMAGES_FILE, IMAGES_URL)
    dataset = get_dataset_from_tarball(tarball, IMAGES_UNPACK_DIR)

    discriminator = discriminator_model()
    discriminator.summary()
    latent_space_size=128
    generator = generator_model(latent_space_size)
    generator.summary()

    epochs=6

    gan = GAN(
        discriminator=discriminator,
        generator=generator,
        latent_space_size=latent_space_size,
    )
    gan.compile(
        d_optimiser=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimiser=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    gan.fit(
        dataset,
        epochs=epochs,
        callbacks=[
            GANMonitor(num_images=10, latent_space_size=latent_space_size)
        ]
    )

    discriminator.save("/build/doggan_discriminator.h5")
    generator.save("/build/doggan_generator.h5")

main()
