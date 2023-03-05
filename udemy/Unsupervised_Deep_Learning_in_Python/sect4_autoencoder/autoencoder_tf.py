#!/bin/env python
# Course link:
#   https://www.udemy.com/course/unsupervised-deep-learning-in-python/learn/lecture/7342180

from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def get_mnist(limit=None):
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = X_train.astype('float32') / 255.
  X_test = X_test.astype('float32') / 255.
  X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
  X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
  print(f'X_train.shape={X_train.shape}')
  print(f'X_test.shape={X_test.shape}')
  return X_train, y_train, X_test, y_train


class Autoencoder:
  def __init__(self, input_data_size: int, encoder_size: int):
    # This is the size of our encoded representations
    self.encoder_size = encoder_size

    # This is our input image
    self.input_data_size = input_data_size

    self._compose_model()

  def _compose_model(self):
    """Defines the autoencoder."""
    self.input_dim = tf.keras.Input(shape=(self.input_data_size,))

    # "encoded" is the encoded representation of the input
    encoded = tf.keras.layers.Dense(
        self.encoder_size, activation='relu')(self.input_dim)

    # "decoded" is the lossy reconstruction of the input
    decoded = tf.keras.layers.Dense(
        self.input_data_size, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    self._autoencoder = tf.keras.Model(self.input_dim, decoded)

    # This model maps an input to its encoded representation
    self._encoder = tf.keras.Model(self.input_dim, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = tf.keras.Input(shape=(self.encoder_size,))

    # Retrieve the last layer of the autoencoder model
    decoder_layer = self._autoencoder.layers[-1]

    # Create the decoder model
    self._decoder = tf.keras.Model(encoded_input, decoder_layer(encoded_input))

    # First, we'll configure our model to use a per-pixel binary crossentropy
    # loss, and the Adam optimizer:
    self._autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

  @property
  def autoencoder(self):
    return self._autoencoder

  @property
  def encoder(self):
    return self._encoder

  @property
  def decoder(self):
    return self._decoder

  def fit(self, X_train, X_test, epochs: int=30, batch_size: int=64):
    history = self._autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test))

    return history

  def predict(self, X, only_decoded_data: bool=True):
    encoded_data = self.encoder.predict(X)
    decoded_data = self.decoder.predict(encoded_data)
    if only_decoded_data:
      return decoded_data

    return encoded_data, decoded_data


def main():
  X_train, y_train, X_test, y_test = get_mnist()

  model = Autoencoder(784, 300)
  model.fit(X_train, X_test, epochs=10)

  # plot reconstruction
  done = False
  while not done:
    i = np.random.choice(len(X_test))
    x = X_test[i]
    im = model.predict(np.array([x])).reshape(28, 28)
    plt.subplot(1,2,1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(im, cmap='gray')
    plt.title("Reconstruction")
    plt.show()

    ans = input("Generate another?")
    if ans and ans[0] in ('n' or 'N'):
      done = True

if __name__ == '__main__':
  main()
