#!/usr/bin/env python3.8
"""

"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_datasets as tfds


def make_simple_cnn_model(input_shape: Tuple[int, ...],
                          output_shape: int,
                          num_conv_layers: int):
    """Create a simplistic CNN model.

    The model is shaped like:

    [ Input ]
        | (shape: input_shape)
        V
    [ Convolutional Layer ]
        | (shape: 32)
        V
    [ Convolutional Layer ]
        | (shape: 64)
        V
    [ Convolutional Layer ]
        | (shape: 128)
        V
    [ ... ] (total conv layers = num_conv_layers)
        |
        V
    [ Dense Layer ]
        | (shape: output_shape)
        V

    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    for n in range(num_conv_layers):
        size = 32 * 2 ** n  # 32, 64, 128, ...

        model.add(tf.keras.layers.Conv2D(size, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

    return model


def main():
    datasets, info = tfds.load(
        'emnist',
        split=None,
        with_info=True,
        as_supervised=True,
        data_dir='data',
        download=True,
    )

    train_ds = datasets['train']
    test_ds = datasets['test']

    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(lambda img, label: (tf.image.convert_image_dtype(img, dtype=tf.float32), label))

    model = make_simple_cnn_model(
        input_shape=info.features['image'].shape,
        output_shape=info.features['label'].num_classes,
        num_conv_layers=2,
    )

    model.compile(
        optimizer='Adam',
        metrics=['accuracy'],
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    model.fit(
        train_ds,
        epochs=10,
        verbose=1,
    )


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())

    main(**args)


if __name__ == '__main__':
    cli()
