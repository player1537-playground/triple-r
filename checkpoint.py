#!/usr/bin/env python3.8
"""

"""

from dataclasses import dataclass
from functools import partial
import logging
import os
from pathlib import Path
import re
from shutil import rmtree
import subprocess
from time import sleep
from typing import Tuple, Optional

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('PIL.Image').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import whatreallyhappened as wrh

np.set_string_function(lambda x: f'array(..., dtype={x.dtype}, shape={x.shape})')


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


def make_resnet50_model(input_shape: Tuple[int, ...],
                        output_shape: int):
    model = tf.keras.applications.ResNet50(
        weights=None,  # random initialization
        input_shape=input_shape,
        classes=output_shape,
        include_top=True,  # include fully-connected layer
    )

    # Thanks https://github.com/bnicolae/ai-apps/blob/master/resnet-50/keras_resnet50.py#L120-L131
    # ResNet-50 model that is included with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = tf.keras.regularizers.l2(0.00005)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == tf.keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = tf.keras.models.Model.from_config(model_config)

    return model


def make_wide_model(input_shape: Tuple[int, ...],
                    output_shape: int,
                    width: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    model.add(tf.keras.layers.Conv2D(width, kernel_size=(3, 3), activation='relu'))

    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

    return model


def make_long_model(input_shape: Tuple[int, ...],
                    output_shape: int,
                    width: int,
                    length: int):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    for i in range(length):
        model.add(tf.keras.layers.Conv2D(width, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

    return model


def main(iterations, make_model_fn, format, checkpoint_dir, log_to):
    wrh.open(str(log_to) % {
    }, 'a')

    wrh.push('checkpoint.py')

    wrh.log('model', '%s', make_model_fn.__name__)
    wrh.log('format', '%s', format)
    wrh.log('checkpoint_dir', '%s', checkpoint_dir)
    wrh.log('iterations', '%d', iterations)

    input_shape = (224, 224, 3)
    output_shape = 200

    wrh.push('creating model')
    model = make_model_fn(
        input_shape=input_shape,
        output_shape=output_shape,
    )
    wrh.pop('creating model')

    for i in range(iterations):
        wrh.push('checkpoint')

        if format == 'hdf5':
            weights = checkpoint_dir / 'checkpoint.h5'
            model.save(weights)
            size = weights.stat().st_size

        elif format == 'tensorflow':
            weights = checkpoint_dir / 'checkpoint'
            rmtree(weights, ignore_errors=True)
            model.save(weights)
            size = sum(f.stat().st_size for f in weights.glob('**/*') if f.is_file())

        if i == 0:
            print(f'{size},', end='', file=os.fdopen(100, "w"))

        wrh.log('size', '%lu', size)
        wrh.pop('checkpoint')

    wrh.pop('checkpoint.py')


def cli():
    def make_model_fn(s):
        if s.startswith('CNN-'):
            num_conv_layers = int(s[len('CNN-'):])
            fn = partial(make_simple_cnn_model, num_conv_layers=num_conv_layers)

        elif s.startswith('Wide-'):
            width = int(s[len('Wide-'):])
            fn = partial(make_wide_model, width=width)

        elif s.startswith('Long-'):
            width, length = map(int, s[len('Long-'):].split('-', 1))
            fn = partial(make_long_model, width=width, length=length)

        elif s == 'ResNet50':
            fn = make_resnet50_model

        fn.__name__ = s
        return fn

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--model', dest='make_model_fn', required=True, type=make_model_fn)
    parser.add_argument('--format', required=True, choices=('hdf5', 'tensorflow'))
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--log-to', required=True, type=Path)
    args = vars(parser.parse_args())

    main(**args)


if __name__ == '__main__':
    cli()
