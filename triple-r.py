#!/usr/bin/env python3.8
"""

"""

from time import sleep
from typing import Tuple

import horovod.tensorflow.keras as hvd
from horovod.tensorflow import join as hvd_join
import matplotlib.pyplot as plt
from mpi4py import MPI
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


def tensorflow_main(data_dir):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    datasets, info = tfds.load(
        'emnist',
        split=None,
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
        download=True,
    )

    train_ds = datasets['train']
    test_ds = datasets['test']

    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(lambda img, label: (tf.image.convert_image_dtype(img, dtype=tf.float32), label))
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    with strategy.scope():
        model = make_simple_cnn_model(
            input_shape=info.features['image'].shape,
            output_shape=info.features['label'].num_classes,
            num_conv_layers=2,
        )

        opt = tf.keras.optimizers.Adam()

        model.compile(
            optimizer=opt,
            metrics=['accuracy'],
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

    model.fit(
        train_ds,
        epochs=10,
        steps_per_epoch=500,
        verbose=1,
    )


def horovod_main(data_dir):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()
    world_sans_last = world.Split(MPI.UNDEFINED if rank == size -1 else 1, rank)

    hvd.init(world)

    assert hvd.mpi_threads_supported()

    print(f'{hvd.local_rank() = }')

    datasets, info = tfds.load(
        'emnist',
        split=None,
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
        download=True,
    )

    train_ds = datasets['train']
    test_ds = datasets['test']

    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(lambda img, label: (tf.image.convert_image_dtype(img, dtype=tf.float32), label))

    model = make_simple_cnn_model(
        input_shape=info.features['image'].shape,
        output_shape=info.features['label'].num_classes,
        num_conv_layers=2,
    )

    scaled_lr = 0.001 * hvd.size()
    opt = tf.keras.optimizers.Adam(scaled_lr)
    opt = hvd.DistributedOptimizer(
        opt,
        backward_passes_per_step=1,
        average_aggregated_gradients=True,
    )

    model.compile(
        optimizer=opt,
        metrics=['accuracy'],
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        experimental_run_tf_function=False,
    )

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=scaled_lr,
            warmup_epochs=3,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        steps_per_epoch=21811 // hvd.size() // 100,
        callbacks=callbacks,
        epochs=1,
        verbose=1 if hvd.rank() == 0 else 0,
    )
    
    hvd.shutdown()

    if rank != size - 1:

        hvd.init(world_sans_last)

        model.fit(
            train_ds,
            steps_per_epoch=21811 // hvd.size() // 100,
            callbacks=callbacks,
            epochs=1,
            verbose=1 if hvd.rank() == 0 else 0,
        )

        hvd.shutdown()


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(required=True)

    horovod = subparsers.add_parser('horovod')
    horovod.set_defaults(main=horovod_main)
    horovod.add_argument('--data-dir', required=True)

    tensorflow = subparsers.add_parser('tensorflow')
    tensorflow.set_defaults(main=tensorflow_main)
    tensorflow.add_argument('--data-dir', required=True)

    args = vars(parser.parse_args())
    main = args.pop('main')

    main(**args)


if __name__ == '__main__':
    cli()
