#!/usr/bin/env python3.8
"""

"""

from dataclasses import dataclass
from pathlib import Path
import re
from time import sleep
from typing import Tuple, Optional

import horovod.tensorflow.keras as hvd
from horovod.tensorflow import join as hvd_join, _executing_eagerly
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


def create_no_op_optimizer(optimizer):
    class _NoOpOptimizer(tf.keras.optimizers.Optimizer):
        def __init__(self, **kwargs):
            print(f'{self.__class__ = }, {super(self.__class__, self) = }, {super(self.__class__, self).__class__ = }')
            old_class = self.__class__
            self.__class__ = self.__class__.__base__
            self.__init__(**kwargs)
            #super(_NoOpOptimizer, self).__init__(**kwargs)
            self.__class__ = old_class

        def _aggregate_gradients(self, grads_and_vars):
            print(f'_aggregate_gradients called')
            grads, vars = grads_and_vars
            grads_and_vars = [(tf.zeros(shape=v.shape), v) for v in vars]
            return super(self.__class__, self)._aggregate_gradients(grads_and_vars)

        #def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
        #    print(f'_compute_gradients called')
        #    return [(tf.zeros(shape=v.shape), v) for v in var_list]
    
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_NoOpOptimizer.__dict__))
    
    config = optimizer.get_config()
    print(f'{config = }')
    return cls.from_config(config)


def main(events, div, num_conv_layers, dataset, default_verbosity, data_dir, checkpoint_dir):
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
    size = world.Get_size()

    hvd.init(world)
    assert hvd.mpi_threads_supported()
    print(f'{_executing_eagerly() = }')

    #print(f'{hvd.

    datasets, info = tfds.load(
        dataset,
        split=None,
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
        download=True,
    )

    train_ds = datasets['train']
    test_ds = datasets['test']

    train_ds = train_ds.map(lambda img, label: (tf.image.convert_image_dtype(img, dtype=tf.float32), label))
    test_ds = test_ds.map(lambda img, label: (tf.image.convert_image_dtype(img, dtype=tf.float32), label))

    model = make_simple_cnn_model(
        input_shape=info.features['image'].shape,
        output_shape=info.features['label'].num_classes,
        num_conv_layers=num_conv_layers,
    )

    def NoOpOptimizer():
        opt = tf.keras.optimizers.Adam(0.001)
        def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
            print(f'_compute_gradients called {rank=}')
            return [(tf.zeros(shape=v.shape), v) for v in var_list]
        opt._compute_gradients = _compute_gradients
        return opt

    class PreciseEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self, nepochs, nbatches):
            self.nepochs = nepochs
            self.nbatches = nbatches
            self.epoch = None
            self.batch = None

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            self._check_condition_and_stop()

        def on_train_batch_begin(self, batch, logs=None):
            self.batch = batch
            self._check_condition_and_stop()
            print(f'{self.epoch = }, {self.batch = }')

        def _check_condition_and_stop(self):
            on_epoch = self.epoch == self.nepochs - 1
            on_batch = self.batch == self.nbatches - 1
            if on_epoch and on_batch:
                self.model.stop_training = True
                

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        PreciseEarlyStopping(nepochs=3, nbatches=13),
    ]

    if rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir / 'checkpoint.h5', save_weights_only=True))

    events.insert(0, Event(nepochs=2, nworkers=size, batch=32, reload=False))

    initial_epoch = 0
    for event in events:
        opt = tf.keras.optimizers.Adam(0.001)
        print(f'{rank=} {opt.__class__ = }, {opt.__class__.__base__ = }')

        opt = hvd.DistributedOptimizer(
            opt,
            name='DistributedAdam',
            backward_passes_per_step=1,
            average_aggregated_gradients=True,
        )
        print(f'{rank=} {opt.__class__ = }, {opt.__class__.__base__ = }')

        if rank == 1:
            opt = create_no_op_optimizer(opt)
            print(f'{rank=} {opt.__class__ = }, {opt.__class__.__base__ = }')

       # old_allreduce = opt._allreduce
       # def _allreduce(grads):
       #     print(f'{rank=} {grads = }')
       #     return old_allreduce(grads)
       # opt._allreduce = _allreduce

        model.compile(
            optimizer=opt,
            metrics=['accuracy'],
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            experimental_run_tf_function=False,
        )

        if event.reload:
            print(f'Reloading weights')
            #weights = tf.train.latest_checkpoint(checkpoint_dir)
            weights = checkpoint_dir / 'checkpoint.h5'
            if weights is None:
                print(f'Error! Could not load weights!')
                print(f'{checkpoint_dir = }')
                for path in checkpoint_dir.iterdir():
                    print(f'  {path = }')
                raise ValueError('Could not load weights')
            model.load_weights(weights)

        history = model.fit(
            train_ds.repeat().batch(event.batch),
            steps_per_epoch=info.splits['train'].num_examples // event.batch // event.nworkers // div,
            callbacks=callbacks,
            epochs=initial_epoch + event.nepochs,
            initial_epoch=initial_epoch,
            verbose=default_verbosity if hvd.rank() == 0 else 0,
        )

        stats = model.evaluate(
            test_ds.repeat().batch(event.batch),
            steps=info.splits['test'].num_examples // event.batch // event.nworkers // div,
            callbacks=callbacks,
            verbose=default_verbosity if hvd.rank() == 0 else 0,
        )
        if rank == 0:
            print(f'stats = {" ".join(f"{name}={value}" for name, value in zip(model.metrics_names, stats))}')

        initial_epoch += event.nepochs


@dataclass
class Event:
    nepochs: int
    nworkers: int
    batch: int
    reload: bool
 
    @classmethod
    def parse(cls, s):
        # 0e/nworkers=12
        # 12e/nworkers=6
        match = re.match(r'^(?P<nepochs>[0-9]+)e/(?P<options>[a-z]+=[a-z0-9A-Z]+(?:,[a-z]+=[a-z0-9A-Z]+)*)', s)
        nepochs = int(match.group('nepochs'))
        options = match.group('options')
        options = dict((k, v) for x in options.split(',') for k, v in (x.split('=', 1),))
        nworkers = options.get('nworkers', None)
        if nworkers is not None:
            nworkers = int(nworkers)
        batch = int(options.get('batch', 32))
        reload = bool(options.get('reload', False))

        return cls(nepochs, nworkers, batch, reload)


def cli():
    def event(s):
        try:
            return Event.parse(s)
        except Exception as e:
            print(e)
            raise argparse.ArgumentError() from e

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('events', nargs='+', type=event)
    parser.add_argument('--div', required=True, type=int)
    parser.add_argument('--num-conv-layers', required=True, type=int)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--default-verbosity', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    args = vars(parser.parse_args())

    main(**args)


if __name__ == '__main__':
    cli()
