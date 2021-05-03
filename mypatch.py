from monkeypatch import monkeypatch
import whatreallyhappened as wrh


import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow.gradient_aggregation import LocalGradientAggregationHelper
from horovod.tensorflow.gradient_aggregation_eager import LocalGradientAggregationHelperEager
from horovod.tensorflow.mpi_ops import rank

from horovod._keras import _PRE_TF_2_4_0


_hvd_size = 1
_abort_after_layer = None
_abort_after_gradient = None
_stop_after_layer = None
_stop_after_gradient = None


def set_params(**kwargs):
    if kwargs['hvd_size'] is not None:
        global _hvd_size
        _hvd_size = kwargs['hvd_size']

    if kwargs['abort_after_layer'] is not None:
        global _abort_after_layer
        _abort_after_layer = kwargs['abort_after_layer']

    if kwargs['abort_after_gradient'] is not None:
        global _abort_after_gradient
        _abort_after_gradient = kwargs['abort_after_gradient']

    if kwargs['stop_after_layer'] is not None:
        global _stop_after_layer
        _stop_after_layer = kwargs['stop_after_layer']

    if kwargs['stop_after_gradient'] is not None:
        global _stop_after_gradient
        _stop_after_gradient = kwargs['stop_after_gradient']



@monkeypatch('horovod._keras.create_distributed_optimizer')
def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense, gradient_predivide_factor,
                                 op, backward_passes_per_step=1,
                                 average_aggregated_gradients=False,
                                 groups=None, *, create_distributed_optimizer):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        _HAS_AGGREGATE_GRAD = True

        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._aggregated_gradients = False

            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                self._name,
                device_dense,
                device_sparse,
                compression,
                sparse_as_dense,
                op,
                gradient_predivide_factor,
                groups)

            self._agg_helper = None
            if backward_passes_per_step > 1:
                if hvd._executing_eagerly():
                    self._agg_helper = LocalGradientAggregationHelperEager(
                        backward_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients=average_aggregated_gradients,
                    )
                else:
                    self._agg_helper = LocalGradientAggregationHelper(
                        backward_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients=average_aggregated_gradients,
                        rank=rank(),
                        optimizer_type=LocalGradientAggregationHelper._OPTIMIZER_TYPE_KERAS,
                    )

            super(self.__class__, self).__init__(**kwargs)

        def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            wrh.push('create_distributed_optimizer._DistributedOptimizer._compute_gradients')
            wrh.log(f'loss', '%r', loss)
            for i, x in enumerate(var_list):
                wrh.log(f'var_list[{i}]', '%r', x)
            wrh.log('_PRE_TF_2_4_0', '%r', _PRE_TF_2_4_0)
            if _PRE_TF_2_4_0:
                ret = super(self.__class__, self)._compute_gradients(
                    loss, var_list, grad_loss, tape)
                for i, (x, y) in enumerate(ret):
                    wrh.log(f'allreduced_grads[{i}]', '%r', x)
                    wrh.log(f'weights[{i}]', '%r', y)
                wrh.log('ret', '%r', ret)
                wrh.pop('create_distributed_optimizer._DistributedOptimizer._compute_gradients')
                return ret

            tape = backprop.GradientTape() if tape is None else tape
            grads_and_vars = super(self.__class__, self)._compute_gradients(
                # pylint: disable=protected-access
                loss,
                var_list,
                grad_loss,
                tape=tape)
            grads, weights = list(zip(*grads_and_vars))

            gradient_counter = 0
            def make_wrapper(tensor):
                layer_counter = 0
                def wrapper(inp):
                    nonlocal gradient_counter, layer_counter

                    if _stop_after_layer is not None or _abort_after_layer is not None:
                        if _stop_after_layer == layer_counter or _abort_after_layer == layer_counter:
                            if _stop_after_gradient is not None or _abort_after_gradient is not None:
                                gradient_counter += 1

                                if gradient_counter >= _stop_after_layer:
                                    return tf.zeros_like(inp)

                                elif gradient_counter >= _abort_after_layer:
                                    raise tf.errors.AbortedError(None, None, 'abort after layers')

                            else:
                                


                    _gradient_counter += 1
                    _layer_counter += 1
                    _counter += 1
                    if _counter > 2:
                        raise tf.errors.AbortedError(None, None, 'my personal error')
                    wrh.push('create_distributed_optimizer._DistributedOptimizer._compute_gradients.wrapper')
                    ret = tf.divide(inp, _hvd_size)
                    wrh.pop('create_distributed_optimizer._DistributedOptimizer._compute_gradients.wrapper')
                    return ret

                return tf.py_function(wrapper, (tensor,), tensor.dtype)
                
            grads = list(grads)
            for i, grad in enumerate(grads):
                wrh.log('grads[{i}]', '%r', grad)
                grads[i] = make_wrapper(grad)

            allreduced_grads = self._allreduce(grads, weights)
            ret = list(zip(allreduced_grads, weights))

            for i, (x, y) in enumerate(ret):
                wrh.log(f'allreduced_grads[{i}]', '%r', x)
                wrh.log(f'allreduced_grads[{i}].op', '%r', x.op)
                wrh.log(f'allreduced_grads[{i}].op.inputs', '%r', x.op.inputs)
                wrh.log(f'allreduced_grads[{i}].op.outputs', '%r', x.op.outputs)
                wrh.log(f'weights[{i}]', '%r', y)
            wrh.pop('create_distributed_optimizer._DistributedOptimizer._compute_gradients')
            return ret

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.
            See Optimizer.get_gradients() for more info.
            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            return self._allreduce(gradients, params)

        def _aggregate_gradients(self, grads_and_vars):
            if _PRE_TF_2_4_0:
                grads, vars = list(zip(*grads_and_vars))
                aggregated_grads = self._allreduce(grads, vars)
                return aggregated_grads
            else:
                return super(self.__class__, self)._aggregate_gradients(
                    grads_and_vars)

        def _allreduce(self, grads, vars):
            self._aggregated_gradients = True

            if self._agg_helper:
                return self._agg_helper.compute_gradients(tuple(grads), tuple(vars))
            else:
                return self._allreduce_grads(grads, vars)

        def apply_gradients(self, *args, **kwargs):
            if self._agg_helper:
                if isinstance(args[0], zip):
                    # If grad_and_vars are passed in as a zip object
                    # convert to a list. This is necessary for TF2.4+
                    # b/c args[0] is used in both conditional branches
                    # inside _agg_helper.apply_gradients().
                    args = list(args)
                    args[0] = list(args[0])
                    args = tuple(args)

                results = self._agg_helper.apply_gradients(
                    lambda: super(self.__class__, self).apply_gradients(*args, **kwargs),
                    self,
                    *args,
                    **kwargs,
                )
            else:
                results = super(self.__class__, self).apply_gradients(*args, **kwargs)

            if _PRE_TF_2_4_0 and not self._aggregated_gradients:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                                'using TensorFlow 2.0, please specify '
                                '`experimental_run_tf_function=False` in `compile()`.')

            return results

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    config = optimizer.get_config()
    if not _PRE_TF_2_4_0 and issubclass(optimizer.lr.__class__,
                                        keras.optimizers.schedules.LearningRateSchedule):
        lr_cls = type(optimizer.lr.__class__.__name__, (optimizer.lr.__class__,),
                      dict(optimizer.lr.__dict__))
        config['learning_rate'] = lr_cls.from_config(config['learning_rate']['config'])

    return cls.from_config(config)
