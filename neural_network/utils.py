import random

import six
import tensorflow as tf
from tensorflow.python.framework import ops, tensor_util
from tensorflow.python.keras.utils import losses_utils, tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_util


class MaskCalculator(tf.keras.layers.Layer):
    def __init__(self, output_dim: int, pad_token_id: int, **kwargs):
        self.output_dim = output_dim
        self.pad_token_id = pad_token_id
        super(MaskCalculator, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MaskCalculator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.permute_dimensions(
            x=tf.keras.backend.repeat(
                x=tf.keras.backend.cast(
                    x=tf.math.not_equal(
                        x=inputs,
                        y=self.pad_token_id
                    ),
                    dtype='float32'
                ),
                n=self.output_dim
            ),
            pattern=(0, 2, 1)
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        shape = list(input_shape)
        shape.append(self.output_dim)
        return tuple(shape)

    def get_config(self):
        return {
            "output_dim": self.output_dim,
            "pad_token_id": self.pad_token_id
        }


class AttentionMaskLayer(tf.keras.layers.Layer):
    def __init__(self, pad_token_id: int, **kwargs):
        self.pad_token_id = pad_token_id
        super(AttentionMaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionMaskLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.cast(
            x=tf.math.not_equal(
                x=inputs,
                y=self.pad_token_id
            ),
            dtype='float32'
        )

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 1
        return input_shape

    def get_config(self):
        return {"pad_token_id": self.pad_token_id}


class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
                 fn,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name=None,
                 **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction,
                                                  name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
                y_pred, y_true
            )
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = tf.keras.backend.eval(v) if \
                tf_utils.is_tensor_or_variable(v) \
                else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dice_loss(y_true, y_pred):
    gamma = 1e-3
    numerator = 2 * tf.reduce_sum(y_true * y_true * y_pred * y_pred) + gamma
    denominator = tf.reduce_sum(y_true * y_true + y_pred * y_pred) + gamma
    return 1.0 - numerator / denominator


def named_entity_loss(y_true, y_pred):
    y_pred_ = ops.convert_to_tensor(y_pred)
    y_true_ = math_ops.cast(y_true, y_pred_.dtype)
    if len(y_pred_.shape) != 3:
        err_msg = f'len(y_pred.shape) is wrong! Expected 3, ' \
                  f'got {len(y_pred_.shape)}.'
        raise ValueError(err_msg)
    if y_pred_.shape[2] != 5:
        err_msg = f'y_pred.shape[2] is wrong! Expected 5, ' \
                  f'got {y_pred_.shape[2]}.'
        raise ValueError(err_msg)
    if len(y_true_.shape) != 3:
        err_msg = f'len(y_true.shape) is wrong! Expected 3, ' \
                  f'got {len(y_true_.shape)}.'
        raise ValueError(err_msg)
    if y_true_.shape[2] != 5:
        err_msg = f'y_true.shape[2] is wrong! Expected 5, ' \
                  f'got {y_true_.shape[2]}.'
        raise ValueError(err_msg)
    y_pred_ = tf.nn.softmax(y_pred_, axis=-1)
    loss_val = dice_loss(1.0 - y_true_[:, :, 0:1], 1.0 - y_pred_[:, :, 0:1])
    loss_val += 0.1 * tf.keras.losses.categorical_crossentropy(
        y_true_, y_pred_,
        from_logits=False, label_smoothing=0.01
    )
    return loss_val


class NamedEntityLoss(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
                 name='named_entity_loss'):
        super(NamedEntityLoss, self).__init__(named_entity_loss, name=name,
                                              reduction=reduction)


class NamedEntityPrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ = ops.convert_to_tensor(y_pred)
        y_true_ = math_ops.cast(y_true, y_pred_.dtype)
        if len(y_pred_.shape) != 3:
            err_msg = f'len(y_pred.shape) is wrong! Expected 3, ' \
                      f'got {len(y_pred_.shape)}.'
            raise ValueError(err_msg)
        if len(y_true_.shape) != 3:
            err_msg = f'len(y_true.shape) is wrong! Expected 3, ' \
                      f'got {len(y_true_.shape)}.'
            raise ValueError(err_msg)
        y_pred_ = tf.argmax(y_pred_, axis=-1)
        y_pred_ = tf.math.not_equal(y_pred_, tf.zeros_like(y_pred_))
        y_pred_ = math_ops.cast(y_pred_, y_true_.dtype)
        y_true_ = 1.0 - y_true_[:, :, 0]
        super(NamedEntityPrecision, self).update_state(y_true_, y_pred_,
                                                       sample_weight)


class NamedEntityRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ = ops.convert_to_tensor(y_pred)
        y_true_ = math_ops.cast(y_true, y_pred_.dtype)
        if len(y_pred_.shape) != 3:
            err_msg = f'len(y_pred.shape) is wrong! Expected 3, ' \
                      f'got {len(y_pred_.shape)}.'
            raise ValueError(err_msg)
        if len(y_true_.shape) != 3:
            err_msg = f'len(y_true.shape) is wrong! Expected 3, ' \
                      f'got {len(y_true_.shape)}.'
            raise ValueError(err_msg)
        y_pred_ = tf.argmax(y_pred_, axis=-1)
        y_pred_ = tf.math.not_equal(y_pred_, tf.zeros_like(y_pred_))
        y_pred_ = math_ops.cast(y_pred_, y_true_.dtype)
        y_true_ = 1.0 - y_true_[:, :, 0]
        super(NamedEntityRecall, self).update_state(y_true_, y_pred_,
                                                    sample_weight)


def generate_random_seed() -> int:
    return random.randint(0, 2147483646)
