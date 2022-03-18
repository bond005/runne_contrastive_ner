import re
from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertConfig, TFBertModel

from neural_network.utils import AttentionMaskLayer


def distance_based_probability(vects):
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.keras.backend.square(x - y),
                                      axis=1, keepdims=True)
    dist = tf.keras.backend.sqrt(
        tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon())
    )
    margin = 1.0
    p = (1.0 + tf.math.exp(-margin)) / (1.0 + tf.math.exp(dist - margin))
    return p


def distance_based_probability_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def build_siamese_nn(bert_name: str, from_pytorch: bool, max_seq_len: int,
                     learning_rate: float,
                     base_name: str) -> Tuple[tf.keras.Model, TFBertModel]:
    re_for_name = re.compile(f'^[a-zA-Z]+[_a-zA-Z0-9]*[a-zA-Z0-9]+$')
    if re_for_name.search(base_name) is None:
        err_msg = f'{base_name} is a wrong name for ' \
                  f'Tensorflow models and layers!'
        raise ValueError(err_msg)
    bert_config = BertConfig.from_pretrained(bert_name)
    if bert_config.model_type != 'bert':
        err_msg = f'The transformer  {bert_name} is wrong ' \
                  f'because it is not BERT.'
        raise ValueError(err_msg)
    max_position_embeddings = bert_config.max_position_embeddings
    if max_seq_len > max_position_embeddings:
        err_msg = f'max_seq_len = {max_seq_len} is too large! ' \
                  f'It must be less then {max_position_embeddings + 1}.'
        raise ValueError(err_msg)
    left_word_ids = tf.keras.layers.Input(
        shape=(max_seq_len,),
        dtype=tf.int32,
        name=f'left_word_ids_{base_name}'
    )
    left_output_mask = tf.keras.layers.Input(
        shape=(max_seq_len,),
        dtype=tf.int32,
        name=f'left_output_mask_{base_name}'
    )
    left_attention_mask = AttentionMaskLayer(
        pad_token_id=bert_config.pad_token_id,
        name=f'left_attention_mask_{base_name}',
        trainable=False
    )(left_word_ids)
    right_word_ids = tf.keras.layers.Input(
        shape=(max_seq_len,),
        dtype=tf.int32,
        name=f'right_word_ids_{base_name}'
    )
    right_attention_mask = AttentionMaskLayer(
        pad_token_id=bert_config.pad_token_id,
        name=f'right_attention_mask_{base_name}',
        trainable=False
    )(right_word_ids)
    right_output_mask = tf.keras.layers.Input(
        shape=(max_seq_len,),
        dtype=tf.int32,
        name=f'right_output_mask_{base_name}'
    )
    transformer_layer = TFBertModel.from_pretrained(
        pretrained_model_name_or_path=bert_name,
        from_pt=from_pytorch, name=f'BertNLU_{base_name}'
    )
    left_sequence_output = transformer_layer(
        [left_word_ids, left_attention_mask]
    )[0]
    right_sequence_output = transformer_layer(
        [right_word_ids, right_attention_mask]
    )[0]
    left_pooled_output = tf.keras.layers.GlobalAveragePooling1D(
        name=f'LeftPooling_{base_name}'
    )(left_sequence_output, mask=left_output_mask)
    left_normalization = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name=f'LeftNorm_{base_name}'
    )(left_pooled_output)
    right_pooled_output = tf.keras.layers.GlobalAveragePooling1D(
        name=f'RightPooling_{base_name}'
    )(right_sequence_output, mask=right_output_mask)
    right_normalization = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name=f'RightNorm_{base_name}'
    )(right_pooled_output)
    distance_layer = tf.keras.layers.Lambda(
        function=distance_based_probability,
        output_shape=distance_based_probability_shape,
        name=f'ProbaDistLayer_{base_name}'
    )([left_normalization, right_normalization])
    siamese_nn = tf.keras.Model(
        inputs=[
            left_word_ids, left_output_mask,
            right_word_ids, right_output_mask
        ],
        outputs=distance_layer,
        name=f'SiameseBERT_{base_name}'
    )
    radam = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    siamese_nn.compile(
        optimizer=ranger,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.001),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return siamese_nn, transformer_layer
