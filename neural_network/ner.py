import codecs
import json
import os
import re
from typing import List, Tuple

import numpy as np
from scipy.stats import hmean
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import BertConfig, TFBertModel, BertTokenizer

from neural_network.utils import AttentionMaskLayer, MaskCalculator
from neural_network.utils import generate_random_seed
from neural_network.utils import NamedEntityLoss
from neural_network.utils import NamedEntityPrecision, NamedEntityRecall


class NamedEntityEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience: int = 0, verbose: bool = False):
        super(NamedEntityEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_metric = -np.Inf
        self.best_loss = np.Inf
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = -np.Inf
        self.best_loss = np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise ValueError('There are no logged metrics!')
        if 'loss' not in logs:
            err_msg = f'The united loss is not found in the {list(logs.keys)}!'
            raise ValueError(err_msg)
        training_loss = logs['loss']
        validation_loss = np.Inf
        training_score, validation_score = self.get_monitor_value(logs)
        if self.verbose:
            print('')
            info_msg = 'Training loss is {0:.6f}'.format(training_loss)
            print(info_msg)
            info_msg = 'Training F1-hmean is {0:.5f}'.format(training_score)
            print(info_msg)
            if validation_score is not None:
                if 'val_loss' not in logs:
                    err_msg = f'The united validation loss is not found ' \
                              f'in the {list(logs.keys)}!'
                    raise ValueError(err_msg)
                validation_loss = logs['val_loss']
                info_msg = 'Validation loss is {0:.6f}'.format(validation_loss)
                print(info_msg)
                info_msg = 'Validation F1-hmean is {0:.5f}'.format(
                    validation_score)
                print(info_msg)
        if validation_score is None:
            return

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if validation_score > self.best_metric:
            if self.verbose:
                info_msg = f'Validation F1 = {validation_score} is better ' \
                           f'than previous F1 value = {self.best_metric}. ' \
                           f'The best weights are updated.'
                print(info_msg)
            self.best_metric = validation_score
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
        elif abs(validation_score - self.best_metric) < 1e-2:
            if validation_loss < self.best_loss:
                if self.verbose:
                    info_msg = f'Validation loss = {validation_loss} is ' \
                               f'better than previous loss value = ' \
                               f'{self.best_loss}. The best weights are ' \
                               f'updated.'
                    print(info_msg)
                self.best_epoch = epoch
                self.best_weights = self.model.get_weights()
                self.best_loss = validation_loss
                self.wait = 0
        if validation_score > self.best_metric:
            self.best_metric = validation_score
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss

        if (self.wait >= self.patience) and (epoch > 0):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.best_weights is not None:
                if self.verbose:
                    info_msg = f'Restoring model weights from the end of ' \
                               f'the best epoch: {self.best_epoch + 1}.'
                    print(info_msg)
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if (self.stopped_epoch > 0) and self.verbose:
            print(f'Epoch {self.stopped_epoch + 1}: early stopping')

    @staticmethod
    def get_monitor_value(logs):
        logs = logs or {}
        monitored = dict()
        val_monitored = dict()
        all_possible_metrics = sorted(list(logs.keys()))
        for cur_metric_name in all_possible_metrics:
            value = logs[cur_metric_name]
            if cur_metric_name.startswith('val_'):
                if cur_metric_name.endswith('_named_entity_precision'):
                    base_metric_name = cur_metric_name[4:-23]
                    err_msg = f'Metric {cur_metric_name} is wrong! ' \
                              f'All possible metrics: {all_possible_metrics}'
                    if base_metric_name.strip() != base_metric_name:
                        raise ValueError(err_msg)
                    if len(base_metric_name) == 0:
                        raise ValueError(err_msg)
                    if base_metric_name in val_monitored:
                        if 'precision' in val_monitored[base_metric_name]:
                            err_msg = f'Metric {cur_metric_name} is ' \
                                      f'duplicated! Processed metrics: ' \
                                      f'{sorted(list(val_monitored.keys()))}.' \
                                      f' All possible metrics: ' \
                                      f'{all_possible_metrics}'
                            raise ValueError(err_msg)
                        val_monitored[base_metric_name]['precision'] = value
                    else:
                        val_monitored[base_metric_name] = {'precision': value}
                elif cur_metric_name.endswith('_named_entity_recall'):
                    base_metric_name = cur_metric_name[4:-20]
                    err_msg = f'Metric {cur_metric_name} is wrong! ' \
                              f'All possible metrics: {all_possible_metrics}'
                    if base_metric_name.strip() != base_metric_name:
                        raise ValueError(err_msg)
                    if len(base_metric_name) == 0:
                        raise ValueError(err_msg)
                    if base_metric_name in val_monitored:
                        if 'recall' in val_monitored[base_metric_name]:
                            err_msg = f'Metric {cur_metric_name} is ' \
                                      f'duplicated! Processed metrics: ' \
                                      f'{sorted(list(val_monitored.keys()))}.' \
                                      f' All possible metrics: ' \
                                      f'{all_possible_metrics}'
                            raise ValueError(err_msg)
                        val_monitored[base_metric_name]['recall'] = value
                    else:
                        val_monitored[base_metric_name] = {'recall': value}
            else:
                if cur_metric_name.endswith('_named_entity_precision'):
                    base_metric_name = cur_metric_name[:-23]
                    err_msg = f'Metric {cur_metric_name} is wrong! ' \
                              f'All possible metrics: {all_possible_metrics}'
                    if base_metric_name.strip() != base_metric_name:
                        raise ValueError(err_msg)
                    if len(base_metric_name) == 0:
                        raise ValueError(err_msg)
                    if base_metric_name in monitored:
                        if 'precision' in monitored[base_metric_name]:
                            err_msg = f'Metric {cur_metric_name} is ' \
                                      f'duplicated! Processed metrics: ' \
                                      f'{sorted(list(monitored.keys()))}. ' \
                                      f'All possible metrics: ' \
                                      f'{all_possible_metrics}'
                            raise ValueError(err_msg)
                        monitored[base_metric_name]['precision'] = value
                    else:
                        monitored[base_metric_name] = {'precision': value}
                elif cur_metric_name.endswith('_named_entity_recall'):
                    base_metric_name = cur_metric_name[:-20]
                    err_msg = f'Metric {cur_metric_name} is wrong! ' \
                              f'All possible metrics: {all_possible_metrics}'
                    if base_metric_name.strip() != base_metric_name:
                        raise ValueError(err_msg)
                    if len(base_metric_name) == 0:
                        raise ValueError(err_msg)
                    if base_metric_name in monitored:
                        if 'recall' in monitored[base_metric_name]:
                            err_msg = f'Metric {cur_metric_name} is ' \
                                      f'duplicated! Processed metrics: ' \
                                      f'{sorted(list(monitored.keys()))}. ' \
                                      f'All possible metrics: ' \
                                      f'{all_possible_metrics}'
                            raise ValueError(err_msg)
                        monitored[base_metric_name]['recall'] = value
                    else:
                        monitored[base_metric_name] = {'recall': value}
        if len(monitored) == 0:
            err_msg = f'Logs do not contain any monitored metric! ' \
                      f'All possible metrics: {all_possible_metrics}'
            raise ValueError(err_msg)
        eps = 1e-3
        f1_list = []
        for monitored_metric in monitored:
            if 'precision' not in monitored[monitored_metric]:
                err_msg = f'The precision does not exist ' \
                          f'for the {monitored_metric} output.'
                raise ValueError(err_msg)
            if 'recall' not in monitored[monitored_metric]:
                err_msg = f'The recall does not exist ' \
                          f'for the {monitored_metric} output.'
                raise ValueError(err_msg)
            precision = monitored[monitored_metric]['precision']
            recall = monitored[monitored_metric]['recall']
            if abs(precision + recall) > 1e-6:
                f1_cur = 2.0 * (precision * recall) / (precision + recall)
            else:
                f1_cur = 0.0
            f1_list.append(f1_cur + eps)
        f1_hmean = hmean(f1_list)
        if len(val_monitored) == 0:
            return f1_hmean, None
        if set(val_monitored.keys()) != set(monitored.keys()):
            err_msg = 'The training metrics do not correspond to ' \
                      'the validation metrics!'
            raise ValueError(err_msg)
        val_f1_list = []
        for monitored_metric in val_monitored:
            if 'precision' not in val_monitored[monitored_metric]:
                err_msg = f'The precision does not exist ' \
                          f'for the {monitored_metric} output.'
                raise ValueError(err_msg)
            if 'recall' not in val_monitored[monitored_metric]:
                err_msg = f'The recall does not exist ' \
                          f'for the {monitored_metric} output.'
                raise ValueError(err_msg)
            precision = val_monitored[monitored_metric]['precision']
            recall = val_monitored[monitored_metric]['recall']
            if abs(precision + recall) > 1e-6:
                f1_cur = 2.0 * (precision * recall) / (precision + recall)
            else:
                f1_cur = 0.0
            val_f1_list.append(f1_cur + eps)
        val_f1_hmean = hmean(val_f1_list)
        return f1_hmean, val_f1_hmean


def get_nn_output_name(target_name: str) -> str:
    return target_name.title().replace('-', '').replace(':', '')


def build_ner(bert_name: str, from_pytorch: bool, max_seq_len: int,
              named_entities: List[str], learning_rate: float,
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
                  f'It must be less than {max_position_embeddings + 1}.'
        raise ValueError(err_msg)
    output_embedding_size = bert_config.hidden_size
    word_ids = tf.keras.layers.Input(
        shape=(max_seq_len,),
        dtype=tf.int32,
        name=f'base_word_ids_{base_name}'
    )
    attention_mask = AttentionMaskLayer(
        pad_token_id=bert_config.pad_token_id,
        name=f'base_attention_mask_{base_name}',
        trainable=False
    )(word_ids)
    transformer_layer = TFBertModel.from_pretrained(
        pretrained_model_name_or_path=bert_name,
        from_pt=from_pytorch, name=f'BertNLU_{base_name}'
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=output_embedding_size,
        pad_token_id=bert_config.pad_token_id,
        trainable=False,
        name=f'MaskCalculator_{base_name}'
    )(word_ids)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='MaskedOutput_'
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='MaskedEmdOutput', mask_value=0.0
    )(masked_sequence_output)
    outputs = []
    losses = []
    metrics = []
    for cur_entity in named_entities:
        new_layer_name = get_nn_output_name(cur_entity) + f'_{base_name}'
        new_dropout_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(
                rate=0.5,
                seed=generate_random_seed(),
                name=new_layer_name + '_dropout_'
            ),
            name=new_layer_name + '_dropout'
        )(masked_sequence_output)
        new_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=5,
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_normal(
                    seed=generate_random_seed()
                ),
                bias_initializer=tf.keras.initializers.zeros(),
                name=new_layer_name + '_'
            ),
            name=new_layer_name
        )(new_dropout_layer)
        outputs.append(new_output_layer)
        losses.append(
            (
                new_layer_name,
                NamedEntityLoss(name=f'{new_layer_name}_loss')
            )
        )
        metrics.append(
            (
                new_layer_name,
                [
                    tf.keras.metrics.CategoricalAccuracy(
                        name=f'{new_layer_name}_acc'
                    ),
                    NamedEntityPrecision(
                        name=f'{new_layer_name}_named_entity_precision'
                    ),
                    NamedEntityRecall(
                        name=f'{new_layer_name}_named_entity_recall'
                    )
                ]
            )
        )
    ner_model = tf.keras.Model(
        word_ids,
        outputs,
        name=f'NamedEntityRecognizer_{base_name}'
    )
    radam = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ner_model.compile(
        optimizer=ranger,
        loss=dict(losses),
        metrics=dict(metrics)
    )
    return ner_model, transformer_layer


def load_ner(path: str) -> Tuple[tf.keras.Model, BertTokenizer, int, List[str]]:
    if not os.path.isdir(path):
        err_msg = f'The directory "{path}" does not exist!'
        raise ValueError(err_msg)
    config_name = os.path.join(path, 'ner.json')
    if not os.path.isfile(config_name):
        err_msg = f'The file "{config_name}" does not exist!'
        raise ValueError(err_msg)
    weights_name = os.path.join(path, 'ner.h5')
    if not os.path.isfile(weights_name):
        err_msg = f'The file "{weights_name}" does not exist!'
        raise ValueError(err_msg)
    with codecs.open(config_name, mode='r', encoding='utf-8') as fp:
        config_data = json.load(fp)
    if not isinstance(config_data, dict):
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'Expected {{"a": 1}}, got {type(config_data)}.'
        raise ValueError(err_msg)
    if not 'named_entities' in config_data:
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "named_entities" key is not found.'
        raise ValueError(err_msg)
    if not 'max_sent_len' in config_data:
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "max_sent_len" key is not found.'
        raise ValueError(err_msg)
    if not 'bert' in config_data:
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "bert" key is not found.'
        raise ValueError(err_msg)
    if not 'base_name' in config_data:
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "base_name" key is not found.'
        raise ValueError(err_msg)
    if not isinstance(config_data['max_sent_len'], int):
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "max_sent_len" value is incorrect. ' \
                  f'Expected {type(1)}, ' \
                  f'got {type(config_data["max_sent_len"])}.'
        raise ValueError(err_msg)
    if config_data['max_sent_len'] < 4:
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "max_sent_len" = {config_data["max_sent_len"]} ' \
                  f'is too small. Expected value greater than 3.'
        raise ValueError(err_msg)
    if not isinstance(config_data['named_entities'], list):
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "named_entities" value is incorrect. ' \
                  f'Expected {type([1, 2])}, ' \
                  f'got {type(config_data["named_entities"])}'
        raise ValueError(err_msg)
    if not isinstance(config_data['bert'], str):
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "bert" value is incorrect. Expected {type("123")}, ' \
                  f'got {type(config_data["bert"])}'
        raise ValueError(err_msg)
    if not isinstance(config_data['base_name'], str):
        err_msg = f'The config data from "{config_name}" is wrong! ' \
                  f'The "base_name" value is incorrect. ' \
                  f'Expected {type("123")}, ' \
                  f'got {type(config_data["base_name"])}'
        raise ValueError(err_msg)
    re_for_name = re.compile(f'^[a-zA-Z]+[_a-zA-Z]*[a-zA-Z]+$')
    if re_for_name.search(config_data["base_name"]) is None:
        err_msg = f'{config_data["base_name"]} is a wrong name for ' \
                  f'Tensorflow models and layers!'
        raise ValueError(err_msg)
    base_name = config_data["base_name"]
    bert_name = config_data['bert']
    if os.path.isdir(os.path.join(path, os.path.normpath(bert_name))):
        bert_name = os.path.join(path, os.path.normpath(bert_name))
    bert_config = BertConfig.from_pretrained(bert_name)
    if bert_config.model_type != 'bert':
        err_msg = f'The transformer  {bert_name} is wrong ' \
                  f'because it is not BERT.'
        raise ValueError(err_msg)
    max_position_embeddings = bert_config.max_position_embeddings
    if config_data['max_sent_len'] > max_position_embeddings:
        err_msg = f'max_seq_len = {config_data["max_sent_len"]} is ' \
                  f'too large! It must be less than ' \
                  f'{max_position_embeddings + 1}.'
        raise ValueError(err_msg)
    output_embedding_size = bert_config.hidden_size
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    word_ids = tf.keras.layers.Input(
        shape=(config_data['max_sent_len'],),
        dtype=tf.int32,
        name=f'base_word_ids_{base_name}'
    )
    attention_mask = AttentionMaskLayer(
        pad_token_id=bert_config.pad_token_id,
        name=f'base_attention_mask_{base_name}',
        trainable=False
    )(word_ids)
    transformer_layer = TFBertModel(
        config=bert_config,
        name=f'BertNLU_{base_name}'
    )
    sequence_output = transformer_layer([word_ids, attention_mask])[0]
    output_mask = MaskCalculator(
        output_dim=output_embedding_size,
        pad_token_id=bert_config.pad_token_id,
        trainable=False,
        name=f'MaskCalculator_{base_name}'
    )(word_ids)
    masked_sequence_output = tf.keras.layers.Multiply(
        name='MaskedOutput_'
    )([output_mask, sequence_output])
    masked_sequence_output = tf.keras.layers.Masking(
        name='MaskedEmdOutput', mask_value=0.0
    )(masked_sequence_output)
    outputs = []
    for cur_entity in config_data['named_entities']:
        new_layer_name = get_nn_output_name(cur_entity) + f'_{base_name}'
        new_dropout_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dropout(
                rate=0.5,
                seed=generate_random_seed(),
                name=new_layer_name + '_dropout_'
            ),
            name=new_layer_name + '_dropout'
        )(masked_sequence_output)
        new_output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                units=5,
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_normal(
                    seed=generate_random_seed()
                ),
                bias_initializer=tf.keras.initializers.zeros(),
                name=new_layer_name + '_'
            ),
            name=new_layer_name
        )(new_dropout_layer)
        outputs.append(new_output_layer)
    ner_model = tf.keras.Model(
        word_ids,
        outputs,
        name=f'NamedEntityRecognizer_{base_name}'
    )
    ner_model.build(input_shape=(None, config_data['max_sent_len']))
    ner_model.load_weights(weights_name)
    return ner_model, tokenizer, config_data['max_sent_len'], \
           config_data['named_entities']
