import codecs
import json
import os
import random
import pickle
import re
import shutil
import sys

import numpy as np
import tensorflow as tf

from neural_network.ner import build_ner, NamedEntityEarlyStopping
from neural_network.siamese_nn import build_siamese_nn


def main():
    if len(sys.argv) < 2:
        err_msg = 'The training file is not specified!'
        raise ValueError(err_msg)
    training_fname = os.path.normpath(sys.argv[1])
    if len(sys.argv) < 3:
        err_msg = 'The validation file is not specified!'
        raise ValueError(err_msg)
    validation_fname = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The path to the trained model is not specified!'
        raise ValueError(err_msg)
    trained_model_path = os.path.normpath(sys.argv[3])
    if len(sys.argv) < 5:
        err_msg = 'The training mode is not specified!'
        raise ValueError(err_msg)
    training_mode = sys.argv[4].strip().lower()
    if len(training_mode) == 0:
        err_msg = 'The training mode is not specified!'
        raise ValueError(err_msg)
    if training_mode not in {'siamese', 'ner'}:
        err_msg = f'The training mode {training_mode} is unknown! ' \
                  f'Possible values: siamese, ner.'
        raise ValueError(err_msg)
    if len(sys.argv) < 6:
        err_msg = 'The mini-batch size is not specified!'
        raise ValueError(err_msg)
    try:
        minibatch_size = int(sys.argv[5].strip())
    except:
        minibatch_size = 0
    if minibatch_size < 1:
        err_msg = f'{sys.argv[5]} is a wrong value of the mini-batch size!'
        raise ValueError(err_msg)
    if len(sys.argv) < 7:
        err_msg = 'The pre-trained BERT model is not specified!'
        raise ValueError(err_msg)
    pretrained_model = sys.argv[6]
    if len(sys.argv) < 8:
        err_msg = 'The pre-trained model framework (PyTorch or Tensorflow) ' \
                  'is not specified!'
        raise ValueError(err_msg)
    re_for_splitting = re.compile(r'[-_\s]+')
    pretrained_framework = ''.join(re_for_splitting.split(sys.argv[7].lower()))
    if pretrained_framework not in {'frompytorch', 'fromtensorflow'}:
        err_msg = f'{sys.argv[7]} is unknown pre-trained model framework! ' \
                  f'Possible values: from-pytorch, from-tensorflow.'
        raise ValueError(err_msg)
    from_pytorch = (pretrained_framework == 'frompytorch')
    if training_mode == 'ner':
        if len(sys.argv) < 9:
            err_msg = 'The NER vocabulary file is not specified!'
            raise ValueError(err_msg)
        ners_fname = os.path.normpath(sys.argv[8])
        if len(sys.argv) >= 10:
            random_seed_ = sys.argv[9]
        else:
            random_seed_ = ''
    else:
        ners_fname = None
        if len(sys.argv) >= 9:
            random_seed_ = sys.argv[8]
        else:
            random_seed_ = ''
    if len(random_seed_) == 0:
        random_seed = 42
    else:
        try:
            random_seed = int(random_seed_)
        except:
            random_seed = -1
        if random_seed < 0:
            err_msg = f'Random seed = {random_seed_} is wrong!'
            raise ValueError(err_msg)

    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    if not os.path.isfile(training_fname):
        raise ValueError(f'The file "{training_fname}" does not exist!')
    if not os.path.isfile(validation_fname):
        raise ValueError(f'The file "{validation_fname}" does not exist!')
    if ners_fname is None:
        ne_list = []
    else:
        if not os.path.isfile(ners_fname):
            raise ValueError(f'The file "{ners_fname}" does not exist!')
        with codecs.open(ners_fname, mode='r', encoding='utf-8') as fp:
            ne_list = list(filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.strip(),
                    fp.readlines()
                )
            ))
        if len(ne_list) == 0:
            err_msg = f'The file {ners_fname} is empty!'
            raise ValueError(err_msg)
    if not os.path.isdir(trained_model_path):
        dirname = os.path.dirname(trained_model_path)
        if len(dirname) > 0:
            if not os.path.isdir(dirname):
                raise ValueError(f'The directory "{dirname}" does not exist!')
        os.mkdir(trained_model_path)

    with open(training_fname, 'rb') as fp:
        training_data = pickle.load(fp)
    print('')
    print(f'training_data[0].shape = {training_data[0].shape}')
    for idx in range(len(training_data[1])):
        print(f'training_data[1][{idx}].shape = {training_data[1][idx].shape}')
    print('')

    with open(validation_fname, 'rb') as fp:
        validation_data = pickle.load(fp)
    print(f'validation_data[0].shape = {validation_data[0].shape}')
    for idx in range(len(validation_data[1])):
        print(f'validation_data[1][{idx}].shape = '
              f'{validation_data[1][idx].shape}')
    print('')

    if training_mode == 'ner':
        trained_model, base_transformer = build_ner(
            bert_name=pretrained_model,
            from_pytorch=from_pytorch,
            max_seq_len=training_data[0].shape[1],
            named_entities=ne_list,
            learning_rate=1e-5,
            base_name=f'RuNNE_ner_seed{random_seed}'
        )
        model_name = os.path.join(trained_model_path, 'ner.h5')
        config_name = os.path.join(trained_model_path, 'ner.json')
        log_name = os.path.join(trained_model_path, 'ner_training_logs')
        with codecs.open(config_name, mode='w', encoding='utf-8') as fp:
            json.dump(
                obj={
                    'named_entities': ne_list,
                    'bert': pretrained_model,
                    'max_sent_len': training_data[0].shape[1],
                    'base_name': f'RuNNE_ner_seed{random_seed}'
                },
                fp=fp,
                ensure_ascii=False,
                indent=4
            )
        callbacks = [
            NamedEntityEarlyStopping(verbose=True, patience=5)
        ]
    else:
        trained_model, base_transformer = build_siamese_nn(
            bert_name=pretrained_model,
            from_pytorch=from_pytorch,
            max_seq_len=training_data[0].shape[1],
            learning_rate=1e-6,
            base_name=f'RuNNE_siamese_seed{random_seed}'
        )
        model_name = os.path.join(trained_model_path, 'siamese_nn.h5')
        log_name = os.path.join(trained_model_path, 'siamese_training_logs')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, verbose=True
            )
        ]
    trained_model.summary()

    if os.path.isdir(log_name):
        shutil.rmtree(log_name)
    os.mkdir(log_name)
    callbacks += [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_name,
            save_weights_only=True,
            save_best_only=True,
            verbose=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_name)
    ]
    training_set = tf.data.Dataset.from_tensor_slices(
        (
            training_data[0],
            tuple(training_data[1])
        )
    ).shuffle(training_data[0].shape[0]).batch(minibatch_size)
    validation_set = tf.data.Dataset.from_tensor_slices(
        (
            validation_data[0],
            tuple(validation_data[1])
        )
    ).batch(minibatch_size)
    trained_model.fit(training_set, validation_data=validation_set,
                      epochs=1000, callbacks=callbacks, verbose=2)
    trained_model.save_weights(model_name)
    if training_mode == 'siamese':
        os.remove(model_name)
        base_transformer.save_pretrained(trained_model_path)


if __name__ == '__main__':
    main()
