import codecs
import os
import random
import re
import pickle
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer, TFBertModel

from io_utils.io_utils import load_data
from data_processing.feature_extraction import calc_features
from data_processing.feature_extraction import calc_features_and_labels
from neural_network.ner import load_ner


def main():
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    if len(sys.argv) < 2:
        err_msg = 'The source file is not specified!'
        raise ValueError(err_msg)
    src_fname = os.path.normpath(sys.argv[1])
    if len(sys.argv) < 3:
        err_msg = 'The BERT model name is not specified!'
        raise ValueError(err_msg)
    bert_path = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The pre-trained model framework (PyTorch or Tensorflow) ' \
                  'is not specified!'
        raise ValueError(err_msg)
    re_for_splitting = re.compile(r'[-_\s]+')
    pretrained_framework = ''.join(re_for_splitting.split(sys.argv[3].lower()))
    if pretrained_framework not in {'frompytorch', 'fromtensorflow', 'fromner'}:
        err_msg = f'{sys.argv[3]} is unknown pre-trained model framework! ' \
                  f'Possible values: from-pytorch, from-tensorflow, from-ner ' \
                  f'(the last value determines the use of BERT model from ' \
                  f'the trained NER).'
        raise ValueError(err_msg)
    from_pytorch = (pretrained_framework == 'frompytorch')
    from_tensorflow = (pretrained_framework == 'fromtensorflow')
    if len(sys.argv) < 5:
        err_msg = 'The destination file with features is not specified!'
        raise ValueError(err_msg)
    dst_fname = os.path.normpath(sys.argv[4])
    if len(sys.argv) < 6:
        err_msg = 'The source data kind is not specified! ' \
                  'Possible values: text, annotation.'
        raise ValueError(err_msg)
    source_data_kind = sys.argv[5].strip().lower()
    if source_data_kind not in {'text', 'annotation'}:
        err_msg = f'{sys.argv[5]} is wrong source data kind!' \
                  f'Possible values: text, annotation.'
        raise ValueError(err_msg)
    if len(sys.argv) < 7:
        err_msg = 'The maximal sentence length is not specified!'
        raise ValueError(err_msg)
    try:
        max_len = int(sys.argv[6])
    except:
        max_len = 0
    if max_len <= 0:
        err_msg = f'The maximal sentence length = {sys.argv[6]} ' \
                  f'is inadmissible!'
        raise ValueError(err_msg)
    if source_data_kind == 'annotation':
        if len(sys.argv) < 8:
            err_msg = 'The named entity vocabulary is not specified!'
            raise ValueError(err_msg)
        ne_voc_fname = os.path.normpath(sys.argv[7])
        if not os.path.isfile(ne_voc_fname):
            err_msg = f'The file "{ne_voc_fname}" does not exist!'
            raise IOError(err_msg)
        with codecs.open(ne_voc_fname, mode='r', encoding='utf-8') as fp:
            named_entity_list = list(filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: it1.strip(), fp.readlines())
            ))
        if len(named_entity_list) < 1:
            raise ValueError(f'The file "{ne_voc_fname}" is empty!')
    else:
        named_entity_list = []

    if not os.path.isfile(src_fname):
        err_msg = f'The file "{src_fname}" does not exist!'
        raise IOError(err_msg)
    if len(dst_fname.strip()) == 0:
        raise ValueError('The destination file name is empty!')
    dst_dir = os.path.dirname(dst_fname)
    if len(dst_dir) > 0:
        if not os.path.isdir(dst_dir):
            err_msg = f'The directory "{dst_dir}" does not exist!'
            raise IOError(err_msg)

    if from_tensorflow:
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = TFBertModel.from_pretrained(bert_path)
    elif from_pytorch:
        bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
        bert_model = TFBertModel.from_pretrained(bert_path, from_pt=True)
    else:
        ner_model, bert_tokenizer, _, ne_list = load_ner(bert_path)
        if len(named_entity_list) > 0:
            if named_entity_list != ne_list:
                err_msg = f'The named entity list does not correspond ' \
                          f'to the model "{bert_path}"!'
                raise ValueError(err_msg)
        bert_layer = list(filter(lambda it: it.name.startswith('BertNLU_'),
                                 ner_model.layers))
        if len(bert_layer) != 1:
            err_msg = f'The model "{bert_path}" is wrong!'
            if len(bert_layer) > 1:
                err_msg += f' There are too many BERT layers: ' \
                           f'{[cur.name for cur in bert_layer]}!'
            else:
                err_msg += ' There are no BERT layers!'
            raise ValueError(err_msg)
        bert_model = bert_layer[0]

    features = []
    if source_data_kind == 'annotation':
        labels = [[] for _ in range(len(named_entity_list))]
        source_data = load_data(src_fname)
        for cur_id in tqdm(sorted(list(source_data.keys()))):
            text, ners = source_data[cur_id]
            X, y = calc_features_and_labels(
                bert_tokenizer,
                bert_model,
                max_len,
                named_entity_list,
                text, ners
            )
            features.append(X)
            for idx in range(len(named_entity_list)):
                labels[idx].append(y[idx])
        features = np.vstack(features)
        for idx in range(len(named_entity_list)):
            labels[idx] = np.vstack(labels[idx])
        print('')
        print(f'X.shape = {features.shape}')
        for ne_id, ne_cls in enumerate(named_entity_list):
            print(f'y[{ne_cls}].shape = {labels[ne_id].shape}')
        with open(dst_fname, 'wb') as fp:
            pickle.dump(
                obj=(features, labels),
                file=fp,
                protocol=pickle.HIGHEST_PROTOCOL
            )
    else:
        with codecs.open(src_fname, mode='r', encoding='utf-8',
                         errors='ignore') as fp:
            cur_line = fp.readline()
            while len(cur_line) > 0:
                prep_line = cur_line.strip()
                if len(prep_line) > 0:
                    X = calc_features(
                        bert_tokenizer,
                        bert_model,
                        max_len,
                        prep_line
                    )
                    features.append(X)
                cur_line = fp.readline()
        features = np.vstack(features)
        print('')
        print(f'X.shape = {features.shape}')
        with open(dst_fname, 'wb') as fp:
            pickle.dump(
                obj=features,
                file=fp,
                protocol=pickle.HIGHEST_PROTOCOL
            )


if __name__ == '__main__':
    main()
