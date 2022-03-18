import codecs
import os
import pickle
import random
import sys

from transformers import BertTokenizer

from io_utils.io_utils import load_data
from trainset_building.trainset_building import build_trainset_for_siam
from trainset_building.trainset_building import build_trainset_for_ner


def main():
    random.seed(42)
    if len(sys.argv) < 2:
        err_msg = 'The source training file is not specified!'
        raise ValueError(err_msg)
    src_fname = os.path.normpath(sys.argv[1])
    if len(sys.argv) < 3:
        err_msg = 'The destination training file is not specified!'
        raise ValueError(err_msg)
    dst_fname = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The NER vocabulary file is not specified!'
        raise ValueError(err_msg)
    ners_fname = os.path.normpath(sys.argv[3])
    if len(sys.argv) < 5:
        err_msg = 'The training mode is not specified!'
        raise ValueError(err_msg)
    training_mode = sys.argv[4].strip().lower()
    if len(training_mode) == 0:
        err_msg = 'The training mode is not specified!'
        raise ValueError(err_msg)
    if len(sys.argv) < 6:
        err_msg = 'The maximal sequence length is not specified!'
        raise ValueError(err_msg)
    try:
        max_len = int(sys.argv[5])
    except:
        max_len = 0
    if max_len < 1:
        err_msg = f'{sys.argv[5]} is inadmissible value ' \
                  f'of the maximal sequence length!'
        raise ValueError(err_msg)
    if len(sys.argv) < 7:
        err_msg = 'The pre-trained BERT model is not specified!'
        raise ValueError(err_msg)
    pretrained_model = sys.argv[6]
    if training_mode == 'siamese':
        if len(sys.argv) < 8:
            err_msg = 'The maximal number of samples is not specified!'
            raise ValueError(err_msg)
        try:
            max_samples = int(sys.argv[7])
        except:
            max_samples = 0
        if max_samples < 1:
            err_msg = f'{sys.argv[7]} is inadmissible value ' \
                      f'of the maximal number of samples!'
            raise ValueError(err_msg)
    else:
        max_samples = 0

    if not os.path.isfile(src_fname):
        raise IOError(f'The file {src_fname} does not exist!')
    if not os.path.isfile(ners_fname):
        raise IOError(f'The file {ners_fname} does not exist!')
    dname = os.path.dirname(dst_fname)
    if len(dname) > 0:
        if not os.path.isdir(dname):
            raise IOError(f'The directory {dname} does not exist!')
    if training_mode not in {'siamese', 'ner'}:
        err_msg = f'The training mode {training_mode} is unknown! ' \
                  f'Possible values: siamese, ner.'
        raise ValueError(err_msg)

    with codecs.open(ners_fname, mode='r', encoding='utf-8') as fp:
        possible_named_entities = list(filter(
            lambda it2: len(it2) > 0,
            map(
                lambda it1: it1.strip(),
                fp.readlines()
            )
        ))
    if len(possible_named_entities) == 0:
        err_msg = f'The file {ners_fname} is empty!'
        raise IOError(err_msg)
    if len(possible_named_entities) != len(set(possible_named_entities)):
        err_msg = f'The file {ners_fname} contains a wrong data! ' \
                  f'Some entities are duplicated!'
        raise IOError(err_msg)

    source_data = load_data(src_fname)
    bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    if training_mode == 'ner':
        prep_data = build_trainset_for_ner(
            data=source_data,
            tokenizer=bert_tokenizer,
            entities=possible_named_entities,
            max_seq_len=max_len
        )
        print('')
        print(f'X.shape = {prep_data[0].shape}')
        for output_idx in range(len(prep_data[1])):
            print(f'y[{output_idx}].shape = {prep_data[1][output_idx].shape}')
    else:
        prep_data = build_trainset_for_siam(
            data=source_data,
            tokenizer=bert_tokenizer,
            entities=possible_named_entities,
            max_seq_len=max_len,
            max_samples=max_samples
        )

    with open(dst_fname, 'wb') as fp:
        pickle.dump(
            file=fp,
            obj=prep_data,
            protocol=pickle.HIGHEST_PROTOCOL
        )


if __name__ == '__main__':
    main()
