import os
import random
import sys

import numpy as np
from scipy.special import softmax
import tensorflow as tf
from tqdm import tqdm

from io_utils.io_utils import load_data, save_data
from data_processing.tokenization import tokenize_text, sentenize_text
from data_processing.postprocessing import decode_entity
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
        err_msg = 'The trained NER is not specified!'
        raise ValueError(err_msg)
    trained_ner_path = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The destination file with results is not specified!'
        raise ValueError(err_msg)
    dst_fname = os.path.normpath(sys.argv[3])

    if not os.path.isfile(src_fname):
        raise ValueError(f'The file "{src_fname}" does not exist!')
    if not os.path.isdir(trained_ner_path):
        raise ValueError(f'The directory "{trained_ner_path}" does not exist!')
    dst_dir = os.path.dirname(dst_fname)
    if len(dst_dir) > 0:
        if not os.path.isdir(dst_dir):
            err_msg = f'The directory "{dst_dir}: is not specified!'
            raise ValueError(err_msg)

    source_data = load_data(src_fname)
    ner_model, ner_tokenizer, max_sent_len, ne_list = load_ner(trained_ner_path)

    for cur_id in tqdm(sorted(list(source_data.keys()))):
        cur_text = source_data[cur_id][0]
        recognized_entities = []
        if len(cur_text.strip()) > 0:
            for sent_start, sent_end in sentenize_text(cur_text):
                words, subtokens, subtoken_bounds = tokenize_text(
                    s=cur_text[sent_start:sent_end],
                    tokenizer=ner_tokenizer
                )
                while (len(subtokens) % max_sent_len) != 0:
                    subtokens.append(ner_tokenizer.pad_token)
                    subtoken_bounds.append(None)
                x = []
                start_pos = 0
                for _ in range(len(subtokens) // max_sent_len):
                    end_pos = start_pos + max_sent_len
                    subtoken_indices = ner_tokenizer.convert_tokens_to_ids(
                        subtokens[start_pos:end_pos]
                    )
                    x.append(
                        np.array(
                            subtoken_indices,
                            dtype=np.int32
                        ).reshape((1, max_sent_len))
                    )
                    start_pos = end_pos
                predicted = ner_model.predict(np.vstack(x), batch_size=1)
                if len(predicted) != len(ne_list):
                    err_msg = f'Number of neural network heads does not ' \
                              f'correspond to number of named entities! ' \
                              f'{len(predicted)} != {len(ne_list)}'
                    raise ValueError(err_msg)
                del x
                probability_matrices = [
                    np.vstack([
                        cur[sample_idx]
                        for sample_idx in range(len(subtokens) // max_sent_len)
                    ])
                    for cur in predicted
                ]
                del predicted
                for ne_idx in range(len(ne_list)):
                    entity_bounds = decode_entity(
                        softmax(probability_matrices[ne_idx], axis=1),
                        words
                    )
                    if len(entity_bounds) > 0:
                        for start_subtoken, end_subtoken in entity_bounds:
                            entity_start = subtoken_bounds[start_subtoken][0]
                            entity_end = subtoken_bounds[end_subtoken - 1][1]
                            recognized_entities.append((
                                ne_list[ne_idx],
                                sent_start + entity_start,
                                sent_start + entity_end
                            ))
                    del entity_bounds
                del words, subtokens, subtoken_bounds
        source_data[cur_id] = (cur_text, recognized_entities)

    random_identifiers = random.sample(
        population=sorted(list(source_data.keys())),
        k=5
    )
    print('')
    print('5 random samples with predictions')
    print('')
    max_txt_width = max(map(lambda it: len(it), ne_list))
    for cur_id in random_identifiers:
        print('====================')
        print(f'Sample {cur_id}')
        print('====================')
        cur_text, cur_predictions = source_data[cur_id]
        cur_text = cur_text.replace('\n', ' ')
        cur_text = cur_text.replace('\r', ' ')
        cur_text = cur_text.replace('\t', ' ')
        print(cur_text)
        print('')
        for entity_type, entity_start, entity_end in cur_predictions:
            entity_text = cur_text[entity_start:entity_end]
            print('{0:>{1}} {2}'.format(entity_type, max_txt_width,
                                        entity_text))
        print('')

    save_data(dst_fname, False, source_data)


if __name__ == '__main__':
    main()
