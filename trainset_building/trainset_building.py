import copy
import random
from typing import Dict, List, Tuple
import warnings

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

from data_processing.tokenization import sentenize_text_with_ners


def transform_indicator_to_classmatrix(indicator: List[int]) -> np.ndarray:
    max_seq_len = len(indicator)
    one_hot_indicator = np.zeros((1, max_seq_len, 5), dtype=np.float32)
    for token_idx in range(len(indicator)):
        if indicator[token_idx] > 0:
            if (token_idx > 0) and (token_idx < (max_seq_len - 1)):
                if indicator[token_idx - 1] > 0:
                    if indicator[token_idx] > 1:
                        if indicator[token_idx + 1] > 0:
                            class_idx = 1  # start ent
                        else:
                            class_idx = 3  # start-end ent
                    else:
                        if indicator[token_idx + 1] > 1:
                            class_idx = 2  # end ent
                        elif indicator[token_idx + 1] > 0:
                            class_idx = 3  # middle ent
                        else:
                            class_idx = 2  # end ent
                elif indicator[token_idx + 1] > 0:
                    if indicator[token_idx + 1] > 1:
                        class_idx = 4  # start-end ent
                    else:
                        class_idx = 1  # start ent
                else:
                    class_idx = 4  # start-end ent
            elif token_idx > 0:  # token_idx == (max_seq_len - 1)
                if indicator[token_idx - 1] > 0:
                    if indicator[token_idx - 1] > 1:
                        class_idx = 4  # start-end ent
                    else:
                        class_idx = 2  # end ent
                else:
                    class_idx = 4  # start-end ent
            else:  # token_idx == 0
                if indicator[token_idx + 1] > 0:
                    if indicator[token_idx + 1] > 1:
                        class_idx = 4  # start-end ent
                    else:
                        class_idx = 1  # start ent
                else:
                    class_idx = 4  # start-end ent
        else:
            class_idx = 0  # no ent
        one_hot_indicator[0, token_idx, class_idx] = 1.0
    return one_hot_indicator


def build_trainset_for_ner(data: Dict[int,
                                      Tuple[str, List[Tuple[str, int, int]]]],
                           tokenizer: BertTokenizer, max_seq_len: int,
                           entities: List[str]) \
        -> Tuple[np.ndarray, List[np.ndarray]]:
    if 'O' in entities:
        err_msg = f'The entities list {entities} is wrong ' \
                  f'because it contains the `O` entity.'
        raise ValueError(err_msg)
    list_of_tokenized_texts = []
    list_of_ne_indicators = []
    max_seq_len_ = max_seq_len
    print(f'Number of texts is {len(data)}.')
    for cur_id in tqdm(sorted(list(data.keys()))):
        text, ners = data[cur_id]
        batch = sentenize_text_with_ners(
            s=text,
            tokenizer=tokenizer,
            ners=ners,
            ne_vocabulary=entities
        )
        for tokenized_text, ne_indicators in batch:
            list_of_tokenized_texts.append(tokenized_text)
            list_of_ne_indicators.append(ne_indicators)
            if len(tokenized_text) > max_seq_len_:
                max_seq_len_ = len(tokenized_text)
    print(f'Number of sentences is {len(list_of_tokenized_texts)}.')
    X = []
    y = [[] for _ in range(len(entities))]
    for tokenized_text, ne_indicators in zip(list_of_tokenized_texts,
                                             list_of_ne_indicators):
        ne_indicators_ = copy.copy(ne_indicators)
        while len(tokenized_text) < max_seq_len_:
            tokenized_text.append(tokenizer.pad_token)
            for ne_id in range(len(entities)):
                ne_indicators_[ne_id].append(0)
        X.append(tokenizer.convert_tokens_to_ids(tokenized_text))
        for ne_id in range(len(entities)):
            y[ne_id].append(
                transform_indicator_to_classmatrix(ne_indicators_[ne_id])
            )
        del ne_indicators_
    X = np.array(X, dtype=np.int32)
    y = [np.concatenate(cur, axis=0) for cur in y]
    if X.shape[1] == max_seq_len:
        return X, y
    indices_of_long_texts = []
    for sample_idx in range(X.shape[0]):
        is_padding = True
        for token_idx in range(max_seq_len, X.shape[1]):
            if X[sample_idx, token_idx] != tokenizer.pad_token_id:
                is_padding = False
                break
        if not is_padding:
            indices_of_long_texts.append(sample_idx)
    iteration = 1
    while len(indices_of_long_texts) > 0:
        print(f'Iter {iteration}: '
              f'there are {len(indices_of_long_texts)} very long texts!')
        new_X = np.full(
            shape=(len(indices_of_long_texts), max_seq_len_),
            fill_value=tokenizer.pad_token_id,
            dtype=np.int32
        )
        new_y = [np.zeros((len(indices_of_long_texts), max_seq_len_, 5),
                          dtype=np.float32) for _ in range(len(y))]
        ndiff = max_seq_len_ - max_seq_len
        for local_idx, global_idx in enumerate(indices_of_long_texts):
            new_X[local_idx, 0:ndiff] = X[global_idx, max_seq_len:]
            X[global_idx, max_seq_len:] = tokenizer.pad_token_id
            for output_idx in range(len(y)):
                new_y[output_idx][local_idx, 0:ndiff, :] = \
                    y[output_idx][global_idx, max_seq_len:, :]
                y[output_idx][global_idx, max_seq_len:, :] = 0.0
        X = np.concatenate((X, new_X), axis=0)
        y = [np.concatenate((y[output_idx], new_y[output_idx]), axis=0)
             for output_idx in range(len(y))]
        indices_of_long_texts = []
        for sample_idx in range(X.shape[0]):
            is_padding = True
            for token_idx in range(max_seq_len, X.shape[1]):
                if X[sample_idx, token_idx] != tokenizer.pad_token_id:
                    is_padding = False
                    break
            if not is_padding:
                indices_of_long_texts.append(sample_idx)
        iteration += 1
    X = X[:, :max_seq_len]
    y = [cur[:, :max_seq_len, :] for cur in y]
    print(f'Number of sentences after cutting is {X.shape[0]}.')
    return X, y


def build_trainset_for_siam(data: Dict[int,
                                       Tuple[str, List[Tuple[str, int, int]]]],
                            tokenizer: BertTokenizer, max_seq_len: int,
                            entities: List[str], max_samples: int) \
        -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                 np.ndarray]:
    eps = 1e-6
    X, y = build_trainset_for_ner(data, tokenizer, max_seq_len, entities)
    entities_in_data = {'O': []}
    for sample_idx in range(X.shape[0]):
        start_pos = -1
        for token_idx in range(X.shape[1]):
            if X[sample_idx, token_idx] == tokenizer.pad_token_id:
                if start_pos >= 0:
                    entities_in_data['O'].append(
                        (
                            sample_idx,
                            (start_pos, token_idx)
                        )
                    )
                    start_pos = -1
                break
            is_entity = False
            for ent_id in range(len(entities)):
                if y[ent_id][sample_idx, token_idx, 0] < eps:
                    is_entity = True
                    break
            if is_entity:
                if start_pos >= 0:
                    entities_in_data['O'].append(
                        (
                            sample_idx,
                            (start_pos, token_idx)
                        )
                    )
                    start_pos = -1
            else:
                if start_pos < 0:
                    start_pos = token_idx
        if start_pos >= 0:
            entities_in_data['O'].append(
                (
                    sample_idx,
                    (start_pos, X.shape[1])
                )
            )
    for ent_id, ent_type in enumerate(entities):
        entities_in_data[ent_type] = []
        for sample_idx in range(X.shape[0]):
            start_pos = -1
            for token_idx in range(X.shape[1]):
                if X[sample_idx, token_idx] == tokenizer.pad_token_id:
                    if start_pos >= 0:
                        entities_in_data[ent_type].append(
                            (
                                sample_idx,
                                (start_pos, token_idx)
                            )
                        )
                        start_pos = -1
                    break
                if y[ent_id][sample_idx, token_idx, 0] >= eps:
                    if start_pos >= 0:
                        entities_in_data[ent_type].append(
                            (
                                sample_idx,
                                (start_pos, token_idx)
                            )
                        )
                        start_pos = -1
                else:
                    if start_pos < 0:
                        start_pos = token_idx
            if start_pos >= 0:
                entities_in_data[ent_type].append(
                    (
                        sample_idx,
                        (start_pos, X.shape[1])
                    )
                )
    del y
    used_pairs = set()
    X_left_tokens = np.empty((max_samples, max_seq_len), dtype=np.int32)
    X_left_masks = np.zeros((max_samples, max_seq_len), dtype=np.int32)
    X_right_tokens = np.empty((max_samples, max_seq_len), dtype=np.int32)
    X_right_masks = np.zeros((max_samples, max_seq_len), dtype=np.int32)
    y = np.empty((max_samples, 1), dtype=np.float32)
    entities_and_O = entities + ['O']
    counter = 0
    for _ in tqdm(range(max_samples)):
        first_entity = random.choice(entities_and_O)
        idx = entities_and_O.index(first_entity)
        if random.random() > 0.3:
            second_entity = random.choice(
                entities_and_O[:idx] + entities_and_O[(idx + 1):]
            )
        else:
            second_entity = first_entity
        first_sample = random.choice(entities_in_data[first_entity])
        second_sample = random.choice(entities_in_data[second_entity])
        while second_sample == first_sample:
            second_sample = random.choice(entities_in_data[second_entity])
        if (first_sample, second_sample) in used_pairs:
            for _ in range(100):
                first_sample = random.choice(entities_in_data[first_entity])
                second_sample = random.choice(entities_in_data[second_entity])
                while second_sample == first_sample:
                    second_sample = random.choice(
                        entities_in_data[second_entity]
                    )
        if (first_sample, second_sample) in used_pairs:
            warn_msg = f'The pair {first_entity}-{second_entity} is not found.'
            warnings.warn(warn_msg)
        X_left_tokens[counter] = X[first_sample[0]]
        X_right_tokens[counter] = X[second_sample[0]]
        for token_idx in range(first_sample[1][0], first_sample[1][1]):
            X_left_masks[counter, token_idx] = 1
        for token_idx in range(second_sample[1][0], second_sample[1][1]):
            X_right_masks[counter, token_idx] = 1
        if first_entity == second_entity:
            y[counter, 0] = 1.0
        else:
            y[counter, 0] = 0.0
        counter += 1
        used_pairs.add((first_sample, second_sample))
        used_pairs.add((second_sample, first_sample))
    if counter < max_samples:
        print(f'{counter} samples from {max_samples} are built.')
    else:
        print(f'All {max_samples} samples are built.')
    del X
    X = (
        X_left_tokens[:counter],
        X_left_masks[:counter],
        X_right_tokens[:counter],
        X_right_masks[:counter]
    )
    return X, y[:counter]
