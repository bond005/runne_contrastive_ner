from typing import List, Tuple

import numpy as np
from transformers import TFBertModel,BertTokenizer

from data_processing.tokenization import sentenize_text, tokenize_text


def calc_features(tokenizer: BertTokenizer, feature_extractor: TFBertModel,
                  max_sent_len: int, source_text: str) -> \
        Tuple[List[Tuple[str, int, int]], np.ndarray]:
    word_features = []
    all_words = []
    for sent_start, sent_end in sentenize_text(source_text):
        words, subtokens, subtoken_bounds = tokenize_text(
            s=source_text[sent_start:sent_end],
            tokenizer=tokenizer
        )
        while (len(subtokens) % max_sent_len) != 0:
            subtokens.append(tokenizer.pad_token)
            subtoken_bounds.append(None)
        x = []
        start_pos = 0
        for _ in range(len(subtokens) // max_sent_len):
            end_pos = start_pos + max_sent_len
            subtoken_indices = tokenizer.convert_tokens_to_ids(
                subtokens[start_pos:end_pos]
            )
            x.append(
                np.array(
                    subtoken_indices,
                    dtype=np.int32
                ).reshape((1, max_sent_len))
            )
            start_pos = end_pos
        predicted = feature_extractor.predict(np.vstack(x), batch_size=1)[0]
        if not isinstance(predicted, np.ndarray):
            predicted = predicted.numpy()
        if len(predicted.shape) != 3:
            err_msg = f'The predicted feature matrix is wrong! ' \
                      f'Expected 3-D array, got {len(predicted.shape)}-D one.'
            raise ValueError(err_msg)
        if predicted.shape[0] != (len(subtokens) // max_sent_len):
            err_msg = f'The predicted feature matrix does not correspond to' \
                      f' the input data! {predicted.shape[0]} != ' \
                      f'{len(subtokens) // max_sent_len}.'
            raise ValueError(err_msg)
        subtoken_features = [predicted[0]]
        for idx in range(1, predicted.shape[0]):
            subtoken_features.append(predicted[idx])
        subtoken_features = np.vstack(subtoken_features)
        del predicted
        for cur_word, word_start, word_end in words:
            word_features.append(
                np.mean(subtoken_features[word_start:word_end],
                        axis=0, keepdims=True)
            )
            all_words.append((
                cur_word,
                subtoken_bounds[word_start][0] + sent_start,
                subtoken_bounds[word_end - 1][1] + sent_start
            ))
    return all_words, np.vstack(word_features)


def find_entity_words(words: List[Tuple[str, int, int]],
                      entity_start: int, entity_end: int) -> Tuple[int, int]:
    start_word_idx = -1
    end_word_idx = -1
    for word_idx, (_, word_start, word_end) in enumerate(words):
        if entity_start < word_end:
            if start_word_idx < 0:
                start_word_idx = word_idx
        if entity_end > word_start:
            end_word_idx = word_idx
        if word_start >= entity_end:
            break
    if (start_word_idx < 0) or (end_word_idx < 0):
        return -1, -1
    true_entity_start = words[start_word_idx][1]
    true_entity_end = words[end_word_idx][2]
    if entity_end <= true_entity_start:
        return -1, -1
    if entity_start >= true_entity_end:
        return -1, -1
    return start_word_idx, end_word_idx + 1


def calc_features_and_labels(tokenizer: BertTokenizer,
                             feature_extractor: TFBertModel, max_sent_len: int,
                             ne_list: List[str], source_text: str,
                             annotation: List[Tuple[str, int, int]]) -> \
        Tuple[np.ndarray, List[np.ndarray]]:
    words, features = calc_features(tokenizer, feature_extractor, max_sent_len,
                                    source_text)
    named_entities = [np.zeros((len(words), 5), dtype=np.float32)
                      for _ in range(len(ne_list))]
    for word_idx in range(len(words)):
        for named_entity_id in range(len(ne_list)):
            named_entities[named_entity_id][word_idx, 0] = 1.0
    for entity_class, entity_char_start, entity_char_end in annotation:
        try:
            named_entity_id = ne_list.index(entity_class)
        except:
            named_entity_id = -1
        if named_entity_id < 0:
            err_msg = f'The entity class "{entity_class}" is unknown!'
            raise ValueError(err_msg)
        entity_start, entity_end = find_entity_words(words, entity_char_start,
                                                     entity_char_end)
        if (entity_start < 0) or (entity_end < 0):
            unknown_entity = (entity_class, entity_char_start, entity_char_end)
            input_text = source_text.replace("\n", " ").replace("\r", " ")
            err_msg = f'The entity {unknown_entity} is not found in the text ' \
                      f'"{input_text}", tokenized by the following words: ' \
                      f'{words}.'
            raise ValueError(err_msg)
        if entity_end - entity_start > 1:
            named_entities[named_entity_id][entity_start, 0] = 0.0
            named_entities[named_entity_id][entity_start, 1] = 1.0
            for word_idx in range(entity_start + 1, entity_end - 1):
                named_entities[named_entity_id][word_idx, 0] = 0.0
                named_entities[named_entity_id][word_idx, 3] = 1.0
            named_entities[named_entity_id][entity_end - 1, 0] = 0.0
            named_entities[named_entity_id][entity_end - 1, 2] = 1.0
        else:
            named_entities[named_entity_id][entity_start, 0] = 0.0
            named_entities[named_entity_id][entity_start, 4] = 1.0
    return features, named_entities
