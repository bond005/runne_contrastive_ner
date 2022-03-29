from typing import Dict, List, Set, Tuple, Union

from razdel import tokenize

from data_processing.tokenization import find_span


def split_text_by_words(s: str, entities: List[Tuple[str, int, int]]) -> \
        Tuple[List[str], List[Tuple[str, int, int]]]:
    word_bounds = list(filter(
        lambda it2: len(s[it2[0]:it2[1]].strip()) > 0,
        map(
            lambda it1: (tuple(it1)[0], tuple(it1)[1]),
            tokenize(s.replace('​', ' '))
        )
    ))
    words = [s[start_:end_] for start_, end_ in word_bounds]
    new_entities = []
    for ne_class, ne_start, ne_end in entities:
        word_start_idx = find_span(word_bounds, ne_start)
        word_end_idx = find_span(word_bounds, ne_end - 1)
        if (word_start_idx < 0) or (word_end_idx < word_start_idx):
            err_msg = f'The entity {(ne_class, ne_start, ne_end)} cannot be ' \
                      f'found in the word bounds {word_bounds}.'
            raise ValueError(err_msg)
        new_entities.append((ne_class, word_start_idx, word_end_idx + 1))
    return words, new_entities


def compare_entities(true_entities: List[Tuple[str, int, int]],
                     predicted_entities: List[Tuple[str, int, int]],
                     entity_class: str) -> \
        Union[None, Dict[str, Set[Tuple[int, int]]]]:
    filtered_true_entities = set(map(
        lambda it: (it[1], it[2]),
        filter(
            lambda it: it[0] == entity_class,
            true_entities
        )
    ))
    filtered_predicted_entities = set(map(
        lambda it: (it[1], it[2]),
        filter(
            lambda it: it[0] == entity_class,
            predicted_entities
        )
    ))
    if (len(filtered_true_entities) == 0) and \
            (len(filtered_predicted_entities) == 0):
        return None
    res = {'tp': set(), 'fp': set(), 'fn': set()}
    if len(filtered_true_entities) == 0:
        res['fp'] = filtered_predicted_entities
    elif len(filtered_predicted_entities) == 0:
        res['fn'] = filtered_true_entities
    else:
        res['tp'] = filtered_true_entities & filtered_predicted_entities
        res['fp'] = filtered_predicted_entities - filtered_true_entities
        res['fn'] = filtered_true_entities - filtered_predicted_entities
    return res


def calc_f1(result: Dict[str, Set[Tuple[int, int]]]) -> float:
    true_keys = {'fp', 'fn', 'tp'}
    if true_keys != set(result.keys()):
        err_msg = f'Keys "{set(result.keys())}" are wrong!'
        raise ValueError(err_msg)
    tp = float(len(result['tp']))
    fp = float(len(result['fp']))
    fn = float(len(result['fn']))
    if tp > 0.0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1
