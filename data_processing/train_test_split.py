import random
import re
from typing import Dict, List, Tuple, Union


def calc_entity_freqs(data: Dict[int, Tuple[str, List[Tuple[str, int, int]]]],
                      id_list: Union[List[int], None] = None) -> Dict[str, int]:
    frequencies = dict()
    re_for_entity = re.compile(r'^[A-Z]+[_A-Z]*[A-Z]+$')
    if id_list is None:
        id_list_ = list(data.keys())
    else:
        id_list_ = id_list
    for identifier in id_list_:
        _, ners = data[identifier]
        for ne_type, _, _ in ners:
            err_msg = f'{ne_type} is inadmissible named entity class!'
            if ne_type.startswith('B-') or ne_type.startswith('I-') or \
                    (ne_type == 'O'):
                raise ValueError(err_msg)
            if re_for_entity.search(ne_type) is None:
                raise ValueError(err_msg)
            frequencies[ne_type] = frequencies.get(ne_type, 0) + 1
    return frequencies


def train_test_split(data: Dict[int, Tuple[str, List[Tuple[str, int, int]]]]) \
        -> Tuple[Dict[int, Tuple[str, List[Tuple[str, int, int]]]],
                 Dict[int, Tuple[str, List[Tuple[str, int, int]]]]]:
    frequencies = calc_entity_freqs(data)
    identifiers = sorted(list(data.keys()))
    print(f'There are {len(frequencies)} named entity classes:')
    max_txt_width = max(map(lambda it: len(it), frequencies))
    max_num_width = max(map(lambda it: len(str(frequencies[it])), frequencies))
    sorted_entity_list = sorted(
        list(frequencies.keys()),
        key=lambda it: (-frequencies[it], it)
    )
    for named_entity in sorted_entity_list:
        freq = frequencies[named_entity]
        if freq < 15:
            err_msg = f'The data cannot be splitted because ' \
                      f'the entity {named_entity} is too rare ' \
                      f'(its frequency is {freq}).'
            raise ValueError(err_msg)
        print('  {0:<{1}} {2:>{3}}'.format(named_entity, max_txt_width,
                                           freq, max_num_width))
    print('')
    random.shuffle(identifiers)
    n = int(round(0.15 * float(len(identifiers))))
    training_frequencies = calc_entity_freqs(data, identifiers[n:])
    test_frequencies = calc_entity_freqs(data, identifiers[:n])
    if set(training_frequencies.keys()) == set(test_frequencies.keys()):
        ok = True
        for it in training_frequencies:
            ratio = training_frequencies[it] / float(frequencies[it])
            if ratio < 0.1:
                ok = False
                break
        if ok:
            for it in test_frequencies:
                ratio = test_frequencies[it] / float(frequencies[it])
                if ratio < 0.1:
                    ok = False
                    break
    else:
        ok = False
    if not ok:
        for _ in range(1000):
            random.shuffle(identifiers)
            training_frequencies = calc_entity_freqs(data, identifiers[n:])
            test_frequencies = calc_entity_freqs(data, identifiers[:n])
            if set(training_frequencies.keys()) == set(test_frequencies.keys()):
                ok = True
                for it in training_frequencies:
                    ratio = training_frequencies[it] / float(frequencies[it])
                    if ratio < 0.1:
                        ok = False
                        break
                if ok:
                    for it in test_frequencies:
                        ratio = test_frequencies[it] / float(frequencies[it])
                        if ratio < 0.1:
                            ok = False
                            break
            else:
                ok = False
            if ok:
                break
    if not ok:
        err_msg = 'The data cannot be splitted.'
        raise ValueError(err_msg)
    data_for_training = dict()
    data_for_testing = dict()
    for it in identifiers[n:]:
        data_for_training[it] = data[it]
    for it in identifiers[:n]:
        data_for_testing[it] = data[it]
    print('For training:')
    for named_entity in sorted_entity_list:
        freq = training_frequencies[named_entity]
        print('  {0:<{1}} {2:>{3}}'.format(named_entity, max_txt_width,
                                           freq, max_num_width))
    print('')
    print('For testing:')
    for named_entity in sorted_entity_list:
        freq = test_frequencies[named_entity]
        print('  {0:<{1}} {2:>{3}}'.format(named_entity, max_txt_width,
                                           freq, max_num_width))
    print('')
    return data_for_training, data_for_testing
