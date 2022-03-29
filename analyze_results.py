import codecs
import json
import os
import random
import sys

import numpy as np
from tqdm import tqdm

from io_utils.io_utils import load_data
from evaluation.evaluation import split_text_by_words, compare_entities
from evaluation.evaluation import calc_f1


def main():
    random.seed(42)
    np.random.seed(42)

    if len(sys.argv) < 2:
        err_msg = 'The true data file is not specified!'
        raise ValueError(err_msg)
    true_fname = os.path.normpath(sys.argv[1])
    if len(sys.argv) < 3:
        err_msg = 'The predicted data file is not specified!'
        raise ValueError(err_msg)
    prediction_fname = os.path.normpath(sys.argv[2])
    if len(sys.argv) < 4:
        err_msg = 'The NER vocabulary file is not specified!'
        raise ValueError(err_msg)
    ners_fname = os.path.normpath(sys.argv[3])
    if len(sys.argv) < 5:
        err_msg = 'The analysis result file is not specified!'
        raise ValueError(err_msg)
    result_fname = os.path.normpath(sys.argv[4])

    if not os.path.isfile(true_fname):
        raise ValueError(f'The file {true_fname} does not exist!')
    if not os.path.isfile(prediction_fname):
        raise ValueError(f'The file {prediction_fname} does not exist!')
    if not os.path.isfile(ners_fname):
        raise ValueError(f'The file {ners_fname} does not exist!')
    result_dirname = os.path.dirname(result_fname)
    if len(result_dirname) > 0:
        if not os.path.isdir(result_dirname):
            raise ValueError(f'The directory {result_dirname} does not exist!')

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

    true_data = load_data(true_fname)
    predicted_data = load_data(prediction_fname)
    set_of_IDs = set(true_data.keys())
    if set_of_IDs != set(predicted_data.keys()):
        err_msg = f'The predicted data "{prediction_fname}" does not ' \
                  f'correspond to the true data "{true_fname}".'
        raise ValueError(err_msg)

    analysis_results = dict()
    for cur_entity in tqdm(sorted(list(possible_named_entities))):
        analysis_fname = result_fname
        point_idx = analysis_fname.rfind('.')
        if point_idx >= 0:
            analysis_fname = analysis_fname[:point_idx] + f'_{cur_entity}' + \
                             analysis_fname[point_idx:]
        else:
            analysis_fname += f'_{cur_entity}'
        results_for_cur_entity = []
        for cur_ID in true_data.keys():
            source_text, true_entities = true_data[cur_ID]
            source_text_, predicted_entities = predicted_data[cur_ID]
            if source_text_ != source_text:
                err_msg = f'The predicted sample {cur_ID} from the file ' \
                          f'"{prediction_fname}" does not correspond to ' \
                          f'the true sample from the file "{true_fname}".'
                raise ValueError(err_msg)
            source_words, true_entities = split_text_by_words(
                s=source_text,
                entities=true_entities
            )
            source_words_, predicted_entities = split_text_by_words(
                s=source_text_,
                entities=predicted_entities
            )
            del source_text_, source_words_
            compared = compare_entities(true_entities, predicted_entities,
                                        cur_entity)
            if compared is not None:
                f1 = calc_f1(compared)
                new_res = {
                    'id': cur_ID,
                    'text': ' '.join(source_words),
                    'tp': list(map(
                        lambda it: [it[0], it[1],
                                    ' '.join(source_words[it[0]:it[1]])],
                        compared['tp']
                    )),
                    'fp': list(map(
                        lambda it: [it[0], it[1],
                                    ' '.join(source_words[it[0]:it[1]])],
                        compared['fp']
                    )),
                    'fn': list(map(
                        lambda it: [it[0], it[1],
                                    ' '.join(source_words[it[0]:it[1]])],
                        compared['fn']
                    )),
                    'f1': f1
                }
                results_for_cur_entity.append(new_res)
            del source_words, true_entities, predicted_entities
        if len(results_for_cur_entity) == 0:
            err_msg = f'There are no data for the entity "{cur_entity}".'
            raise ValueError(err_msg)
        results_for_cur_entity.sort(key=lambda it: (-it['f1'], -len(it['tp']),
                                                    it['id']))
        analysis_results[cur_entity] = results_for_cur_entity
        with codecs.open(analysis_fname, mode='w', encoding='utf-8') as fp:
            json.dump(
                obj=results_for_cur_entity,
                fp=fp,
                ensure_ascii=False,
                indent=4
            )
        del results_for_cur_entity

    with codecs.open(result_fname, mode='w', encoding='utf-8') as fp:
        json.dump(
            obj=analysis_results,
            fp=fp,
            ensure_ascii=False,
            indent=4
        )


if __name__ == '__main__':
    main()
