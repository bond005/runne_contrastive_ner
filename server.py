import copy
import logging
import os
from typing import Dict, List, Tuple, Union

from flask import Flask, request, jsonify
import numpy as np
from scipy.special import softmax
import tensorflow as tf

from io_utils.io_utils import load_data, save_data
from data_processing.tokenization import tokenize_text, sentenize_text
from data_processing.postprocessing import decode_entity
from neural_network.ner import load_ner


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
app = Flask(__name__)

trained_ner_path = os.path.join(os.path.dirname(__file__), 'models',
                                'dp_rubert_from_siamese')
if not os.path.isdir(trained_ner_path):
    raise ValueError(f'The directory "{trained_ner_path}" does not exist!')
ner_model, ner_tokenizer, max_sent_len, ne_list = load_ner(trained_ner_path)


def check_input_data(data: List[Union[str, dict]]) -> str:
    res = ''
    err_msg = ''
    for idx, cur in enumerate(data):
        if isinstance(cur, str):
            if len(res) == 0:
                res = 'str'
            else:
                if res != 'str':
                    err_msg = f'Data type of sample {idx} of input data is ' \
                              f'unexpected! Expected {res}, got {type(cur)}.'
                    break
        elif isinstance(cur, dict):
            if 'text' in cur:
                if isinstance(cur['text'], str):
                    if len(res) == 0:
                        res = 'dict'
                    else:
                        if res != 'dict':
                            err_msg = f'Data type of sample {idx}["text"] of ' \
                                      f'input data is unexpected! ' \
                                      f'Expected str, got {cur["text"]}.'
                            break
                else:
                    err_msg = ''
                    break
            else:
                err_msg = f'Sample {idx} describes uknown data! ' \
                          f'The `text` is not found in the key list ' \
                          f'{sorted(list(cur.keys()))}.'
                break
        else:
            err_msg = f'Data type of sample {idx} of input data is wrong! ' \
                      f'Expected str or dict, got {type(cur)}.'
            break
    if len(err_msg) > 0:
        raise ValueError(err_msg)
    if len(res) == 0:
        raise ValueError('The input data are empty!')
    return res


def extract_texts(data: List[Union[str, dict]]) -> List[str]:
    data_type = check_input_data(data)
    if data_type == 'str':
        prepared_data = data
    else:
        prepared_data = [cur['text'] for cur in data]
    return prepared_data


def enrich_data_with_recognition_results(
        data: List[Union[str, dict]],
        recognition_results: List[List[Tuple[int, int, str]]]
) -> List[Union[str, dict]]:
    data_type = check_input_data(data)
    if len(data) != len(recognition_results):
        err_msg = f'Source data do not correspond to recognition results! ' \
                  f'{len(data)} != {len(recognition_results)}'
        raise ValueError(err_msg)
    enriched_data = []
    if data_type == 'str':
        for text, res in zip(data, recognition_results):
            new_res = {'text': text, 'ners': res}
            enriched_data.append(new_res)
    else:
        for cur_sample, cur_res in zip(data, recognition_results):
            new_res = copy.deepcopy(cur_sample)
            new_res['ners'] = cur_res
            enriched_data.append(new_res)
    return enriched_data


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/recognize', methods=['POST'])
def recognize():
    global ner_model, ner_tokenizer, max_sent_len, ne_list
