import copy
import logging
import os
import shutil
from typing import List, Tuple, Union
import zipfile

from flask import Flask, request, jsonify
import requests
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from urllib.parse import urlencode

from data_processing.tokenization import tokenize_text, sentenize_text
from data_processing.postprocessing import decode_entity
from neural_network.ner import load_ner


ner_logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
app = Flask(__name__)


def download_ner() -> bool:
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://yadi.sk/d/7CQPhR2SAu6mxw'
    final_url = base_url + urlencode(dict(public_key=public_key))
    pk_request = requests.get(final_url)
    direct_link = pk_request.json().get('href')
    response = requests.get(direct_link, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    ner_logger.info(f'Total size of NER is {total_size_in_bytes} bytes.')
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    zip_archive_name = os.path.join(model_path, 'dp_rubert_from_siamese.zip')
    with open(zip_archive_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if (total_size_in_bytes != 0) and (progress_bar.n != total_size_in_bytes):
        return False
    with zipfile.ZipFile(zip_archive_name) as archive:
        archive.extractall(model_path)
    os.remove(zip_archive_name)
    return True


model_path = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.isdir(model_path):
    raise ValueError(f'The directory "{model_path}" does not exist!')
trained_ner_path = os.path.join(model_path, 'dp_rubert_from_siamese')
if not os.path.isdir(trained_ner_path):
    ner_exists = False
else:
    if not os.path.isfile(os.path.join(trained_ner_path, 'ner.h5')):
        ner_exists = False
    elif not os.path.isfile(os.path.join(trained_ner_path, 'ner.json')):
        ner_exists = False
    else:
        ner_exists = True
if not ner_exists:
    if os.path.isdir(trained_ner_path):
        shutil.rmtree(trained_ner_path, ignore_errors=True)
    if not download_ner():
        raise ValueError('The NER cannot be downloaded from Yandex Disk!')
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


def recognize_single_text(cur_text: str) -> List[Tuple[int, int, str]]:
    global ner_model, ner_tokenizer, max_sent_len, ne_list
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
                            sent_start + entity_start,
                            sent_start + entity_end,
                            ne_list[ne_idx]
                        ))
                del entity_bounds
            del words, subtokens, subtoken_bounds
    return recognized_entities


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
    request_data = request.get_json()
    if (not isinstance(request_data, str)) and \
            (not isinstance(request_data, list)):
        err_msg = f'{type(request_data)} is unknown data type for ' \
                  f'the named entity recognizer!'
        resp = jsonify({'message': err_msg})
        resp.status_code = 400
        ner_logger.error(err_msg)
    else:
        if isinstance(request_data, str):
            try:
                res = recognize_single_text(request_data)
                err_msg = ''
            except Exception as e:
                err_msg = str(e)
                res = None
            if res is None:
                resp = jsonify({'message': err_msg})
                resp.status_code = 400
                ner_logger.error(err_msg)
            else:
                resp = jsonify({
                    "text": request_data,
                    "ners": res
                })
                resp.status_code = 200
        else:
            err_msg = ''
            if len(request_data) == 0:
                err_msg = 'The input data are empty!'
                resp = jsonify({'message': err_msg})
                resp.status_code = 400
                ner_logger.error(err_msg)
            else:
                try:
                    texts_for_recognition = extract_texts(request_data)
                except Exception as e:
                    err_msg = str(e)
                    texts_for_recognition = None
                if texts_for_recognition is None:
                    resp = jsonify({'message': err_msg})
                    resp.status_code = 400
                    ner_logger.error(err_msg)
                else:
                    try:
                        res = [recognize_single_text(s)
                               for s in texts_for_recognition]
                        err_msg = ''
                    except Exception as e:
                        err_msg = str(e)
                        res = None
                    if res is None:
                        resp = jsonify({'message': err_msg})
                        resp.status_code = 400
                        ner_logger.error(err_msg)
                    else:
                        try:
                            res = enrich_data_with_recognition_results(
                                request_data, res)
                            err_msg = ''
                        except Exception as e:
                            err_msg = str(e)
                            res = None
                        if res is None:
                            resp = jsonify({'message': err_msg})
                            resp.status_code = 400
                            ner_logger.error(err_msg)
                        else:
                            resp = jsonify(res)
                            resp.status_code = 200
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010)
