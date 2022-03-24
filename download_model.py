import os
import shutil
import zipfile

import requests
from tqdm import tqdm
from urllib.parse import urlencode


def download_ner() -> bool:
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://yadi.sk/d/7CQPhR2SAu6mxw'
    final_url = base_url + urlencode(dict(public_key=public_key))
    pk_request = requests.get(final_url)
    direct_link = pk_request.json().get('href')
    response = requests.get(direct_link, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
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


if __name__ == '__main__':
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
