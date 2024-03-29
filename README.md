[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/bond005/runne_contrastive_ner/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# RuNNE

This project is concerned with my participating in the **RuNNE** competition (**Ru**ssian **N**ested **N**amed **E**ntities) https://github.com/dialogue-evaluation/RuNNE

The RuNNE competition is devoted to a special variant of the well-known [named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) problem: nested named entities, i.e. one named entity can be a part of another one. For example, the phrase "_Donetsk National Technical University_" contains the named entity of ORGANIZATION type, but the subphrase "_Donetsk_" in the abovementioned phrase is the named entity of LOCATION type at the same time.

My solution is the third in the main track of the RuNNE competition. You can see the final results (including my result as the **bond005** user in the **SibNN** team) on this webpage https://codalab.lisn.upsaclay.fr/competitions/1863#results. Also, you can read the paper "Contrastive fine-tuning to improve generalization in deep NER" with DOI [10.28995/2075-7182-2022-21-70-80](https://www.dialog-21.ru/media/5751/bondarenkoi113.pdf), devoted to this solution.

I propose a special two-stage fine-tuning of a pretrained [Transformer neural network](https://deepai.org/machine-learning-glossary-and-terms/transformer-neural-network).

1. The first stage is a fine-tuning of the pretrained Transformer as a Siamese neural network to build new metric space with the following property: named entities of different types have a large distance in this space, and named entities of the same type have a small distance. For learning of the Siamese NN, I apply a special loss function which is known as the [Distance Based Logistic loss](https://arxiv.org/abs/1608.00161) (DBL loss).

2. The second stage is a fine-tuning of the resultant neural network as a usual NER (i.e. sequence classifier) with a [BILOU tagging scheme](https://cogcomp.seas.upenn.edu/page/publication_view/199) using a special loss function combined the [Dice loss](https://arxiv.org/abs/1911.02855) and the [Categorical Cross-Entropy loss with label smoothing](https://papers.nips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html). This NER is represented as a common Transformer base and several neural network heads. The common Transformer base is the Siamese Transformer after the first-stage fine-tuning. Each named entity type is model using an independent neural network head, because named entities are nested, i.e. several NE types can be observed in one sub-phrase.

The key motivation of the described two-stage fine-tuning is increasing of robustness and generalization ability, because the first stage is contrastive-based, and any contrastive-based loss guarantees that the Siamese neural network after its training will calculate a compact space with required semantic properties.

## Installation

This project uses a deep learning, therefore a key dependency is a deep learning framework. I prefer [Tensorflow](https://www.tensorflow.org), and you need to install CPU- or GPU-based build of Tensorflow ver. 2.5.0 or later. You can see more detailed description of dependencies in the `requirements.txt`. But if you want to install exactly the GPU-based build of this library, then before installing all dependencies from the `requirements.txt`, you need to install tensorflow for GPU manually according to the rules described here: https://www.tensorflow.org/install/pip.

Also, for installation you need to Python 3.9. I recommend using a new [Python virtual environment](https://docs.python.org/3/glossary.html#term-virtual-environment) witch can be created with [Anaconda](https://www.anaconda.com) or [venv](https://docs.python.org/3/library/venv.html#module-venv). To install this project in the selected virtual environment, you should activate this environment and run the following commands in the Terminal:

```shell
git clone https://github.com/bond005/runne_contrastive_ner.git
cd runne
python -m pip install -r requirements.txt
```

To check workability and environment setting correctness you can run the unit tests:

```shell
python -m unittest
```

## Usage

### Reproducibility

If you want to reproduce my experiments, then you have to clone the RuNNE competition repository https://github.com/dialogue-evaluation/RuNNE. You can see all training data in the `public_data` folder of this repository. I used `train.jsonl` and `ners.txt` from this folder for training. Also, I used `test.jsonl` to prepare my submit for the final (test) phase of the competition. I did several steps to build my solution and to do submit, and you can reproduce these steps.

#### Step 1

You need to split the source training data (for example, `train.jsonl`) into training and validation sub-sets:

```shell
python split_data.py \
    /path/to/dialogue-evaluation/RuNNE/public_data/train.jsonl \
    /path/to/your/competition/folder/train.jsonl \
    /path/to/your/competition/folder/val.jsonl
```

The first argument of the `split_data.py` script is a source training file, and other arguments are names of the resulted files for training and validation sub-sets.

#### Step 2

You need to prepare both your subsets (for training and for validation) as numpy matrices of indexed token sequence pairs and corresponding labels for the Transformer fine-tuning as Siamese neural network:

```shell
python prepare_trainset.py \
    /path/to/your/competition/folder/train.jsonl \
    /path/to/your/competition/folder/train_siamese_dprubert_128.pkl \
    /path/to/dialogue-evaluation/RuNNE/public_data/ners.txt \
    siamese \
    128 \
    DeepPavlov/rubert-base-cased \
    100000
```

and

```shell
python prepare_trainset.py \
    /path/to/your/competition/folder/val.jsonl \
    /path/to/your/competition/folder/val_siamese_dprubert_128.pkl \
    /path/to/dialogue-evaluation/RuNNE/public_data/ners.txt \
    siamese \
    128 \
    DeepPavlov/rubert-base-cased \
    5000
```

The **1st** and **2nd arguments** are names of the source and the resulted files with dataset.

The **3rd argument** is a named entity type vocabulary `ners.txt`, prepared by competition organizers.

The **4th argument** `siamese` specifies a type of neural network for which this dataset is created. As I wrote earlier, the first-stage fine-tuning is based on training of the Transformer as the Siamese neural network.

The **5th argument** `128` is a maximal number of sub-words in the input phrase. You can set any another value, but it must be not greater than 512.

The **6th argument** `DeepPavlov/rubert-base-cased` is a name of pre-trained BERT model. In this example the [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) is used, but I also checked other pre-trained BERTs, such as [base](https://huggingface.co/sberbank-ai/ruBert-base) and [large](https://huggingface.co/sberbank-ai/ruBert-large) BERTs from [SberAI](https://huggingface.co/sberbank-ai) during the competition.

The **7th (last) argument** sets a target number of data samples in the dataset for Siamese NN. Full dataset for Siamese NN is built as the Cartesian square of a source dataset, and so such dataset size must be restricted to some reasonably value. In this example I set 100000 samples for the training set and 5000 samples for the validation set.

#### Step 3

You need to train your Siamese Transformer using training and validation sets prepared on previous step:

```shell
python train.py \
    /path/to/your/competition/folder/train_siamese_dprubert_128.pkl \
    /path/to/your/competition/folder/val_siamese_dprubert_128.pkl \
    /path/to/your/trained/model/runne_siamese_rubert_deeppavlov \
    siamese \
    16 \
    DeepPavlov/rubert-base-cased \
    from-pytorch
```

The **1st** and **2nd arguments** are names of datasets for training and validation which were prepared on previous step.

The **3rd argument** is path to the folder into which all files of the BERT after Siamese fine-tuning will be saved. Usually, there will be three files: `config.json`, `tf_model.h5` and `vocab.txt`. But some other files such as `tokenizer_config.json` and so on can be appeared in this folder.

The **4th argument** `siamese` specifies a type of neural network for which this dataset is created. As I wrote earlier, the first-stage fine-tuning is based on training of the Transformer as the Siamese neural network.

The **5th argument** `16` is a mini-batch size. You can set any positive integer value, but a very large mini-batch can be brought to out-of-memory on your GPU.

The **6th argument** `DeepPavlov/rubert-base-cased` is a name of pre-trained BERT model. In this example the [DeepPavlov's RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) is used, but in practice I worked with [base](https://huggingface.co/sberbank-ai/ruBert-base) and [large](https://huggingface.co/sberbank-ai/ruBert-large) BERTs from [SberAI](https://huggingface.co/sberbank-ai) during the  competition.

The **7th argument** `from-pytorch` defines a source of the pretrained BERT binary model. Two values are  possible: `from-pytorch` and `from-tensorflow`. In this case, Deep Pavlov team prepared their BERT model using the PyTorch framework, therefore I set `from-pytorch` value.

#### Step 4

You need to prepare both your subsets (for training and for validation) as numpy matrices of indexed token sequences for the second stage of fine-tuning, i.e. final training of BERT as NER:

```shell
python prepare_trainset.py \
    /path/to/your/competition/folder/train.jsonl \
    /path/to/your/competition/folder/train_ner_dprubert_128.pkl \
    /path/to/dialogue-evaluation/RuNNE/public_data/ners.txt \
    ner \
    128 \
    DeepPavlov/rubert-base-cased
```

and

```shell
python prepare_trainset.py \
    /path/to/your/competition/folder/val.jsonl \
    /path/to/your/competition/folder/val_ner_dprubert_128.pkl \
    /path/to/dialogue-evaluation/RuNNE/public_data/ners.txt \
    ner \
    128 \
    DeepPavlov/rubert-base-cased
```

The arguments are similar to described ones on step 2, but I use the `ner` mode instead of the `siamese`, and I don't specify a maximal number of data samples.

#### Step 5

You have to do the second stage of fine-tuning, i.e. to train your named entity recognizer:

```shell
python train.py \
    /path/to/your/competition/folder/train_ner_dprubert_128.pkl \
    /path/to/your/competition/folder/val_ner_dprubert_128.pkl \
    /path/to/your/trained/model/runne_siamese_rubert_deeppavlov \
    ner \
    16 \
    path/to/your/trained/model/runne_ner \
    from-tensorflow \
    /path/to/dialogue-evaluation/RuNNE/public_data/ners.txt
```

This is a very similar to the step 3, but there are some differences:

- I use the `ner` mode instead of the `siamese`;
- I start the training process from my special BERT given after the first stage, i.e. I use `path/to/your/trained/model/runne_ner` and `from-tensorflow` instead of `DeepPavlov/rubert-base-cased` and `from-pytorch`;
- I add the named entity vocabulary as last argument.

All components of the fine-tuned NER after this step will be saved into the specified folder `path/to/your/trained/model/runne_ner`.

#### Step 6.

This is a final step to recognize and prepare the submission:

```shell
python recognize.py \
    /path/to/dialogue-evaluation/RuNNE/public_data/test.jsonl \
    path/to/your/trained/model/runne_ner \
    /path/to/your/submit/for/competition/test.jsonl
```

The prepared submission will be written into the file `/path/to/your/submit/for/competition/test.jsonl`. The submission file format will correspond to the competition rules.

### Docker and REST-API

You can apply the trained model of this NER for your tasks as a Docker-bases microservice. Interaction with the microservice is implemented using REST API. Firstly, you need to build the Docker image:

```shell
docker build -t bond005/runne_contrastive_ner:0.1 .
```

But the easiest way is to download the built image from Docker-Hub:

```shell
docker pull bond005/runne_contrastive_ner:0.1
```

After building (or pulling) you have to run this docker container:

```shell
docker run -p 127.0.0.1:8010:8010 bond005/runne_contrastive_ner:0.1
```

As a result, the microservice will be ready to interaction. You can send a single text, a text list or a special dictionary list. Further I describe an example of interaction between the NER microservice and a simple Python client.

At first, you can check a status of the run microservice:

```python
>>> import requests
>>> resp = requests.get('http://localhost:8010/ready')  # check the microservice status
>>> print(resp.status_code)  # print the status (if it equals to 200, then all right)
200
```

Then you can generate queries to recognize named entities in your data. For example, you can send a single text only:

```python
>>> simple_text = "Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов."
>>> resp = requests.post('http://localhost:8010/recognize', json=simple_text)
>>> print(resp.status_code)
200
>>> data = resp.json()
>>> for cur_key in data: print(cur_key, data[cur_key])
...
ners [[67, 73, 'COUNTRY'], [74, 87, 'PERSON'], [35, 73, 'PROFESSION']]
text Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.
```

Also, you can send a list of multiple texts:

```python
>>> some_text_list = [ \
    "Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.", \
    "Другим новичком в правительстве столицы стал новый заместитель Сергея Собянина по взаимодействию со СМИ - 48-летний генеральный директор Российской газеты Александр Горбенко." \
]
>>> resp = requests.post('http://localhost:8010/recognize', json=some_text_list)
>>> print(resp.status_code)
200
>>> data = resp.json()
>>> for it in data: print(it['text'], '\n', it['ners'], '\n')
...
Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.
 [[67, 73, 'COUNTRY'], [74, 87, 'PERSON'], [35, 73, 'PROFESSION']]

Другим новичком в правительстве столицы стал новый заместитель Сергея Собянина по взаимодействию со СМИ - 48-летний генеральный директор Российской газеты Александр Горбенко.
 [[106, 115, 'AGE'], [137, 147, 'COUNTRY'], [18, 39, 'ORGANIZATION'], [137, 154, 'ORGANIZATION'], [63, 78, 'PERSON'], [155, 173, 'PERSON'], [51, 103, 'PROFESSION'], [116, 154, 'PROFESSION']]
```

At last, you can send a list of special dictionaries, each of them describes a single text (the "text" key in the dictionary) with some additional attributes. All of these attributes will be saved in the response, and the "ners" key will be added:

```python
>>> some_data = [
    {"id": 1, "text": "Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов."},
    {"id": 2, "additional": "some", "text": "Другим новичком в правительстве столицы стал новый заместитель Сергея Собянина по взаимодействию со СМИ - 48-летний генеральный директор Российской газеты Александр Горбенко."} \
]
>>> resp = requests.post('http://localhost:8010/recognize', json=some_data)
>>> print(resp.status_code)
200
>>> data = resp.json()
>>> for it in data: print('\n'.join([f'{cur}: {it[cur]}' for cur in it.keys()]), '\n')
...
id: 1
ners: [[67, 73, 'COUNTRY'], [74, 87, 'PERSON'], [35, 73, 'PROFESSION']]
text: Главным борцом с пробками назначен заместитель министра транспорта России Николай Лямов.

additional: some
id: 2
ners: [[106, 115, 'AGE'], [137, 147, 'COUNTRY'], [18, 39, 'ORGANIZATION'], [137, 154, 'ORGANIZATION'], [63, 78, 'PERSON'], [155, 173, 'PERSON'], [51, 103, 'PROFESSION'], [116, 154, 'PROFESSION']]
text: Другим новичком в правительстве столицы стал новый заместитель Сергея Собянина по взаимодействию со СМИ - 48-летний генеральный директор Российской газеты Александр Горбенко.
```

Each recognized named entity is described as a tuple of three elements:

- the entity first character index in the analyzed text;
- index of the character following the entity last character in the analyzed text;
- the class name of this entity.

## Roadmap

1. The algorithm recognizes nested entities of different entity classes, but it does not recognize nested entities of same entity class. For example, the phrase "*Центральный комитет Коммунистического союза молодёжи Китая*" (in English, "*the Central Committee of the Communist Youth League of China*") describes the organization, and also it contains three nested organizations too - they are nested entities of same entity class. Therefore, recognition of nested entities of the same class will be implemented (for example, using a special syntactical-based postprocessing).

2. The model quality will be improved using more sophisticated hierarchical multitask learning.

## Citation

If you want to cite this project you can use this:

```text
@article{bondarenko2022coner,
  title   = {Contrastive fine-tuning to improve generalization in deep NER},
  author  = {Bondarenko, Ivan},
  doi     = {10.28995/2075-7182-2022-21-70-80},
  journal = {Komp'juternaja Lingvistika i Intellektual'nye Tehnologii},
  volume  = {21},
  year    = {2022}
}
```

## Contact

Ivan Bondarenko - [@Bond_005](https://t.me/Bond_005) - [bond005@yandex.ru](mailto:bond005@yandex.ru)

## Acknowledgment

This project was developed as part of a more fundamental project to create an open source system for automatic transcription and semantic analysis of audio recordings of interviews  in Russian. Many journalists, sociologist and other specialists need to prepare the interview manually, and automatization can help their.

The [Foundation for Assistance to Small Innovative Enterprises](https://fasie.ru/upload/docs/Buklet_FASIE_21_bez_Afr_www.pdf) which is Russian governmental non-profit organization supports an unique program to build free and open-source artificial intelligence systems. This programs is known as "Code - Artificial Intelligence" (see https://fasie.ru/press/fund/kod-ai/?sphrase_id=114059 in Russian). The abovementioned project was started within the first stage of the "Code - Artificial Intelligence" program. You can see the first-stage winners list on this web-page: https://fasie.ru/competitions/kod-ai-results (in Russian).

Therefore, I thank The Foundation for Assistance to Small Innovative Enterprises for this support.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
