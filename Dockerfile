FROM python:3.7
MAINTAINER Ivan Bondarenko <i.bondarenko@g.nsu.ru>

RUN apt-get update && \
    yes | apt-get install apt-utils && \
    yes | apt-get install -y gcc && \
    yes | apt-get install -y make && \
    yes | apt-get install -y apt-transport-https && \
    yes | apt-get install -y build-essential && \
    yes | apt-get install git g++ autoconf-archive make libtool && \
    yes | apt-get install python-setuptools python-dev && \
    yes | apt-get install python3-setuptools python3-dev && \
    yes | apt-get install vim && \
    yes | apt-get install libbz2-dev

RUN python3 --version
RUN pip3 --version

RUN mkdir /usr/src/runne_contrastive_ner

COPY ./server.py /usr/src/runne_contrastive_ner/server.py
COPY ./download_model.py /usr/src/runne_contrastive_ner/download_model.py
COPY ./requirements.txt /usr/src/runne_contrastive_ner/requirements.txt
COPY ./neural_network/ /usr/src/runne_contrastive_ner/neural_network/
COPY ./data_processing/ /usr/src/runne_contrastive_ner/data_processing/
COPY ./io_utils/ /usr/src/runne_contrastive_ner/io_utils/
COPY ./trainset_building/ /usr/src/runne_contrastive_ner/trainset_building/
COPY ./models/ /usr/src/runne_contrastive_ner/models/

WORKDIR /usr/src/runne_contrastive_ner

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 download_model.py

ENTRYPOINT ["python3", "server.py"]
