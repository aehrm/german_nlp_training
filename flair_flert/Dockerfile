FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update -y && apt-get -y install \
    python3 \
    python3-pip \
    curl \
    wget \
    unzip \
    git

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on

COPY conll_03_german/deu.train /root/.flair/datasets/conll_03_german/
COPY conll_03_german/deu.testa /root/.flair/datasets/conll_03_german/
COPY conll_03_german/deu.testb /root/.flair/datasets/conll_03_german/

RUN git clone --depth 1 --branch v0.11 https://github.com/flairNLP/flair.git

RUN pip3 install flair
WORKDIR /flair

COPY train_flert.sh train_flert.sh
CMD ["sh", "train_flert.sh"]

