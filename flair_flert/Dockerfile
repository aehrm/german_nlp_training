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

# adjust if needed, e.g. via --build-arg=USER=anton
ARG USER=evaluator
ARG UID=""

RUN if [ -z "${UID}" ]; then adduser ${USER} --home /docker_home --disabled-password --gecos ""; \
        else adduser ${USER} --uid ${UID} --home /docker_home --disabled-password --gecos ""; fi

WORKDIR /docker_home
RUN chown -R ${USER}:${USER} ./
USER ${USER}

COPY conll_03_german/deu.train /docker_home/.flair/datasets/conll_03_german/
COPY conll_03_german/deu.testa /docker_home/.flair/datasets/conll_03_german/
COPY conll_03_german/deu.testb /docker_home/.flair/datasets/conll_03_german/


RUN git clone --depth 1 --branch v0.11 https://github.com/flairNLP/flair.git

RUN pip3 install flair
WORKDIR /docker_home/flair

COPY train_flert.sh train_flert.sh
USER root
RUN chown -R ${USER}:${USER} train_flert.sh
USER ${USER}
CMD ["sh", "train_flert.sh"]

