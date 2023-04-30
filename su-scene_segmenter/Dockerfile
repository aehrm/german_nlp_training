FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

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

COPY segmenter segmenter
USER root
RUN chown -R ${USER}:${USER} segmenter
USER ${USER}

WORKDIR /docker_home/segmenter/code
RUN python3 -m pip install -U pip setuptools wheel
RUN python3 -m pip install --upgrade cython
RUN python3 -m pip install numpy
RUN python3 -m pip install -r ../requirements.txt

CMD ["sh", "scripts/train.sh"]

