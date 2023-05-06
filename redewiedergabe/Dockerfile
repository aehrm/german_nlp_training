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

#RUN wget 'https://github.com/redewiedergabe/corpus/blob/master/data/additional/data_konvens-paper-2020.zip' -O data.zip
#RUN unzip data.zip


COPY data_konvens-paper-2020 data_konvens-paper-2020
COPY tagger tagger
RUN pip3 install -r requirements.txt

COPY train_model.py train_model.py
USER root
RUN chown -R ${USER}:${USER} data_konvens-paper-2020 tagger train_model.py
USER ${USER}
CMD ["python3", "train_model.py"]
