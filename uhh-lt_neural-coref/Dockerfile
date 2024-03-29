FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update -y && apt-get -y install \
    cuda-drivers \
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


USER ${USER}
ENV PATH="${PATH}:/docker_home/.local/bin"

WORKDIR /neural-coref


COPY neural-coref/requirements.txt requirements.txt
USER root
RUN chown -R ${USER}:${USER} requirements.txt
USER ${USER}
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 torch==1.11.0+cu113
RUN python3 -m pip install -r requirements.txt

COPY neural-coref/* ./
USER root
RUN chown -R ${USER}:${USER} ./
USER ${USER}

RUN git clone https://gitlab2.informatik.uni-wuerzburg.de/kallimachos/DROC-Release.git
COPY tuebadz10/ tuebadz10
USER root
RUN chown -R ${USER}:${USER} tuebadz10
USER ${USER}

# make splits
RUN mkdir data
RUN mkdir data/droc_full
RUN python3 split_droc.py --type-system-xml ./DROC-Release/droc/src/main/resources/CorefTypeSystem.xml ./DROC-Release/droc/DROC-xmi data/droc_full/german.droc_gold_conll
RUN unzip ./tuebadz10/tuebadz-10.0-conll2011.txt.zip -d ./tuebadz10
RUN bash -c 'cd data/; python3 ../split_tuebadz_10.py  ../tuebadz10/tuebadz-10.0-conll2011.txt'
RUN bash -c 'cd data/; wget "http://conll.cemantix.org/download/reference-coreference-scorers.v8.01.tar.gz"; tar xvf reference-coreference-scorers.v8.01.tar.gz'

COPY train_droc_model_from_scratch.sh .
USER root
RUN chown -R ${USER}:${USER} train_droc_model_from_scratch.sh
USER ${USER}
CMD ["sh", "train_droc_model_from_scratch.sh"]

