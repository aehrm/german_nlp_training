FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /


ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update -y && apt-get -y install \
    curl \
    unzip \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get -y install python3.10

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  POETRY_NO_INTERACTION=1

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN curl -sSL https://install.python-poetry.org | python3.10 -
ENV PATH="${PATH}:/root/.local/bin"


WORKDIR /event-classification
COPY event-classification/poetry.lock poetry.lock
COPY event-classification/pyproject.toml pyproject.toml
COPY event-classification/download.sh download.sh

RUN bash ./download.sh
RUN poetry install
RUN poetry add https://github.com/cupy/cupy/releases/download/v10.6.0/cupy_cuda113-10.6.0-cp310-cp310-manylinux1_x86_64.whl

COPY event-classification/**/* ./

ENTRYPOINT ["poetry", "run"]
