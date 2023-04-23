#!/bin/bash

MODEL=${MODEL:-deepset/gbert-large}

TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
mkdir /output/$TIMESTAMP

cd examples/ner
python3 run_ner.py \
    --dataset_name CONLL_03_GERMAN \
    --model_name_or_path ${MODEL} \
    --batch_size 32 \
    --learning_rate 5e-05 \
    --num_epochs 50 \
    --context_size 64 \
    --output_dir /output/$TIMESTAMP
