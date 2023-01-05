#!/bin/bash

MODEL=${MODEL:-deepset/gbert-large}

cd examples/ner
python3 run_ner.py \
    --dataset_name CONLL_03_GERMAN \
    --model_name_or_path xlm-roberta-base \
    --batch_size 4 \
    --learning_rate 5e-06 \
    --num_epochs 20 \
    --context_size 64 \
    --output_dir ../../output/conll-deu-ner-flert