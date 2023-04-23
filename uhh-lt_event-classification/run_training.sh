#!/bin/bash

MODEL=${MODEL:-deepset/gbert-large}

python main.py pretrained_model=${MODEL} 
