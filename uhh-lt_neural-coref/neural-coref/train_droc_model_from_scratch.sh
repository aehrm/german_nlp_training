#!/bin/bash
set -e

SEGMENT_LENGTH=${SEGMENT_LENGTH:-512}
MODEL=${MODEL:-deepset/gbert-large}
MODEL_TYPE=${MODEL_TYPE:-bert}

# preprocess splits
python3 preprocess.py --input_dir data/ --output_dir data/tuba10 --seg_len ${SEGMENT_LENGTH} \
  --language german --tokenizer_name ${MODEL} --input_suffix tuba10_gold_conll --input_format conll-2012 \
  --model_type ${MODEL_TYPE}
#python3 preprocess.py --input_dir data/droc_full --output_dir data/droc_full --seg_len ${SEGMENT_LENGTH} \
#  --language german --tokenizer_name ${MODEL} --input_suffix droc_gold_conll --input_format conll-2012 \
#  --model_type ${MODEL_TYPE}


# append to experiments.conf
cat <<EOT >> experiments.conf
tuba10_custom = \${tuba10}{
  conll_eval_path = \${base.data_dir}/dev.german.tuba10_gold_conll
  conll_test_path = \${base.data_dir}/test.german.tuba10_gold_conll
  data_dir = \${base.data_dir}/tuba10
  bert_tokenizer_name = ${MODEL}
  bert_pretrained_name_or_path = ${MODEL}
  long_doc_strategy = split
  log_root = ./output/
}
EOT

cp data/reference-coreference-scorers data/tuba10 -r

# pre-train on TüBa-D/Z
echo "performing pre-training on the TüBa-D/Z news dataset using ${MODEL}"
python3 run.py tuba10_custom 0
