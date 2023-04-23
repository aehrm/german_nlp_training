#!/bin/bash
set -e

SEGMENT_LENGTH=${SEGMENT_LENGTH:-512}
MODEL=${MODEL:-deepset/gbert-large}
MODEL_TYPE=${MODEL_TYPE:-bert}

# preprocess splits
python3 preprocess.py --input_dir data/ --output_dir data/tuba10 --seg_len ${SEGMENT_LENGTH} \
  --language german --tokenizer_name ${MODEL} --input_suffix tuba10_gold_conll --input_format conll-2012 \
  --model_type ${MODEL_TYPE}
python3 preprocess.py --input_dir data/droc_full --output_dir data/droc_full --seg_len ${SEGMENT_LENGTH} \
  --language german --tokenizer_name ${MODEL} --input_suffix droc_gold_conll --input_format conll-2012 \
  --model_type ${MODEL_TYPE}

TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)
mkdir /output/$TIMESTAMP


# append to experiments.conf
cat <<EOT >> experiments.conf
tuba10_custom = \${tuba10}{
  conll_eval_path = \${base.data_dir}/dev.german.tuba10_gold_conll
  conll_test_path = \${base.data_dir}/test.german.tuba10_gold_conll
  data_dir = \${base.data_dir}/tuba10
  bert_tokenizer_name = ${MODEL}
  bert_pretrained_name_or_path = ${MODEL}
  max_segment_len = ${SEGMENT_LENGTH}
  long_doc_strategy = split
  log_root = /output/$TIMESTAMP/
  use_features = true
  use_segment_distance = false
  num_epochs = 10
}

droc_incremental_no_segment_distance_custom = \${droc_incremental_no_segment_distance}{
  bert_tokenizer_name = ${MODEL}
  bert_pretrained_name_or_path = ${MODEL}
  max_segment_len = ${SEGMENT_LENGTH}
  log_root = /output/$TIMESTAMP/
  use_features = true
  use_segment_distance = false
  num_epochs = 10
}
EOT

cp data/reference-coreference-scorers data/tuba10 -r

# fine-tune on TüBa-D/Z
echo "performing fine-tuning on the TüBa-D/Z news dataset using ${MODEL}"
python3 run.py tuba10_custom 0

FINE_TUNED_MODEL=$(ls /output/$TIMESTAMP/tuba10_custom/model_*.bin)

# fine-tune on DROC, continuing on previous state
echo "performing c2f fine-tuning on the DROC dataset using ${FINE_TUNED_MODEL}"
python3 run.py droc_incremental_no_segment_distance_custom 0 --model ${FINE_TUNED_MODEL}
