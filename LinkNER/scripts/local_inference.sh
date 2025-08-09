#!/bin/bash

cd /mnt/nas-alinlp/qinghui.zz/projects/linkner/A_Linkner
DATA_DIR="Your data directory path"
STEP1_TRAINED_MODEL="Your trained model path"
TEST_DIR="Your results directory path"
BERT_CONFIG_DIR="Your BERT config directory path"

# spanner test
python main.py \
    --state inference \
    --data_dir "$DATA_DIR" \
    --inference_model "$STEP1_TRAINED_MODEL" \
    --uncertainty_type confidence \
    --results_dir "$TEST_DIR" \
    --bert_config_dir "$BERT_CONFIG_DIR" \
    --max_spanLen 5 \
    --n_class 5 \
    --etrans_func softmax \
    --loss ce \
    --test_mode ori \
    # --use_combiner true \
    # --comb_files "/path/to/other_model1.jsonl,/path/to/other_model2.jsonl" \
    # --comb_method weighted \
    # --comb_weights "1.0,0.8,0.7"  # base, model1, model2

# # ener test
# python main.py \
#     --state inference \
#     --data_dir "$DATA_DIR" \
#     --inference_model "$STEP1_TRAINED_MODEL" \
#     --uncertainty_type ener \
#     --results_dir "$TEST_DIR" \
#     --bert_config_dir "$BERT_CONFIG_DIR" \
#     --max_spanLen 5 \
#     --n_class 5 \
#     --etrans_func exp \
#     --loss edl \
#     --test_mode ori
