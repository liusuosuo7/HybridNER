#!/bin/bash

# Example script for running LinkNER with combination functionality
# This script shows how to use the enhanced spanNER with model combination

# Basic configuration
DATANAME="conll03"
DATA_DIR="data/conll03"
RESULTS_DIR="results/"
BERT_CONFIG_DIR="bert_large_uncased"

# Combination-specific configuration
USE_COMBINATION=true
COMBINATION_METHOD="voting_weightByOverallF1"  # Options: voting_majority, voting_weightByOverallF1, voting_weightByCategotyF1, voting_spanPred_onlyScore
COMBINATION_RESULTS_DIR="results/conll03"
COMBINATION_PROB_FILE="conll03_spanner_prob.pkl"
COMBINATION_STANDARD_FILE="conll03_CcnnWglove_lstmCrf_72102467_test_9088.txt"

# List of model result files to combine (example from spanNER combination)
COMBINATION_MODELS=(
    "conll03_CflairWnon_lstmCrf_1_test_9241.txt"
    "conll03_CbertWglove_lstmCrf_1_test_9201.txt"
    "conll03_CbertWnon_lstmCrf_1_test_9246.txt"
    "conll03_CflairWglove_lstmCrf_1_test_9302.txt"
    "conll03_CelmoWglove_lstmCrf_95803618_test_9211.txt"
    "conll03_spanNER_generic_test_9157.txt"
    "conll03_spanNER_lenDecode_9228.txt"
)

# Entity classes for CoNLL-03
COMBINATION_CLASSES=("ORG" "PER" "LOC" "MISC")

echo "Running LinkNER with Enhanced SpanNER Combination Functionality"
echo "================================================================"
echo "Dataset: $DATANAME"
echo "Combination Method: $COMBINATION_METHOD"
echo "Number of models to combine: ${#COMBINATION_MODELS[@]}"
echo "Entity classes: ${COMBINATION_CLASSES[*]}"
echo ""

# Training with combination
echo "Training phase..."
python run_localModel.py \
    --state train \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --results_dir $RESULTS_DIR \
    --bert_config_dir $BERT_CONFIG_DIR \
    --use_combination $USE_COMBINATION \
    --combination_method $COMBINATION_METHOD \
    --combination_models ${COMBINATION_MODELS[*]} \
    --combination_results_dir $COMBINATION_RESULTS_DIR \
    --combination_prob_file $COMBINATION_PROB_FILE \
    --combination_standard_file $COMBINATION_STANDARD_FILE \
    --combination_classes ${COMBINATION_CLASSES[*]} \
    --n_class 5 \
    --max_spanLen 5 \
    --batch_size 10 \
    --lr 4e-6 \
    --iteration 10

echo ""
echo "Training completed!"

# Inference with combination
echo "Inference phase..."
python run_localModel.py \
    --state inference \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --results_dir $RESULTS_DIR \
    --bert_config_dir $BERT_CONFIG_DIR \
    --use_combination $USE_COMBINATION \
    --combination_method $COMBINATION_METHOD \
    --combination_models ${COMBINATION_MODELS[*]} \
    --combination_results_dir $COMBINATION_RESULTS_DIR \
    --combination_prob_file $COMBINATION_PROB_FILE \
    --combination_standard_file $COMBINATION_STANDARD_FILE \
    --combination_classes ${COMBINATION_CLASSES[*]} \
    --n_class 5 \
    --max_spanLen 5 \
    --batch_size 10

echo ""
echo "Inference completed!"

# Linking to LLM with combination
echo "Linking to LLM phase..."
python run_localModel.py \
    --state link \
    --dataname $DATANAME \
    --data_dir $DATA_DIR \
    --results_dir $RESULTS_DIR \
    --bert_config_dir $BERT_CONFIG_DIR \
    --use_combination $USE_COMBINATION \
    --combination_method $COMBINATION_METHOD \
    --combination_models ${COMBINATION_MODELS[*]} \
    --combination_results_dir $COMBINATION_RESULTS_DIR \
    --combination_prob_file $COMBINATION_PROB_FILE \
    --combination_standard_file $COMBINATION_STANDARD_FILE \
    --combination_classes ${COMBINATION_CLASSES[*]} \
    --selectShot_dir "data/conll03/spanSelect.dev" \
    --linkSave_dir "results/" \
    --threshold 0.4 \
    --n_class 5 \
    --max_spanLen 5

echo ""
echo "All phases completed! 🎉"
echo ""
echo "The enhanced LinkNER now uses spanNER with combination functionality!"
echo "This allows it to combine predictions from multiple models for better performance."