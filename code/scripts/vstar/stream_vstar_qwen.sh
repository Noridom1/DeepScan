#!/bin/bash

METHOD=${1:-"clean"}
DEBUG=${2:-"False"}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="/root/autodl-tmp/Qwen3-VL-8B-Instruct"
NAME="Qwen3-VL-8B-Instruct"

DATASET="vstar"
DATASET_FORMAT="vstar"
SPLIT="test_questions"

mkdir -p /root/autodl-tmp/playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD

# change module to ccot module
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python code/src/run.py \
        --model-path $CKPT \
        --question-file /root/autodl-tmp/playground/data/eval/$DATASET/$SPLIT.tsv \
        --answers-file eval/$DATASET/deepscan/$CKPT/$SPLIT/$METHOD/${CHUNKS}_${IDX}_${NAME}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --image-size 5000 \
        --temperature 0 \
        --method_name $DATASET_FORMAT.$METHOD \
        --debug $DEBUG \
        &
done

wait

output_file= eval/$DATASET/deepscan/$CKPT/$SPLIT/$METHOD/${NAME}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat eval/$DATASET/deepscan/$CKPT/$SPLIT/$METHOD/${CHUNKS}_${IDX}_${NAME}.jsonl >> "$output_file"
done

# Eval
echo $METHOD $DATASET $SPLIT
python /root/autodl-tmp/playground/data/eval/$DATASET/eval.py \
    --path $output_file

