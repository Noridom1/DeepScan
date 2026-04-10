#!/bin/bash
set -euo pipefail

METHOD=${1:-clean}
DEBUG=${2:-False}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT="/root/autodl-tmp/Qwen3-VL-8B-Instruct"
NAME="$(basename "$CKPT")"

DATASET="vstar"
DATASET_FORMAT="vstar"
SPLIT="test_questions"

REPO_DIR="$(pwd)"
OUT_DIR="$REPO_DIR/eval/$DATASET/deepscan/$NAME/$SPLIT/$METHOD"
QUESTION_FILE="/root/autodl-tmp/playground/data/eval/$DATASET/$SPLIT.tsv"

mkdir -p "$OUT_DIR"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python code/src/run.py \
        --model-path "$CKPT" \
        --question-file "$QUESTION_FILE" \
        --answers-file "$OUT_DIR/${CHUNKS}_${IDX}_${NAME}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --single-pred-prompt \
        --image-size 5000 \
        --temperature 0 \
        --method_name "${DATASET_FORMAT}.${METHOD}" \
        --debug "$DEBUG" \
        &
done

wait

output_file="$OUT_DIR/${NAME}.jsonl"

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUT_DIR/${CHUNKS}_${IDX}_${NAME}.jsonl" >> "$output_file"
done

# Eval
echo "$METHOD $DATASET $SPLIT"
python "/root/autodl-tmp/playground/data/eval/$DATASET/eval.py" \
    --path "$output_file"
