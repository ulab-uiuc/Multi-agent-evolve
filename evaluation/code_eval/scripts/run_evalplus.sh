#!/bin/bash

export HUMANEVAL_OVERRIDE_PATH=evaluation/code_eval/data/HumanEvalPlus.jsonl
export MBPP_OVERRIDE_PATH=evaluation/code_eval/data/MbppPlus.jsonl

# Set defaults if not specified - fix argument assignments
DATASET=${1:-humaneval}
MODEL=${2:-"andrewzh/Absolute_Zero_Reasoner-Coder-3b"}
GREEDY=${3:-1}
TEMP=${4:-0.8}
TOP_P=${5:-0.9}
N_SAMPLES=${6:-1}

# If greedy mode, force n_samples to 1
if [ "$GREEDY" -eq 1 ]; then
    N_SAMPLES=1
fi

echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Greedy: $GREEDY (1=yes, 0=no)"
echo "Temperature: $TEMP"
echo "Top-P: $TOP_P"
echo "Number of samples: $N_SAMPLES"

# Extract model identifier for output file
MODEL_BASE=$(basename "$MODEL")
echo "Model base: $MODEL_BASE"

# Execute command directly without quoting the arguments
if [ "$GREEDY" -eq 1 ]; then
    evalplus.codegen --model "$MODEL" \
                    --dataset $DATASET \
                    --backend vllm \
                    --trust_remote_code \
                    --greedy
    TEMP_VAL="0.0"
else
    evalplus.codegen --model "$MODEL" \
                    --dataset $DATASET \
                    --backend vllm \
                    --temperature $TEMP \
                    --top-p $TOP_P \
                    --trust_remote_code \
                    --n-samples $N_SAMPLES
    TEMP_VAL="$TEMP"
fi

# The actual output file - use a glob pattern to find the file
echo "Waiting for output file to be generated..."
sleep 2  # Give some time for the file to be created

# Use find to locate the file with a more flexible pattern that matches actual filename format
OUTPUT_FILE=$(find "evalplus_results/${DATASET}" -name "*${MODEL_BASE}_vllm_temp_${TEMP_VAL}.jsonl" ! -name "*.raw.jsonl" -type f | head -n 1)

# Run evaluation with found file
evalplus.evaluate --dataset "$DATASET" \
    --samples "$OUTPUT_FILE" \
    --output_file "evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json" \
    --min-time-limit 10.0 \
    --gt-time-limit-factor 8.0

echo "Evaluation complete. Results saved to evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json"