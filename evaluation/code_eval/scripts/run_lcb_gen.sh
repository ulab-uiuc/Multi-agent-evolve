#!/bin/bash

# Default values
MODEL_PATH="andrewzh/Absolute_Zero_Reasoner-Coder-7b"
CUDA_GPU_ID="0"
NUM_GPUS=1
BATCH_SIZE=128
N=1
TEMPERATURE=0.0
TOP_P=1.0
MAX_TOKENS=8096

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -g|--gpu)
      CUDA_GPU_ID="$2"
      shift 2
      ;;
    -n|--n)
      N="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -p|--top_p)
      TOP_P="$2"
      shift 2
      ;;
    -b|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -k|--max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

cd evaluation/code_eval/coding/LiveCodeBench

# Run LiveCodeBench with the AZR template and a local model
CUDA_VISIBLE_DEVICES=$CUDA_GPU_ID python -m lcb_runner.runner.main \
  --model $MODEL_PATH \
  --trust_remote_code \
  --scenario codegeneration \
  --release_version release_v5 \
  --tensor_parallel_size $NUM_GPUS \
  --use_cache \
  --n $N \
  --temperature $TEMPERATURE \
  --max_tokens $MAX_TOKENS \
  --custom_output_save_name $MODEL_PATH \
  --top_p $TOP_P \
  --timeout 60 \
  --evaluate --continue_existing --continue_existing_with_eval
