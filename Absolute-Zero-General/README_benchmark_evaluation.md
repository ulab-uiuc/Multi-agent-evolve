# Benchmark Evaluation for General Tasks

This implementation adds support for evaluating models on standard benchmarks like MATH, GSM8K, HellaSwag, etc. when the task type is "general".

## Features

1. **Dataset Preparation**: Automatically loads and formats popular benchmarks
2. **LLM-as-Judge Evaluation**: Uses an external LLM to compare model outputs with ground truth
3. **Integrated Training**: Automatically runs benchmark evaluation during training at specified intervals

## Setup

### 1. Prepare Test Datasets

First, run the dataset preparation script to download and format the benchmark datasets:

```bash
cd /path/to/Multi-agent-evolve/Absolute-Zero-General
python scripts/prepare_test_datasets.py --output_dir ./validation_datasets --datasets math gsm8k hellaswag arc --num_samples 500
```

This will create parquet files in the `validation_datasets` directory:
- `math_test.parquet`
- `gsm8k_test.parquet` 
- `hellaswag_test.parquet`
- `arc_challenge_test.parquet`
- `arc_easy_test.parquet`

### 2. Configure Training

Add the following configuration to your training config:

```yaml
# Task type (set to 'general' to enable benchmark evaluation)
task_type: general

# Benchmark evaluation settings
benchmark_validation_dir: "./validation_datasets"
benchmark_names: ["math", "gsm8k", "hellaswag", "arc_challenge"]
benchmark_eval_model: "meta/llama-3.1-405b-instruct"
benchmark_evaluation_frequency: 100  # Evaluate every 100 steps
benchmark_max_samples: 500  # Limit samples per benchmark for faster evaluation
```

### 3. Usage in Training

The benchmark evaluation will automatically run when:
- `task_type` is set to "general"
- Benchmark datasets are available in the specified directory
- The current step is a multiple of `benchmark_evaluation_frequency`

## Supported Benchmarks

### Mathematics
- **MATH**: High school competition mathematics problems
- **GSM8K**: Grade school math word problems

### Reasoning
- **HellaSwag**: Commonsense reasoning tasks
- **ARC-Challenge/Easy**: Science reasoning questions
- **MMLU**: Massive Multitask Language Understanding (multiple subjects)

### Truthfulness
- **TruthfulQA**: Evaluates model truthfulness and honesty

## Evaluation Metrics

The evaluation uses LLM-as-a-judge with different prompts based on the benchmark type. The LLM provides a binary TRUE/FALSE evaluation for each answer:

1. **Mathematical Accuracy**: Compares numerical results and mathematical reasoning
2. **Multiple Choice Accuracy**: Checks if the correct choice letter is selected
3. **Truthfulness Accuracy**: Evaluates factual accuracy and truthfulness
4. **General Accuracy**: Overall correctness assessment

Each evaluation returns either:
- **1.0** (TRUE): The answer is correct
- **0.0** (FALSE): The answer is incorrect

The final accuracy metrics are calculated as the percentage of correct answers for each benchmark and overall.

## File Structure

```
absolute_zero_reasoner/
├── rewards/
│   └── reward_managers.py (contains BenchmarkEvaluationRewardManager)
├── utils/
│   └── benchmark_config.py (configuration management)
├── trainer/ppo/
│   └── reason_rl_ray_trainer.py (modified trainer with benchmark support)
└── scripts/
    └── prepare_test_datasets.py (dataset preparation script)
```

## Customization

### Adding New Benchmarks

1. Add a new loading function in `scripts/prepare_test_datasets.py`:
```python
def load_my_benchmark(split: str = "test", num_samples: int = None) -> List[Dict]:
    # Load your dataset
    return data
```

2. Add it to the main function and update the choices

3. Update `benchmark_config.py` to include metadata about your benchmark

### Custom Evaluation Logic

You can modify the `BenchmarkEvaluationRewardManager` class to:
- Change the LLM evaluation prompts
- Add custom answer extraction logic
- Implement different scoring mechanisms

## Example Output

During training, you'll see benchmark evaluation results like:

```
============ Benchmark Evaluation ============
Sample 1/100 | Source: math, Correct: ✓
Sample 2/100 | Source: gsm8k, Correct: ✗
Sample 3/100 | Source: hellaswag, Correct: ✓
...
Evaluation Complete | Overall Accuracy: 0.760 (76/100) ✓

Validation Results (Step 100):
┌─────────────────────────────────┬─────────┐
│ Metric                          │ Value   │
├─────────────────────────────────┼─────────┤
│ val/benchmark_accuracy/overall  │ 0.7600  │
│ val/benchmark_accuracy/math     │ 0.8100  │
│ val/benchmark_accuracy/gsm8k    │ 0.7200  │
│ val/benchmark_accuracy/hellaswag│ 0.7300  │
└─────────────────────────────────┴─────────┘
```

This provides comprehensive evaluation across multiple domains to assess the model's general capabilities during training.
