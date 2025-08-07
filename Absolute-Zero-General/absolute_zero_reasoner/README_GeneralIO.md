# GeneralIO PPO Training Setup

This document explains how to use the updated `main_azr_ppo.py` to train models on general tasks using the GeneralIORewardManager and GeneralIORayPPOTrainer.

## Key Changes Made

### 1. Import Additions
```python
from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager, GeneralIORewardManager
```

### 2. Task Type Detection
The system now automatically detects the task type from configuration:
```python
task_type = getattr(config.azr, 'task_type', 'code')  # defaults to 'code' for backward compatibility
```

### 3. Conditional Reward Manager Creation
- **For general tasks**: Uses `GeneralIORewardManager` with LLM-as-a-judge API
- **For code tasks**: Uses `CodeIORewardManager` (original behavior)

### 4. Conditional Trainer Selection
- **For general tasks**: Uses `GeneralIORayPPOTrainer`
- **For code tasks**: Uses `CodeIORayPPOTrainer`

## Configuration for General Tasks

### Required Configuration Fields

```yaml
azr:
  task_type: general  # KEY: Set this to 'general' to enable GeneralIO mode
  problem_types:
    - general
  
reward_fn:
  # LLM API configuration for reward computation
  llm_model_name: "meta/llama-3.1-405b-instruct"
  temperature: 0.7
  max_tokens: 1000
  top_p: 0.95
  stream: true

azr:
  reward:
    generation_reward_config:
      format_reward: false  # No format checking for general tasks
    n_samples: 3  # Number of samples for difficulty estimation
```

### Example Usage

1. **Use the provided config file**:
   ```bash
   python main_azr_ppo.py --config-name=azr_ppo_trainer_general
   ```

2. **Run the example script**:
   ```bash
   python run_general_io_example.py
   ```

## How It Works

### For Generation Tasks (`gen_general`)
1. **LLM Judge Score**: External LLM evaluates the generated response
2. **Difficulty Score**: Actor model generates multiple responses, evaluated by LLM, difficulty = 1 - average_score
3. **Final Reward**: `0.5 * llm_judge_score + 0.5 * difficulty_score`

### For Prediction Tasks (`pred_general`)
1. **Direct LLM Score**: Uses LLM-as-a-judge score directly as the reward

### Training Pipeline
1. **Initialization**: Seeds with 100 examples from MATH, HumanEval, and MT-Bench datasets
2. **Generation**: Model generates responses to questions
3. **Evaluation**: Responses evaluated using LLM-as-a-judge
4. **Dataset Expansion**: Valid responses added to training dataset
5. **Training**: PPO training on the expanded dataset

## Key Features

### 1. **Automatic Dataset Seeding**
- Automatically loads seed examples from popular datasets
- Creates initial training data in the correct format

### 2. **LLM-as-a-Judge Integration**
- Uses external LLM API for reward computation
- Configurable model, temperature, and other parameters

### 3. **Dynamic Dataset Expansion**
- Training dataset grows with high-quality generated examples
- Maintains dataset size limits to prevent memory issues

### 4. **Flexible Configuration**
- Backward compatible with existing code tasks
- Easy switching between task types via configuration

## Files Modified

1. **`main_azr_ppo.py`**: Main training script with conditional logic
2. **`configs/azr_ppo_trainer_general.yaml`**: Configuration template for general tasks
3. **`run_general_io_example.py`**: Example script demonstrating usage

## Important Notes

1. **API Access**: Ensure you have access to the LLM API specified in the configuration
2. **Memory Management**: The system automatically limits dataset size to prevent memory issues
3. **Backward Compatibility**: Existing code task configurations will continue to work unchanged
4. **Model Path**: Update the model path in the configuration to point to your trained model

## Troubleshooting

### Common Issues

1. **Import Errors**: These are expected for project dependencies (verl, etc.) and don't affect functionality
2. **API Access**: Ensure your LLM API credentials are properly configured
3. **Memory Issues**: Reduce batch size or dataset limits if you encounter memory problems
4. **Path Issues**: Ensure model paths and output directories are correctly configured

### Debug Mode

Enable debug mode in the configuration for detailed logging:
```yaml
trainer:
  debug: true
  debug_port: 5680
```
