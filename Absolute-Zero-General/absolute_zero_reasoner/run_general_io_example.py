#!/usr/bin/env python3
"""
Example script to run GeneralIO PPO training with the updated main_azr_ppo.py

This script demonstrates how to train a model on general tasks using the 
GeneralIORewardManager and GeneralIORayPPOTrainer.

Usage:
    python run_general_io_example.py

Make sure to:
1. Set up your environment with the required dependencies (verl, ray, etc.)
2. Configure your model path and other settings in the config file
3. Ensure you have access to the LLM API for reward computation
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main function
from main_azr_ppo import run_ppo
from omegaconf import OmegaConf

def create_general_io_config():
    """Create a configuration for general IO tasks"""
    
    # Load the general config template
    config_path = project_root / "configs" / "azr_ppo_trainer_general.yaml"
    config = OmegaConf.load(config_path)
    
    # Override some settings for this example
    config.trainer.total_training_steps = 100  # Short run for example
    config.trainer.default_local_dir = "./general_io_outputs"
    config.trainer.experiment_name = "general_io_example"
    
    # Model configuration - adjust to your model
    config.actor_rollout_ref.model.path = "~/models/deepseek-llm-7b-chat"  # Change this to your model path
    
    # Training configuration
    config.data.train_batch_size = 64  # Smaller batch for example
    config.trainer.save_freq = 20
    config.trainer.eval_freq = 10
    config.trainer.log_freq = 5
    
    # Ensure we're using general tasks
    config.azr.task_type = "general"
    config.azr.problem_types = ["general"]
    
    # LLM API configuration for reward computation
    config.reward_fn.llm_model_name = "meta/llama-3.1-405b-instruct"
    config.reward_fn.temperature = 0.7
    config.reward_fn.max_tokens = 1000
    
    return config

def main():
    """Main function to run the general IO training"""
    
    print("=" * 60)
    print("Starting General IO PPO Training Example")
    print("=" * 60)
    
    # Create configuration
    config = create_general_io_config()
    
    print(f"Task type: {config.azr.task_type}")
    print(f"Problem types: {config.azr.problem_types}")
    print(f"Training steps: {config.trainer.total_training_steps}")
    print(f"Batch size: {config.data.train_batch_size}")
    print(f"Output directory: {config.trainer.default_local_dir}")
    
    # Ensure output directory exists
    os.makedirs(config.trainer.default_local_dir, exist_ok=True)
    
    print("\nStarting training...")
    
    try:
        # Run the training
        run_ppo(config)
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
