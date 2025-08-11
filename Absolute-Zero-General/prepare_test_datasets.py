#!/usr/bin/env python3
"""
Prepare simple benchmark datasets for testing benchmark evaluation.
This script creates minimal test datasets that can be used to verify the benchmark evaluation pipeline.
"""

import os
import pandas as pd
from pathlib import Path

def create_test_benchmarks():
    """Create minimal test benchmark datasets."""
    
    validation_dir = Path("./validation_datasets")
    validation_dir.mkdir(exist_ok=True)
    
    print(f"Creating test benchmark datasets in: {validation_dir}")
    
    # Create math test dataset
    math_data = [
        {
            "prompt": [{"role": "user", "content": "What is 2 + 3?"}],
            "answer": "5",
            "data_source": "math_test",
            "extra_info": {"metric": "math_accuracy", "difficulty": "easy"},
            "reward_model": {"style": "rule", "ground_truth": "5"}
        },
        {
            "prompt": [{"role": "user", "content": "Solve: x + 5 = 12"}],
            "answer": "x = 7",
            "data_source": "math_test", 
            "extra_info": {"metric": "math_accuracy", "difficulty": "medium"},
            "reward_model": {"style": "rule", "ground_truth": "x = 7"}
        },
        {
            "prompt": [{"role": "user", "content": "What is the derivative of x^2?"}],
            "answer": "2x",
            "data_source": "math_test",
            "extra_info": {"metric": "math_accuracy", "difficulty": "medium"},
            "reward_model": {"style": "rule", "ground_truth": "2x"}
        }
    ]
    
    math_df = pd.DataFrame(math_data)
    math_path = validation_dir / "math_validation.parquet"
    math_df.to_parquet(math_path)
    print(f"Created {math_path} with {len(math_data)} samples")
    
    # Create GSM8K test dataset
    gsm8k_data = [
        {
            "prompt": [{"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}],
            "answer": "18",
            "data_source": "gsm8k_test",
            "extra_info": {"metric": "math_accuracy", "difficulty": "hard"},
            "reward_model": {"style": "rule", "ground_truth": "18"}
        },
        {
            "prompt": [{"role": "user", "content": "A baker made 121 cakes. He sold 105 of them. How many cakes does he have left?"}],
            "answer": "16",
            "data_source": "gsm8k_test",
            "extra_info": {"metric": "math_accuracy", "difficulty": "easy"},
            "reward_model": {"style": "rule", "ground_truth": "16"}
        }
    ]
    
    gsm8k_df = pd.DataFrame(gsm8k_data)
    gsm8k_path = validation_dir / "gsm8k_validation.parquet"
    gsm8k_df.to_parquet(gsm8k_path)
    print(f"Created {gsm8k_path} with {len(gsm8k_data)} samples")
    
    # Create general QA test dataset
    qa_data = [
        {
            "prompt": [{"role": "user", "content": "What is the capital of France?"}],
            "answer": "Paris",
            "data_source": "qa_test",
            "extra_info": {"metric": "general_accuracy", "difficulty": "easy"},
            "reward_model": {"style": "rule", "ground_truth": "Paris"}
        },
        {
            "prompt": [{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
            "answer": "William Shakespeare",
            "data_source": "qa_test",
            "extra_info": {"metric": "general_accuracy", "difficulty": "easy"},
            "reward_model": {"style": "rule", "ground_truth": "William Shakespeare"}
        },
        {
            "prompt": [{"role": "user", "content": "What is the chemical symbol for gold?"}],
            "answer": "Au",
            "data_source": "qa_test",
            "extra_info": {"metric": "general_accuracy", "difficulty": "medium"},
            "reward_model": {"style": "rule", "ground_truth": "Au"}
        }
    ]
    
    qa_df = pd.DataFrame(qa_data)
    qa_path = validation_dir / "truthfulqa_validation.parquet"
    qa_df.to_parquet(qa_path)
    print(f"Created {qa_path} with {len(qa_data)} samples")
    
    # Create multiple choice test dataset  
    mc_data = [
        {
            "prompt": [{"role": "user", "content": "Which of the following is the largest planet in our solar system? A) Earth B) Mars C) Jupiter D) Saturn"}],
            "answer": "C) Jupiter",
            "data_source": "multiple_choice_test",
            "extra_info": {"metric": "multiple_choice_accuracy", "difficulty": "easy"},
            "reward_model": {"style": "rule", "ground_truth": "C) Jupiter"}
        },
        {
            "prompt": [{"role": "user", "content": "What is 15% of 200? A) 25 B) 30 C) 35 D) 40"}],
            "answer": "B) 30",
            "data_source": "multiple_choice_test",
            "extra_info": {"metric": "multiple_choice_accuracy", "difficulty": "medium"},
            "reward_model": {"style": "rule", "ground_truth": "B) 30"}
        }
    ]
    
    mc_df = pd.DataFrame(mc_data)
    mc_path = validation_dir / "arc_challenge_validation.parquet"
    mc_df.to_parquet(mc_path)
    print(f"Created {mc_path} with {len(mc_data)} samples")
    
    print("\n=== Test Benchmark Datasets Created ===")
    print(f"Directory: {validation_dir.absolute()}")
    print("Files created:")
    for file in validation_dir.glob("*.parquet"):
        print(f"  - {file.name}")
    
    print("\nTo use these datasets:")
    print("1. Update your config to point to this validation directory")
    print("2. Set benchmark_names to include: math, gsm8k, truthfulqa, arc_challenge")
    print("3. Run training with task_type: general")
    
    return validation_dir

if __name__ == "__main__":
    create_test_benchmarks()
