#!/usr/bin/env python3
"""
Script to extract and save 1000 fixed random samples from FusionBench dataset.
This will create a JSON file with fixed data to replace the random sampling in training.
"""

import json
import random
from datasets import load_dataset
from pathlib import Path

def extract_fixed_fusionbench_data(num_samples=1000, seed=42, output_path=None, default_data_dir=None):
    """
    Extract and save fixed random samples from FusionBench dataset
    
    Args:
        num_samples (int): Number of samples to extract (default: 1000)
        seed (int): Random seed for reproducibility (default: 42)
        output_path (str): Path to save the JSON file (if None, will use default_data_dir)
        default_data_dir (str): Default data directory path (if None, will use "./data/fixed_datasets")
    """
    print(f"Extracting {num_samples} fixed samples from FusionBench dataset...")
    
    # Set default paths
    if default_data_dir is None:
        default_data_dir = Path("./data/fixed_datasets")
    else:
        default_data_dir = Path(default_data_dir)
    
    if output_path is None:
        output_path = default_data_dir / "fixed_fusionbench_1000.json"
    else:
        output_path = Path(output_path)
    
    # Create default data directory
    default_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using default data directory: {default_data_dir}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    try:
        # Load FusionBench dataset
        print("Loading FusionBench dataset...")
        general_data = load_dataset("ulab-ai/FusionBench", "train", split="data")
        print(f"Loaded {len(general_data)} total samples from FusionBench")
        
        # Sample fixed random data
        all_data = list(general_data)
        if len(all_data) < num_samples:
            print(f"Warning: Dataset has only {len(all_data)} samples, using all available")
            num_samples = len(all_data)
        
        selected_samples = random.sample(all_data, k=num_samples)
        print(f"Selected {len(selected_samples)} samples")
        
        # Convert to our format and create both examples and example_pairs
        examples = []
        example_pairs = []
        sample_map = {}
        benchmark_count = {}
        
        for idx, item in enumerate(selected_samples):
            question = item["query"]
            answer = item["ground_truth"]
            io_prompt = f"{question}"
            chosen_references = []

            origin = item['task_name']
            if origin not in benchmark_count.keys():
                benchmark_count[origin] = 0
            benchmark_count[origin] += 1

            if "Let's think step by step." in question:
                _question = question.replace("Let's think step by step.", "").strip()
                sample_map[_question] = 1
            else:
                sample_map[question] = 1
            
            # Create io_item (for examples)
            io_item = {
                "data_source": 'gen_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": question,
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': 'train',
                    'index': idx,
                    'metric': 'gen_general',
                    'chosen_references': chosen_references,
                }
            }
            
            # Create io_item_pair (for example_pairs)
            io_item_pair = {
                "data_source": 'gen_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": question,
                "answer": answer,
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': 'train',
                    'index': idx,
                    'metric': 'gen_general',
                    'chosen_references': chosen_references,
                }
            }
            
            examples.append(io_item)
            example_pairs.append(io_item_pair)
        
        # Create the data structure to save
        fixed_data = {
            "metadata": {
                "num_samples": len(examples),
                "source": "FusionBench",
                "dataset_config": "ulab-ai/FusionBench",
                "split": "train",
                "seed": seed,
                "default_data_dir": str(default_data_dir),
                "extraction_date": str(Path(__file__).stat().st_mtime),
                "description": "Fixed 1000 samples from FusionBench for reproducible training"
            },
            "examples": examples,
            "example_pairs": example_pairs
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved {len(examples)} examples and {len(example_pairs)} example pairs to {output_path}")
        print(f"Default data directory: {default_data_dir}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Show a few sample questions for verification
        print("\nFirst 3 sample questions:")
        for i, example in enumerate(examples[:3]):
            print(f"{i+1}. {example['question'][:100]}...")
            print(f"   Answer: {example['reward_model']['ground_truth'][:50]}...")
            print()
        
        for k in benchmark_count.keys():
            print("Benchmark: ", k)
            print("Count: ", benchmark_count[k])

        print("Different samples count:", len(sample_map.keys()))

        return output_path
        
    except Exception as e:
        print(f"Error extracting FusionBench data: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract fixed FusionBench samples")
    parser.add_argument("--num_samples", type=int, default=1000, 
                       help="Number of samples to extract (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for JSON file (default: use default_data_dir)")
    parser.add_argument("--default_data_dir", type=str, default=None,
                       help="Default data directory (default: ./data/fixed_datasets)")
    
    args = parser.parse_args()
    
    output_path = extract_fixed_fusionbench_data(
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output,
        default_data_dir=args.default_data_dir
    )
    
    print(f"\nFixed FusionBench data successfully created at: {output_path}")
    print("You can now run the trainer and it will automatically find the data.")
    print("The trainer will look in the default_data_dir first, then fall back to other locations.")
