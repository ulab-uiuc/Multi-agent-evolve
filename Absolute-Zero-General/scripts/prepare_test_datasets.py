#!/usr/bin/env python3
"""
Script to prepare test datasets for evaluation in general tasks.
Supports MATH, HellaSwag, GSM8K, and other common benchmarks.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Any
import argparse
from pathlib import Path


def load_math_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MATH dataset."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['problem']}"}],
            "ground_truth": item['solution'],
            "answer": item['answer'].split('The answer is')[-1].strip() if 'The answer is' in item['answer'] else item['answer'],
            "data_source": "math",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_gsm8k_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Extract the final answer
        answer = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
        
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['question']}"}],
            "ground_truth": item['answer'],
            "answer": answer,
            "data_source": "gsm8k",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


# def load_hellaswag_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
#     """Load HellaSwag dataset."""
#     dataset = load_dataset("hellaswag", split=split)
    
#     data = []
#     for i, item in enumerate(dataset):
#         if num_samples and i >= num_samples:
#             break
            
#         # Format choices
#         choices = item['endings']
#         choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
#         prompt_text = f"""Complete the following scenario by choosing the most likely continuation:

# Context: {item['ctx']}

# Choices:
# {choice_text}

# Choose the most appropriate continuation (A, B, C, or D):"""
        
#         data.append({
#             "prompt": [{"role": "user", "content": prompt_text}],
#             "ground_truth": chr(65 + int(item['label'])),  # Convert 0,1,2,3 to A,B,C,D
#             "answer": chr(65 + int(item['label'])),
#             "data_source": "hellaswag",
#             "choices": choices,
#             "extra_info": {"metric": "multiple_choice_accuracy"}
#         })
    
#     return data


def load_mmlu_dataset(subject: str = "abstract_algebra", split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MMLU dataset for a specific subject."""
    dataset = load_dataset("cais/mmlu", subject, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = item['choices']
        choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
        prompt_text = f"""Answer the following multiple choice question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": chr(65 + item['answer']),  # Convert 0,1,2,3 to A,B,C,D
            "answer": chr(65 + item['answer']),
            "data_source": f"mmlu_{subject}",
            "subject": subject,
            "choices": choices,
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data


def load_arc_dataset(split: str = "test", challenge: bool = True, num_samples: int = None) -> List[Dict]:
    """Load ARC dataset."""
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = [choice['text'] for choice in item['choices']['text']]
        labels = item['choices']['label']
        choice_text = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
        
        prompt_text = f"""Answer the following science question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer:"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": item['answerKey'],
            "answer": item['answerKey'],
            "data_source": f"arc_{'challenge' if challenge else 'easy'}",
            "choices": choices,
            "choice_labels": labels,
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data


def load_truthfulqa_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load TruthfulQA dataset."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Answer the following question truthfully:\n\n{item['question']}"}],
            "ground_truth": item['best_answer'],
            "answer": item['best_answer'],
            "data_source": "truthfulqa",
            "category": item['category'],
            "extra_info": {"metric": "truthfulness_accuracy"}
        })
    
    return data


def save_dataset_to_parquet(data: List[Dict], output_path: str, dataset_name: str):
    """Save dataset to parquet format."""
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(data)
    output_file = os.path.join(output_path, f"{dataset_name}_test.parquet")
    df.to_parquet(output_file)
    print(f"Saved {len(data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare test datasets for evaluation")
    parser.add_argument("--output_dir", type=str, default="./validation_datasets", 
                       help="Output directory for validation datasets")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to load per dataset (None for all)")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa", "all"],
                       default=["all"], help="Datasets to prepare")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets_to_load = args.datasets
    if "all" in datasets_to_load:
        datasets_to_load = ["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa"]
    
    print(f"Preparing datasets: {datasets_to_load}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples per dataset: {args.num_samples or 'All'}")
    
    # Load and save datasets
    if "math" in datasets_to_load:
        print("\nLoading MATH dataset...")
        math_data = load_math_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(math_data, args.output_dir, "math")
    
    if "gsm8k" in datasets_to_load:
        print("\nLoading GSM8K dataset...")
        gsm8k_data = load_gsm8k_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(gsm8k_data, args.output_dir, "gsm8k")
    
    # if "hellaswag" in datasets_to_load:
    #     print("\nLoading HellaSwag dataset...")
    #     hellaswag_data = load_hellaswag_dataset(num_samples=args.num_samples)
    #     save_dataset_to_parquet(hellaswag_data, args.output_dir, "hellaswag")
    
    if "mmlu" in datasets_to_load:
        print("\nLoading MMLU dataset (sample subjects)...")
        # Load a few representative subjects
        subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge"]
        for subject in subjects:
            mmlu_data = load_mmlu_dataset(subject=subject, num_samples=args.num_samples)
            save_dataset_to_parquet(mmlu_data, args.output_dir, f"mmlu_{subject}")
    
    if "arc" in datasets_to_load:
        print("\nLoading ARC dataset...")
        arc_challenge_data = load_arc_dataset(challenge=True, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_challenge_data, args.output_dir, "arc_challenge")
        
        arc_easy_data = load_arc_dataset(challenge=False, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_easy_data, args.output_dir, "arc_easy")
    
    if "truthfulqa" in datasets_to_load:
        print("\nLoading TruthfulQA dataset...")
        truthfulqa_data = load_truthfulqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(truthfulqa_data, args.output_dir, "truthfulqa")
    
    print(f"\nAll datasets prepared and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
