#!/usr/bin/env python3

import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import DataLoader
from verl.utils.dataset.rl_dataset import collate_fn

def test_benchmark_loading():
    """Test loading benchmark data to debug the indexing issue."""
    
    print("=== Testing Benchmark Data Loading ===")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test benchmark files (you'll need to provide actual paths)
    benchmark_dir = "/path/to/validation"  # Update this path
    test_files = []
    
    # Check for common benchmark files
    potential_files = [
        "math_validation.parquet",
        "gsm8k_validation.parquet", 
        "hellaswag_validation.parquet"
    ]
    
    for filename in potential_files:
        filepath = os.path.join(benchmark_dir, filename)
        if os.path.exists(filepath):
            test_files.append(filepath)
            print(f"Found benchmark file: {filepath}")
    
    if not test_files:
        print("No benchmark files found. Creating a mock test...")
        # Create mock data for testing
        import pandas as pd
        
        mock_data = [
            {
                "prompt": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
                "data_source": "mock_math",
                "extra_info": {"metric": "math_accuracy"},
                "reward_model": {"style": "rule", "ground_truth": "4"}
            },
            {
                "prompt": [{"role": "user", "content": "What is the capital of France?"}],
                "answer": "Paris", 
                "data_source": "mock_qa",
                "extra_info": {"metric": "general_accuracy"},
                "reward_model": {"style": "rule", "ground_truth": "Paris"}
            }
        ]
        
        # Save mock data
        mock_path = "/tmp/mock_benchmark.parquet"
        df = pd.DataFrame(mock_data)
        df.to_parquet(mock_path)
        test_files = [mock_path]
        print(f"Created mock benchmark file: {mock_path}")
    
    # Test loading each file
    for test_file in test_files:
        print(f"\n--- Testing file: {test_file} ---")
        
        try:
            # Load dataset
            dataset = RLHFDataset(
                parquet_files=[test_file],
                tokenizer=tokenizer,
                prompt_key="prompt",
                max_prompt_length=1024,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error',
                extra_source_key="benchmark"
            )
            
            print(f"Dataset loaded successfully. Size: {len(dataset)}")
            
            # Test dataloader
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=min(len(dataset), 2),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn
            )
            
            print(f"DataLoader created. Number of batches: {len(dataloader)}")
            
            # Test accessing the data
            for i, batch_data in enumerate(dataloader):
                print(f"\nBatch {i+1}:")
                print(f"  Keys: {list(batch_data.keys())}")
                
                from verl import DataProto
                batch = DataProto.from_single_dict(batch_data)
                print(f"  DataProto length: {len(batch)}")
                print(f"  Batch keys: {list(batch.batch.keys())}")
                print(f"  Non-tensor keys: {list(batch.non_tensor_batch.keys()) if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch else 'None'}")
                
                # Test indexing
                for j in range(min(len(batch), 2)):
                    try:
                        item = batch[j]
                        print(f"    Item {j} accessible: ✓")
                        print(f"    Item {j} type: {type(item)}")
                        if hasattr(item, 'non_tensor_batch') and item.non_tensor_batch:
                            print(f"    Item {j} data_source: {item.non_tensor_batch.get('data_source', 'N/A')}")
                    except Exception as e:
                        print(f"    Item {j} error: {str(e)}")
                
                # Only test first batch
                break
                
        except Exception as e:
            print(f"Error loading {test_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_benchmark_loading()
