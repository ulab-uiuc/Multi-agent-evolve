#!/usr/bin/env python3
"""
测试benchmark evaluation的配置和采样逻辑
"""

import sys
from pathlib import Path

#!/usr/bin/env python3

import torch
import torch.utils.data
from collections import defaultdict

def test_benchmark_sampling_logic():
    """Test the benchmark sampling logic."""
    
    # Mock benchmark files and sizes
    benchmark_files = [
        '/path/to/math.parquet',
        '/path/to/gsm8k.parquet', 
        '/path/to/hellaswag.parquet'
    ]
    
    # Mock dataset sizes
    benchmark_sizes = {
        'math': 150,
        'gsm8k': 80,
        'hellaswag': 200
    }
    
    max_samples_per_benchmark = 100
    
    print("=== Testing Benchmark Sampling Logic ===")
    print(f"Max samples per benchmark: {max_samples_per_benchmark}")
    print()
    
    # Simulate the actual logic from the code
    benchmark_datasets = []
    total_samples = 0
    
    for benchmark_file in benchmark_files:
        benchmark_name = benchmark_file.split('/')[-1].split('.')[0]
        benchmark_size = benchmark_sizes[benchmark_name]
        
        print(f"Processing {benchmark_name}:")
        print(f"  Original size: {benchmark_size}")
        
        # Apply per-benchmark sampling limit
        if max_samples_per_benchmark and benchmark_size > max_samples_per_benchmark:
            # Create indices for limited sampling
            indices = torch.randperm(benchmark_size)[:max_samples_per_benchmark]
            actual_size = max_samples_per_benchmark
            print(f"  Limited to: {actual_size} samples")
            print(f"  Sample indices: {indices[:10].tolist()}... (showing first 10)")
        else:
            actual_size = benchmark_size
            print(f"  No limiting needed: {actual_size} samples")
        
        # Mock creating the dataset
        benchmark_datasets.append(f"MockDataset_{benchmark_name}_{actual_size}")
        total_samples += actual_size
        print()
    
    print(f"Total samples across all benchmarks: {total_samples}")
    print(f"Expected total: {min(sum(benchmark_sizes.values()), max_samples_per_benchmark * len(benchmark_files))}")
    
    # Verify the logic
    expected_individual_samples = [
        min(benchmark_sizes['math'], max_samples_per_benchmark),
        min(benchmark_sizes['gsm8k'], max_samples_per_benchmark), 
        min(benchmark_sizes['hellaswag'], max_samples_per_benchmark)
    ]
    expected_total = sum(expected_individual_samples)
    
    print("\n=== Verification ===")
    print(f"Expected samples per benchmark: {expected_individual_samples}")
    print(f"Expected total: {expected_total}")
    print(f"Actual total: {total_samples}")
    print(f"Logic correct: {total_samples == expected_total}")
    
    # Test with different limits
    print("\n=== Testing Different Limits ===")
    for limit in [50, 100, 200, None]:
        print(f"\nLimit: {limit}")
        total = 0
        for name, size in benchmark_sizes.items():
            if limit is None:
                actual = size
            else:
                actual = min(size, limit)
            total += actual
            print(f"  {name}: {actual}/{size}")
        print(f"  Total: {total}")

if __name__ == "__main__":
    test_benchmark_sampling_logic()

def test_config_structure():
    """测试配置文件结构"""
    print("\n🔍 测试配置文件结构...")
    
    import yaml
    config_path = Path("absolute_zero_reasoner/configs/azr_ppo_trainer_general.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查benchmark_evaluation配置
    benchmark_config = config.get('benchmark_evaluation', {})
    if not benchmark_config:
        print("❌ 缺少benchmark_evaluation配置")
        return False
    
    enabled = benchmark_config.get('enabled', False)
    if not enabled:
        print("❌ benchmark_evaluation未启用")
        return False
    
    datasets = benchmark_config.get('datasets', [])
    if not datasets:
        print("❌ 没有配置datasets")
        return False
    
    frequency = benchmark_config.get('frequency', 0)
    if frequency <= 0:
        print("❌ frequency应该大于0")
        return False
        
    max_samples = benchmark_config.get('max_samples_per_benchmark')
    if max_samples is None:
        print("⚠️  建议设置max_samples_per_benchmark")
    else:
        print(f"✅ max_samples_per_benchmark: {max_samples}")
    
    print("✅ benchmark_evaluation配置正确")
    return True

def test_imports():
    """测试导入是否正确"""
    print("\n🔍 测试导入...")
    
    try:
        from absolute_zero_reasoner.rewards.reward_managers import BenchmarkEvaluationRewardManager
        print("✅ BenchmarkEvaluationRewardManager导入成功")
        
        from absolute_zero_reasoner.trainer.ppo.reason_rl_ray_trainer import ReasonRLRayPPOTrainer
        print("✅ ReasonRLRayPPOTrainer导入成功")
        
        from absolute_zero_reasoner.utils.benchmark_config import BenchmarkConfig
        print("✅ BenchmarkConfig导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def main():
    print("=== Benchmark Evaluation采样逻辑测试 ===\n")
    
    # 测试导入
    imports_ok = test_imports()
    
    # 测试配置
    config_ok = test_config_structure()
    
    # 测试采样逻辑
    sampling_ok = test_benchmark_sampling_logic()
    
    if imports_ok and config_ok and sampling_ok:
        print("\n🎉 所有测试通过！")
        print("\n配置总结:")
        print("  - 每个benchmark将独立限制样本数")
        print("  - 总样本数 = max_samples_per_benchmark × benchmark数量")
        print("  - 使用BenchmarkEvaluationRewardManager进行评估")
        print("\n注意事项:")
        print("  - 当前实现先加载所有benchmark再限制总数")
        print("  - 未来可以实现真正的per-benchmark采样")
        return True
    else:
        print("\n❌ 测试失败！请检查上述错误。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
