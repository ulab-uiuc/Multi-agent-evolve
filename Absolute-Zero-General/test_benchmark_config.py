#!/usr/bin/env python3
"""
测试general任务的benchmark evaluation配置
"""

import sys
from pathlib import Path

def test_imports():
    """测试导入是否正确"""
    try:
        from absolute_zero_reasoner.rewards.reward_managers import BenchmarkEvaluationRewardManager
        print("✅ BenchmarkEvaluationRewardManager导入成功")
        
        from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import GeneralIORayPPOTrainer
        print("✅ GeneralIORayPPOTrainer导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_config_structure():
    """测试配置文件结构"""
    import yaml
    config_path = Path("absolute_zero_reasoner/configs/azr_ppo_trainer_general.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查task_type
    task_type = config.get('azr', {}).get('task_type')
    if task_type != 'general':
        print(f"❌ task_type应该是'general'，但是是'{task_type}'")
        return False
    print("✅ task_type配置正确")
    
    # 检查val_files是否为空
    val_files = config.get('data', {}).get('val_files', [])
    if val_files and len(val_files) > 0:
        print(f"❌ val_files应该为空，但包含: {val_files}")
        return False
    print("✅ val_files配置正确（空列表）")
    
    # 检查benchmark_evaluation配置
    benchmark_config = config.get('benchmark_evaluation', {})
    if not benchmark_config:
        print("❌ 缺少benchmark_evaluation配置")
        return False
    
    enabled = benchmark_config.get('enabled', False)
    if not enabled:
        print("❌ benchmark_evaluation未启用")
        return False
    print("✅ benchmark_evaluation已启用")
    
    return True

def test_trainer_logic():
    """测试trainer逻辑"""
    print("\n🔍 检查trainer逻辑...")
    
    # 检查GeneralIORayPPOTrainer是否有_is_general_task方法
    try:
        from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import GeneralIORayPPOTrainer
        
        # 检查方法是否存在
        if not hasattr(GeneralIORayPPOTrainer, '_is_general_task'):
            print("❌ GeneralIORayPPOTrainer缺少_is_general_task方法")
            return False
        print("✅ _is_general_task方法存在")
        
        if not hasattr(GeneralIORayPPOTrainer, '_run_benchmark_evaluation'):
            print("❌ GeneralIORayPPOTrainer缺少_run_benchmark_evaluation方法")
            return False
        print("✅ _run_benchmark_evaluation方法存在")
        
        return True
    except Exception as e:
        print(f"❌ trainer逻辑检查失败: {e}")
        return False

def main():
    print("=== General任务Benchmark Evaluation配置测试 ===\n")
    
    # 测试导入
    print("1. 测试导入...")
    imports_ok = test_imports()
    
    # 测试配置
    print("\n2. 测试配置文件...")
    config_ok = test_config_structure()
    
    # 测试trainer逻辑
    print("\n3. 测试trainer逻辑...")
    trainer_ok = test_trainer_logic()
    
    if imports_ok and config_ok and trainer_ok:
        print("\n🎉 所有测试通过！")
        print("\n配置总结:")
        print("  - General任务将使用BenchmarkEvaluationRewardManager进行评估")
        print("  - 训练使用GeneralIORewardManager")
        print("  - 验证使用BenchmarkEvaluationRewardManager") 
        print("  - 不再使用标准验证集")
        print("\n下一步:")
        print("1. 运行 prepare_test_datasets.py 准备基准数据集")
        print("2. 使用 azr_ppo_trainer_general.yaml 配置开始训练")
        return True
    else:
        print("\n❌ 测试失败！请检查上述错误。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
