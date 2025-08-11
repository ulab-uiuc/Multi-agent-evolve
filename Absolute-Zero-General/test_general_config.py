#!/usr/bin/env python3
"""
测试general配置是否正确设置
"""

import yaml
import sys
from pathlib import Path

def test_general_config():
    """测试general配置文件"""
    config_path = Path("absolute_zero_reasoner/configs/azr_ppo_trainer_general.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("🔍 检查配置文件...")
    
    # 检查任务类型
    task_type = config.get('azr.task_type', 'unknown')
    print(f"任务类型: {task_type}")
    if task_type != 'general':
        print(f"❌ 任务类型应该是'general'，但是是'{task_type}'")
        return False
    print("✅ 任务类型正确")
    
    # 检查validation文件是否为空
    val_files = config.get('data', {}).get('val_files', [])
    print(f"验证文件: {val_files}")
    if val_files and len(val_files) > 0:
        print(f"❌ 对于general任务，val_files应该为空，但是包含: {val_files}")
        return False
    print("✅ 验证文件配置正确（空列表）")
    
    # 检查benchmark评估配置
    benchmark_config = config.get('benchmark_evaluation', {})
    if not benchmark_config:
        print("❌ 缺少benchmark_evaluation配置")
        return False
    
    enabled = benchmark_config.get('enabled', False)
    if not enabled:
        print("❌ benchmark_evaluation未启用")
        return False
    print("✅ benchmark_evaluation已启用")
    
    datasets = benchmark_config.get('datasets', [])
    print(f"基准数据集: {datasets}")
    if not datasets:
        print("❌ 没有配置基准数据集")
        return False
    print("✅ 基准数据集配置正确")
    
    frequency = benchmark_config.get('frequency', 0)
    print(f"评估频率: {frequency}")
    if frequency <= 0:
        print("❌ 评估频率应该大于0")
        return False
    print("✅ 评估频率配置正确")
    
    # 检查reward_fn配置
    reward_fn = config.get('reward_fn', {})
    if not reward_fn:
        print("❌ 缺少reward_fn配置")
        return False
    
    style = reward_fn.get('style', 'unknown')
    print(f"奖励函数样式: {style}")
    if style != 'llm_as_judge':
        print(f"❌ 对于general任务，reward_fn.style应该是'llm_as_judge'，但是是'{style}'")
        return False
    print("✅ 奖励函数样式正确")
    
    print("\n🎉 所有配置检查通过！")
    print("配置要点:")
    print(f"  - 任务类型: {task_type}")
    print(f"  - 验证文件: {val_files if val_files else '无（仅基准评估）'}")
    print(f"  - 基准数据集: {', '.join(datasets)}")
    print(f"  - 评估频率: 每{frequency}步")
    print(f"  - 奖励函数: {style}")
    
    return True

def test_required_files():
    """检查所需文件是否存在"""
    print("\n🔍 检查所需文件...")
    
    required_files = [
        "absolute_zero_reasoner/rewards/reward_managers.py",
        "absolute_zero_reasoner/trainer/ppo/reason_rl_ray_trainer.py",
        "absolute_zero_reasoner/utils/benchmark_config.py",
        "prepare_test_datasets.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("=== General任务配置测试 ===\n")
    
    config_ok = test_general_config()
    files_ok = test_required_files()
    
    if config_ok and files_ok:
        print("\n🎉 所有测试通过！系统已正确配置为general任务模式。")
        print("\n下一步:")
        print("1. 运行 prepare_test_datasets.py 准备基准数据集")
        print("2. 使用 azr_ppo_trainer_general.yaml 配置开始训练")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！请检查上述错误。")
        sys.exit(1)
