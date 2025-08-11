#!/usr/bin/env python3
"""
测试general任务的完整集成
"""

import sys
from pathlib import Path
import yaml

def test_config_integration():
    """测试配置文件是否正确"""
    print("🔍 测试配置文件...")
    
    config_path = Path("absolute_zero_reasoner/configs/azr_ppo_trainer_general.yaml")
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查任务类型
    task_type = config.get('azr', {}).get('task_type', 'unknown')
    if task_type != 'general':
        print(f"❌ azr.task_type应该是'general'，但是是'{task_type}'")
        return False
    print(f"✅ azr.task_type: {task_type}")
    
    # 检查validation文件是否为空
    val_files = config.get('data', {}).get('val_files', [])
    if val_files and len(val_files) > 0:
        print(f"❌ 对于general任务，val_files应该为空，但是包含: {val_files}")
        return False
    print("✅ validation文件正确设置为空")
    
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
    
    return True

def test_trainer_integration():
    """测试trainer集成"""
    print("\n🔍 测试trainer集成...")
    
    try:
        # 导入测试
        from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import GeneralIORayPPOTrainer
        print("✅ GeneralIORayPPOTrainer导入成功")
        
        # 检查是否有正确的方法
        if not hasattr(GeneralIORayPPOTrainer, '_is_general_task'):
            print("❌ GeneralIORayPPOTrainer缺少_is_general_task方法")
            return False
        print("✅ _is_general_task方法存在")
        
        if not hasattr(GeneralIORayPPOTrainer, '_run_benchmark_evaluation'):
            print("❌ GeneralIORayPPOTrainer缺少_run_benchmark_evaluation方法")
            return False
        print("✅ _run_benchmark_evaluation方法存在")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入GeneralIORayPPOTrainer失败: {e}")
        return False

def test_reward_manager_integration():
    """测试reward manager集成"""
    print("\n🔍 测试reward manager集成...")
    
    try:
        # 导入测试
        from absolute_zero_reasoner.rewards.reward_managers import BenchmarkEvaluationRewardManager
        print("✅ BenchmarkEvaluationRewardManager导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入BenchmarkEvaluationRewardManager失败: {e}")
        return False

def test_main_script_integration():
    """测试main脚本集成"""
    print("\n🔍 测试main脚本集成...")
    
    try:
        # 检查main脚本是否有正确的导入
        main_path = Path("absolute_zero_reasoner/main_azr_ppo.py")
        if not main_path.exists():
            print(f"❌ main脚本不存在: {main_path}")
            return False
        
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否导入了BenchmarkEvaluationRewardManager
        if 'BenchmarkEvaluationRewardManager' not in content:
            print("❌ main脚本未导入BenchmarkEvaluationRewardManager")
            return False
        print("✅ main脚本正确导入BenchmarkEvaluationRewardManager")
        
        # 检查是否有general task的逻辑
        if "task_type == 'general'" not in content:
            print("❌ main脚本缺少general task逻辑")
            return False
        print("✅ main脚本包含general task逻辑")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查main脚本失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=== General任务集成测试 ===\n")
    
    tests = [
        ("配置文件", test_config_integration),
        ("Trainer集成", test_trainer_integration),
        ("Reward Manager集成", test_reward_manager_integration),
        ("Main脚本集成", test_main_script_integration),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n系统已正确配置为general任务模式：")
        print("1. ✅ 配置文件设置正确")
        print("2. ✅ Trainer支持benchmark评估")
        print("3. ✅ Reward Manager正确集成")
        print("4. ✅ Main脚本逻辑正确")
        print("\n可以开始训练general任务了！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败！请检查上述错误。")
        sys.exit(1)

if __name__ == "__main__":
    main()
