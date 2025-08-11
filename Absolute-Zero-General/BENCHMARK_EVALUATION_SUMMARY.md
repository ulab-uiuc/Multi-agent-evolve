# General任务Benchmark Evaluation配置总结

## 主要修改

### 1. main_azr_ppo.py
- **导入BenchmarkEvaluationRewardManager**: 添加了对benchmark评估的支持
- **General任务配置**: 
  - 训练使用`GeneralIORewardManager`
  - 验证使用`BenchmarkEvaluationRewardManager`而不是标准验证
  - 传递`benchmark_reward_fn`参数给trainer

### 2. azr_ray_trainer.py (GeneralIORayPPOTrainer)
- **新增参数**: 添加`benchmark_reward_fn`参数到构造函数
- **新增方法**:
  - `_is_general_task()`: 检查任务类型是否为general
  - `_run_benchmark_evaluation()`: 调用父类的benchmark评估方法
- **验证逻辑**: 修改验证逻辑，general任务只运行benchmark评估

### 3. reason_rl_ray_trainer.py (ReasonRLRayPPOTrainer)
- **Benchmark设置**: 在`__init__`中为general任务设置benchmark评估
- **采样逻辑**: 
  - 每个benchmark独立计算样本数
  - 总样本数 = `max_samples_per_benchmark × benchmark数量`
  - 当前实现: 先统计各benchmark大小，再应用总限制
- **验证流程**: general任务跳过标准验证，只运行benchmark评估

### 4. 配置文件 (azr_ppo_trainer_general.yaml)
- **val_files**: 设置为空列表（不使用标准验证）
- **task_type**: 设置为'general'
- **benchmark_evaluation**: 配置benchmark评估参数

## 采样逻辑说明

### 目标行为
每个benchmark最多采样`max_samples_per_benchmark`个样本：
- math: 最多100个样本
- gsm8k: 最多100个样本  
- hellaswag: 最多100个样本
- 总计: 最多300个样本

### 当前实现
1. 统计每个benchmark的样本数量
2. 计算总限制: `max_samples_per_benchmark × benchmark数量`
3. 创建合并数据集
4. 如果总大小超过限制，使用`torch.utils.data.Subset`截断

### 未来改进
- 实现真正的per-benchmark采样
- 在数据集创建阶段就限制每个benchmark的样本数

## 验证流程

### General任务
1. **初始验证**: 运行benchmark评估而不是标准验证
2. **训练中验证**: 按频率运行benchmark评估
3. **最终验证**: 运行benchmark评估

### 其他任务
1. 保持原有的标准验证流程

## 配置示例

```yaml
azr:
  task_type: general

data:
  val_files: []  # 空列表，不使用标准验证

benchmark_evaluation:
  enabled: true
  datasets: ['math', 'gsm8k', 'hellaswag']
  frequency: 100
  max_samples_per_benchmark: 100
```

## 使用方法

1. 使用`azr_ppo_trainer_general.yaml`配置文件
2. 确保`task_type: general`
3. 运行`prepare_test_datasets.py`准备benchmark数据集
4. 启动训练，系统将自动使用benchmark评估

## 注意事项

1. **采样均匀性**: 当前实现可能不能保证每个benchmark都有相同数量的样本
2. **内存使用**: 会同时加载所有benchmark数据集
3. **评估频率**: 建议根据训练步数调整benchmark评估频率

## 验证方法

运行测试脚本验证配置：
```bash
python test_benchmark_sampling.py
```
