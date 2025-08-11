# General任务Benchmark Evaluation配置总结

## 问题诊断

### 当前错误
用户遇到的错误：`Cannot index by location index with a non-integer key` 和 `Completed benchmark evaluation with 0 metrics`

### 根本原因
1. **Benchmark数据集缺失**: 默认配置指向`./validation_datasets`目录，但该目录不存在或为空
2. **数据索引问题**: 在BenchmarkEvaluationRewardManager中尝试使用`data[i]`访问DataProto对象时出错

## 解决方案

### 1. 准备Benchmark数据集
运行以下命令创建测试数据集：
```bash
python prepare_test_datasets.py
```

这会在`./validation_datasets`目录下创建以下测试文件：
- `math_validation.parquet`
- `gsm8k_validation.parquet` 
- `truthfulqa_validation.parquet`
- `arc_challenge_validation.parquet`

### 2. 配置文件更新
确保`azr_ppo_trainer_general.yaml`包含：
```yaml
azr:
  task_type: general

benchmark_validation_dir: "./validation_datasets"
benchmark_names: ["math", "gsm8k", "truthfulqa", "arc_challenge"]
benchmark_max_samples: 10
```

### 3. 错误处理改进
已在代码中添加了以下改进：
- 更好的错误处理和调试信息
- 检查benchmark文件是否存在
- 优雅地处理空数据集情况
- 详细的状态日志

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
  - 使用`torch.utils.data.Subset`和`ConcatDataset`实现真正的per-benchmark采样
  - 当前实现: 为每个benchmark单独限制样本数，然后合并
- **错误处理**: 改进了错误处理和日志记录

### 4. reward_managers.py (BenchmarkEvaluationRewardManager)
- **调试信息**: 添加了详细的调试日志
- **错误捕获**: 包装了主要逻辑以捕获和报告错误
- **数据验证**: 验证DataProto对象的结构和内容

### 5. 配置文件 (azr_ppo_trainer_general.yaml)
- **val_files**: 设置为空列表（不使用标准验证）
- **task_type**: 设置为'general'
- **benchmark_evaluation**: 配置benchmark评估参数

## 采样逻辑说明

### 实现的行为 ✅
每个benchmark最多采样`max_samples_per_benchmark`个样本：
- math: 最多10个样本
- gsm8k: 最多10个样本  
- truthfulqa: 最多10个样本
- arc_challenge: 最多10个样本
- 总计: 最多40个样本

### 具体实现
1. 为每个benchmark文件单独创建RLHFDataset
2. 如果样本数超过限制，使用`torch.randperm()`随机选择样本
3. 创建`torch.utils.data.Subset`来限制样本数
4. 使用`torch.utils.data.ConcatDataset`合并所有benchmark数据集

## 验证流程

### General任务
1. **初始验证**: 运行benchmark评估而不是标准验证
2. **训练中验证**: 按频率运行benchmark评估
3. **最终验证**: 运行benchmark评估

### 其他任务
1. 保持原有的标准验证流程

## 使用步骤

### 1. 准备数据
```bash
python prepare_test_datasets.py
```

### 2. 验证配置
```bash
python test_benchmark_config.py
```

### 3. 启动训练
```bash
python main_azr_ppo.py --config configs/azr_ppo_trainer_general.yaml
```

## 故障排除

### 如果仍然出现索引错误
1. 检查DataProto版本兼容性
2. 验证parquet文件格式是否正确
3. 确认tokenizer配置正确

### 如果没有benchmark文件
1. 运行`prepare_test_datasets.py`创建测试数据
2. 或者更新`benchmark_validation_dir`指向实际数据目录
3. 确认文件名格式：`{benchmark_name}_validation.parquet`

### 如果LLM评估失败
1. 检查OpenAI API密钥是否有效
2. 确认网络连接正常
3. 可以临时修改为rule-based评估进行测试

## 注意事项

1. **测试数据**: 当前创建的是最小测试数据集，实际使用时需要更完整的benchmark
2. **API调用**: BenchmarkEvaluationRewardManager需要外部LLM API，可能产生费用
3. **内存使用**: 会同时加载所有benchmark数据集
4. **评估频率**: 建议根据训练步数调整benchmark评估频率

## 验证命令

```bash
# 创建测试数据
python prepare_test_datasets.py

# 测试配置
python test_benchmark_config.py

# 运行benchmark采样测试
python test_benchmark_sampling.py
```
