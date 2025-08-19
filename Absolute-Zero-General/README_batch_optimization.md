# GeneralIORewardManager 批量评估优化

## 概述

最新的 `GeneralIORewardManager` 已经优化了 self-judge 功能，支持批量处理以提高效率。

## 主要优化

### 🚀 批量处理优势

**优化前（单个处理）：**
- 每个评估请求单独创建数据集和数据加载器
- dataset len=1，多次调用 actor model
- 大量的数据加载和模型初始化开销

**优化后（批量处理）：**
- 将多个评估请求打包成批处理
- 可配置的批处理大小（默认8）
- 显著减少模型调用次数和数据加载开销

### 📊 性能对比

| 评估请求数量 | 优化前调用次数 | 优化后调用次数（batch_size=8） | 效率提升 |
|-------------|----------------|--------------------------------|----------|
| 16          | 16             | 2                              | 8x       |
| 32          | 32             | 4                              | 8x       |
| 100         | 100            | 13                             | ~7.7x    |

## 新增配置参数

### `self_judge_batch_size` (int, 默认: 8)

控制 self-judge 模式下的批处理大小：
- 较大的值可以提高效率，但需要更多内存
- 较小的值内存使用更少，但效率相对较低
- 推荐值：4-16，具体取决于GPU内存

## 使用示例

```python
from transformers import AutoTokenizer
from absolute_zero_reasoner.rewards.reward_managers import GeneralIORewardManager

# 创建启用批量处理的 reward manager
reward_manager = GeneralIORewardManager(
    tokenizer=tokenizer,
    # ... 其他配置参数
    use_self_judge=True,
    self_judge_batch_size=8,  # 批处理大小
)

# 批量评估会自动应用于：
# 1. 生成任务的 LLM judge 评分
# 2. 预测任务的 LLM judge 评分  
# 3. solver scores 计算中的评估
reward_tensor, all_scores, valid_questions = reward_manager(
    data=your_data_proto,
    problem_type='gen',
    rollout_actor_wg=your_rollout_actor,
    n_samples=3,
)
```

## 批量处理流程

### 生成任务批量评估
```
数据 → 生成所有评估提示 → 批量调用 actor model → 提取分数 → 组合最终奖励
     ↓
   单次数据加载 + 模型调用（批处理大小）
```

### 预测任务批量评估
```
数据 → 生成所有评估提示 → 批量调用 actor model → 提取分数 → 设置奖励
     ↓
   单次数据加载 + 模型调用（批处理大小）
```

### Solver Scores 批量评估
```
问题 → 生成解决方案 → 批量评估解决方案 → 计算平均分数 → 用于难度计算
      ↓
    批量评估多个解决方案
```

## 配置建议

### 根据GPU内存调整批处理大小

```python
# GPU内存较小（<8GB）
self_judge_batch_size=2

# GPU内存中等（8-16GB）  
self_judge_batch_size=4

# GPU内存较大（16-32GB）
self_judge_batch_size=8

# GPU内存很大（>32GB）
self_judge_batch_size=16
```

### 根据任务类型优化

```python
# 生成任务（通常需要更多评估）
self_judge_batch_size=8

# 预测任务（评估相对较少）
self_judge_batch_size=4

# 测试/调试模式
self_judge_batch_size=1
```

## 实现细节

### 批量处理方法

1. **`_generate_self_judge_responses_batch()`**
   - 主要的批量处理方法
   - 支持任意大小的提示列表
   - 自动分批处理以适应指定的批大小

2. **`_get_evaluation_scores_batch()`**
   - 统一的批量评估接口
   - 自动选择 self-judge 或外部 LLM 模式

3. **分批处理逻辑**
   - 使用 `SubsetRandomSampler` 进行批处理
   - 追踪原始索引以正确映射结果
   - 错误处理和回退机制

### 内存管理

- 临时文件自动清理
- 批处理后及时释放内存
- 支持大数据集的分批处理

## 兼容性

- 完全向后兼容现有代码
- 默认批处理大小为8，平衡效率和内存使用
- 外部 LLM 模式仍使用逐个调用（可进一步优化）

## 性能监控

批量处理会输出调试信息：
```
Self-Judge Response 0: <模型评分响应>
Self-Judge Response 1: <模型评分响应>
...
```

## 注意事项

1. **内存使用**：较大的批处理大小会使用更多GPU内存
2. **错误处理**：单个评估失败不会影响整个批次
3. **调试**：可以设置 `batch_size=1` 进行详细调试
4. **负载均衡**：在多GPU环境中会自动处理负载分配

这个优化显著提高了 self-judge 模式的效率，特别是在处理大量评估请求时。
