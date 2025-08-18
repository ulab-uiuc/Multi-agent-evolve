# GeneralIORewardManager Self-Judge 功能

## 概述

`GeneralIORewardManager` 现在支持 self-judge 功能，允许使用 actor model 本身进行打分，而不是依赖外部 LLM API。这个功能可以：

- 减少对外部 API 的依赖和成本
- 提供更一致的评分标准（因为使用同一个模型）
- 在训练过程中创建更紧密的反馈循环

## 新增参数

### `use_self_judge` (bool, 默认: False)

- `True`: 使用 actor model 进行自评分
- `False`: 使用外部 LLM API 进行评分（原始行为）

## 使用方法

### 1. 启用 Self-Judge

```python
from transformers import AutoTokenizer
from absolute_zero_reasoner.rewards.reward_managers import GeneralIORewardManager

tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# 创建启用 self-judge 的 reward manager
reward_manager = GeneralIORewardManager(
    tokenizer=tokenizer,
    # ... 其他配置参数
    use_self_judge=True,  # 启用 self-judge
)
```

### 2. 调用奖励计算

```python
# 对于生成任务（需要 rollout_actor_wg）
reward_tensor, all_scores, valid_questions = reward_manager(
    data=your_data_proto,
    problem_type='gen',
    rollout_actor_wg=your_rollout_actor,  # 必需用于 self-judge
    n_samples=3,
)

# 对于预测任务（可选 rollout_actor_wg）
reward_tensor, all_scores, valid_questions = reward_manager(
    data=your_data_proto,
    problem_type='pred',
    rollout_actor_wg=your_rollout_actor,  # 可选
)
```

## 工作原理

### Self-Judge 模式流程

1. **创建评估提示**：生成包含评分标准的提示
2. **模型自评**：使用 actor model 生成评分响应
3. **提取分数**：从响应中提取 1-10 的分数
4. **归一化**：转换为 0-1 区间的奖励值

### 评分标准

模型会根据以下标准进行评分（1-10 分制）：

- **10分**: 完美、完整且清晰
- **8-9分**: 基本正确，可能有小问题
- **5-7分**: 部分正确但有明显问题
- **2-4分**: 有一定价值但大体错误
- **1分**: 完全错误或不相关

## 配置示例

### Self-Judge 配置
```python
config = {
    "use_self_judge": True,
    "temperature": 0.7,        # 控制模型生成的随机性
    "max_tokens": 1000,        # 评估响应的最大长度
    "top_p": 0.95,
    # model_name 在 self-judge 模式下会被忽略
}
```

### 外部 LLM 配置
```python
config = {
    "use_self_judge": False,
    "model_name": "meta/llama-3.1-405b-instruct",  # 外部 API 模型
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.95,
}
```

## 优势与考虑

### Self-Judge 优势
- ✅ 降低 API 调用成本
- ✅ 减少网络延迟
- ✅ 提供一致的评分标准
- ✅ 在训练中形成反馈循环

### 外部 LLM 优势
- ✅ 可能更客观的评分
- ✅ 使用更强大的评估模型
- ✅ 避免评分偏差

### 使用建议

- **训练阶段**: 推荐使用 self-judge 以降低成本和提高效率
- **评估阶段**: 可以考虑使用外部 LLM 以提高评分客观性
- **生成任务**: self-judge 特别适合，因为可以创建更紧密的反馈循环
- **预测任务**: 两种模式都可以有效工作

## 技术细节

### 重要参数

- `rollout_actor_wg`: 在 self-judge 模式下，这个参数对于生成任务是必需的
- `temperature`/`max_tokens`/`top_p`: 控制模型生成评分响应的参数

### 错误处理

- 如果 self-judge 模式下 `rollout_actor_wg` 为 `None`，会回退到默认分数 0.5
- 如果无法从响应中提取分数，会使用默认分数
- 所有错误都会被捕获并记录，不会中断训练流程

## 兼容性

这个新功能完全向后兼容，不会影响现有代码：

- 默认 `use_self_judge=False` 保持原有行为
- 所有现有参数和方法都保持不变
- API 接口没有破坏性变更

## 示例文件

- `example_self_judge_usage.py`: 完整使用示例
- `config_self_judge_example.py`: 配置文件示例

## 注意事项

1. Self-judge 模式需要确保 `rollout_actor_wg` 在生成任务中可用
2. 评分质量依赖于 actor model 的能力
3. 可能需要调整 temperature 等参数来获得最佳评分效果
4. 建议在使用前进行小规模测试以验证评分质量
