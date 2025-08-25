# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import hydra
from pathlib import Path
from pprint import pprint

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import CodeIORayPPOTrainer, GeneralIORayPPOTrainer
from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager, GeneralIORewardManager, BenchmarkEvaluationRewardManager


@hydra.main(config_path='configs', config_name='azr_ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(address=None, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if config.trainer.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", config.trainer.debug_port))
        print(f"Debugger listening on port {config.trainer.debug_port}")
        debugpy.wait_for_client()
        print("Debugger attached!")

    # generator one batch, solver one batch
    # proposor和solver把每个type都完成train_batch_size次rollout之后更新
    config.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.train_batch_size * len(config.azr.problem_types) * (2 if config.azr.train_propose else 1)
    pprint(f"auto setting ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
    config.azr.data_selection_strategy.data_len = config.data.train_batch_size * config.azr.data_selection_strategy.update_iteration
    pprint(f"auto setting data_len: {config.azr.data_selection_strategy.data_len}")

    # Determine task type for path and assertions
    task_type = getattr(config.azr, 'task_type', 'code')
    
    if task_type == 'general':
        # For general tasks, use a different path structure
        config.trainer.default_local_dir = (Path(config.trainer.default_local_dir) / 'general_io' / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type).as_posix()
        # Set default problem types for general tasks if not specified
        if not hasattr(config.azr, 'problem_types') or not config.azr.problem_types:
            config.azr.problem_types = ['general']
    else:
        # Original path structure for code tasks
        config.trainer.default_local_dir = (Path(config.trainer.default_local_dir) / config.data.train_files.split('/')[-1].split('.')[0] / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type).as_posix()
        # Original assertion for code tasks
        assert not (not config.azr.reward.generation_reward_config.reject_multiple_functions and config.azr.data_selection_strategy.composite_function_n_min > 0), "If reject_multiple_functions is False, composite_function_n_min must be 0"

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # base model chat template
    if config.actor_rollout_ref.model.pretrained_tokenizer:
        tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError


    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if config.actor_rollout_ref.ref.include_ref:
        mapping[Role.RefPolicy] = global_pool_id
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Determine task type and create appropriate reward manager
    task_type = getattr(config.azr, 'task_type', 'code')  # default to 'code' for backward compatibility
    
    if task_type == 'general':
        # Use GeneralIORewardManager for training
        reward_fn = GeneralIORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            split='train',
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            generation_reward_config=config.azr.reward.generation_reward_config,
            eval_reward_config=getattr(config.azr.reward, 'eval_reward_config', {}),
            model_name=getattr(config.reward_fn, 'llm_model_name', 'meta/llama-3.1-405b-instruct'),
            max_prompt_length=config.data.max_prompt_length,
            temperature=getattr(config.reward_fn, 'temperature', 0.7),
            max_tokens=getattr(config.reward_fn, 'max_tokens', 1000),
            top_p=getattr(config.reward_fn, 'top_p', 0.95),
            stream=getattr(config.reward_fn, 'stream', True),
            boxed_retry=config.reward_fn.boxed_retry,
            judge_with_actor=config.reward_fn.judge_with_actor,
            infer_together=config.reward_fn.infer_together
            # judge_with_actor only available for infering question and answer score together
        )

        # For validation, use BenchmarkEvaluationRewardManager instead
        val_reward_fn = BenchmarkEvaluationRewardManager(
            tokenizer=tokenizer,
            model_name=getattr(config.azr, 'benchmark_eval_model', 'meta/llama-3.1-405b-instruct'),
            temperature=getattr(config.reward_fn, 'temperature', 0.0),
            max_tokens=getattr(config.reward_fn, 'max_tokens', 500),
            top_p=getattr(config.reward_fn, 'top_p', 0.95),
            stream=getattr(config.reward_fn, 'stream', True),
            boxed_retry=config.reward_fn.boxed_retry,
            # maybe judge_with_actor as well?
        )
    else:
        # Use CodeIORewardManager for code tasks (default behavior)
        reward_fn = CodeIORewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='train',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.azr.reward.generation_reward_config,
            valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.azr.reward.extract_code_block,
            code_f_reward_type=config.azr.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = CodeIORewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            reward_fn_extraction_type=config.reward_fn.extraction_type,
            math_metric=config.reward_fn.math_metric,
            split='test',
            splitter=config.reward_fn.splitter,
            output_path=config.trainer.default_local_dir,
            max_prompt_length=config.data.max_prompt_length,
            generation_reward_config=config.azr.reward.generation_reward_config,
            valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
            debug=config.trainer.debug,
            extract_code_block=config.azr.reward.extract_code_block,
            code_f_reward_type=config.azr.reward.code_f_reward_type,
            boxed_retry=config.reward_fn.boxed_retry,
        )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # Set wandb tags based on task type
    if task_type == 'general':
        wandb_tags = [
            'generalio', config.azr.pred_data_mix_strategy,
            config.azr.data_selection_strategy.get('valid_question_filter', 'all'),
        ]
    else:
        wandb_tags = [
            'codeio', config.azr.pred_data_mix_strategy, 'executor-' + config.azr.executor,
            config.azr.data_selection_strategy.valid_program_filter, config.azr.gen_data_probabilities_strategy,
        ]
    
    wandb_tags.extend(config.azr.problem_types)
    if config.trainer.wandb_tags is not None:
        config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(',')
    else:
        config.trainer.wandb_tags = wandb_tags

    # Create appropriate trainer based on task type
    if task_type == 'general':
        trainer = GeneralIORayPPOTrainer(
            past_epoch_window=config.azr.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=None,  # No standard validation for general tasks
            benchmark_reward_fn=val_reward_fn,  # Use benchmark evaluation instead
        )
    else:
        trainer = CodeIORayPPOTrainer(
            past_epoch_window=config.azr.past_epoch_window,
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        import os
        import traceback
        traceback.print_exc()
        os._exit(1)
