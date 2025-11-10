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
import json
import os
from pathlib import Path
from pprint import pprint
from typing import List

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import CodeIORayPPOTrainer, GeneralIORayPPOTrainer
from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager, GeneralIORewardManager, BenchmarkEvaluationRewardManager


def load_api_keys(api_file_path: str = "api.json") -> List[str]:
    """
    Load API keys from a JSON file.
    
    Args:
        api_file_path: Path to the API keys JSON file
        
    Returns:
        List of API keys
    """
    try:
        # Try multiple possible locations for the API file
        possible_paths = [
            api_file_path,  # Current working directory
            os.path.join(os.path.dirname(__file__), api_file_path),  # Same directory as this script
            os.path.join(os.path.dirname(__file__), "..", api_file_path),  # Parent directory of this script
            os.path.join(os.path.dirname(__file__), "..", "..", api_file_path),  # Project root directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading API keys from: {path}")
                with open(path, 'r') as f:
                    data = json.load(f)
                    api_keys = data.get('api_keys', [])
                    if api_keys:
                        return api_keys
        
        print(f"Warning: API file {api_file_path} not found in any expected locations")
        print(f"Searched in: {possible_paths}")
        return [""]
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return [""]


@hydra.main(config_path='configs', config_name='azr_ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    
    def run(self, config):
        # Load API keys from api.json
        api_keys = load_api_keys()
        print(f"Loaded {len(api_keys)} API keys for parallel evaluation")
        
        # Set up dynamic timestamp and directories before resolving config
        from datetime import datetime
        import os
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        date_part = datetime.now().strftime("%Y-%m-%d")
        time_part = datetime.now().strftime("%H-%M-%S")
        
        # Set timestamp
        config.experiment_timestamp = timestamp
        
        # Create base experiment directory
        experiment_base_dir = f"./outputs/{date_part}/{time_part}_{config.trainer.project_name}_{config.trainer.experiment_name}"
        config.experiment_base_dir = experiment_base_dir
        
        # Set other directories
        config.trainer.output_dir = experiment_base_dir
        config.agent_output_dir = f"{experiment_base_dir}/agent_output"
        
        # Set checkpoint directory
        config.trainer.default_local_dir = f"{config.trainer.default_local_dir}/{date_part}/{time_part}_{config.trainer.project_name}_{config.trainer.experiment_name}"
        if hasattr(config, 'trainer') and hasattr(config.trainer, 'resume_dir'):
            # resume_mode has three choice: disable, auto and resume_path
            # disable cleans dataset and starts from scratch
            # auto resumes from the latest global_step checkpoint, use for resuming training
            # resume_path allows you to continue from any desired checkpoint, use for evaluation or resuming training. You have to give a specific global_step in resume_from_path
            if hasattr(config, 'trainer') and hasattr(config.trainer, 'resume_mode') and config.trainer.resume_mode != "disable":
                print("[DEBUG]: using resume_dir to set default local directory!")
                config.trainer.default_local_dir = config.trainer.resume_dir
                print("[DEBUG]: local_dir set to ", config.trainer.default_local_dir)
        
        os.makedirs(experiment_base_dir, exist_ok=True)
        os.makedirs(config.agent_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.trainer.default_local_dir), exist_ok=True)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if config.trainer.debug:
            import debugpy
            debugpy.listen(("0.0.0.0", config.trainer.debug_port))
            print(f"Debugger listening on port {config.trainer.debug_port}")
            debugpy.wait_for_client()
            print("Debugger attached!")

        # generator one batch, solver one batch
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

            assert not (not config.azr.reward.generation_reward_config.reject_multiple_functions and config.azr.data_selection_strategy.composite_function_n_min > 0), "If reject_multiple_functions is False, composite_function_n_min must be 0"

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        if config.actor_rollout_ref.model.pretrained_tokenizer:
            tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        if task_type == 'general':
            # Use GeneralIORewardManager for training
            reward_fn = GeneralIORewardManager(
                tokenizer=tokenizer,
                num_examine=0,
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
                use_format_reward=getattr(config.azr, 'use_format_reward', True),
                agent_output_dir=config.agent_output_dir
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
                api_keys=api_keys,  # Pass the loaded API keys
            )
        else:
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
                processor=processor,
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
