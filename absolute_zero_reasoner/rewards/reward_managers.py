import os
from functools import partial
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from openai import OpenAI
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import absolute_zero_reasoner.rewards.custom_evaluate as custom_evaluate
from absolute_zero_reasoner.rewards.code_reward import (
    parse_code_input_output,
    parse_inputs_message,
    parse_code_function,
    ast_edit_distance,
    get_code_complexity_reward,
    get_halstead_reward,
    get_type_counts_reward,
)
from absolute_zero_reasoner.rewards.custom_evaluate import get_format_reward, extract_answer, extract_thought
from absolute_zero_reasoner.data_construction.process_data import boxed_instruction, instruction_following
from absolute_zero_reasoner.data_construction.constructor import get_code_problem_predictor_prompt
from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
from absolute_zero_reasoner.utils.code_utils.checks import check_composite_function, check_no_definitions


class CodeIORewardManager():
    """The reward manager."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        math_metric: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        debug: bool = False,
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        batched_estimate: bool = False,
        extract_code_block: bool = True,
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',
        boxed_retry: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = partial(custom_evaluate.get_reward, math_metric=math_metric, boxed_retry=boxed_retry)
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.split = split
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.generation_reward_config = generation_reward_config
        self.valid_program_filter = valid_program_filter
        self.batched_estimate = batched_estimate
        self.debug = debug
        self.extract_code_block = extract_code_block
        self.use_original_code_as_ref = generation_reward_config.use_original_code_as_ref
        self.num_inputs = num_inputs
        self.code_f_reward_type = code_f_reward_type
        self.boxed_retry = boxed_retry

    @staticmethod
    def extract_input_output(extracted_content: str, return_input: bool = True, return_output: bool = False) -> Tuple[str, str]:
        input_pattern = r"```input\s*\n?(.*?)\n?```"
        output_pattern = r"```output\s*\n?(.*?)\n?```"
        assert not (return_input and return_output), "Cannot return both input and output"
        assert return_input or return_output, "Must return at least one of input or output"

        # Use flags for case-insensitive matching and dotall
        flags = re.DOTALL | re.IGNORECASE
        if return_input:
            input_matches = list(re.finditer(input_pattern, extracted_content, flags))
            if not input_matches:
                # Try alternative pattern without explicit input block
                input_matches = list(re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not input_matches:
                # Match input() function call and preserve quotes
                input_matches = list(re.finditer(r'input\s*\((.*?)\)', extracted_content, flags))
            if not input_matches:
                # Match <input> tag with optional closing tag, strip spaces
                input_matches = list(re.finditer(r"<input>\s*(.*?)(?:</input>|\s*$)", extracted_content, flags))
            if not input_matches:
                # Match "The input is" pattern case-insensitively
                input_matches = list(re.finditer(r"the input is\s*(.*?)\.?$", extracted_content, flags))
            # if still no input matches, use the extracted answer as the input
            # Don't strip() here to preserve quotes
            input_snippet = input_matches[-1].group(1) if input_matches else extracted_content
            return input_snippet

        if return_output:
            output_matches = list(re.finditer(output_pattern, extracted_content, flags))
            if not output_matches:
                # Try alternative pattern without explicit output block
                output_matches = list(re.finditer(r"# Output:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not output_matches:
                # Match output() function call and preserve quotes
                output_matches = list(re.finditer(r'output\s*\((.*?)\)', extracted_content, flags))
            if not output_matches:
                # Match <output> tag with optional closing tag, strip spaces
                output_matches = list(re.finditer(r"<output>\s*(.*?)(?:</output>|\s*$)", extracted_content, flags))
            if not output_matches:
                # Match "The output is" pattern case-insensitively, strip space after "is" and period at end
                output_matches = list(re.finditer(r"the output is\s*(.*?)\.?$", extracted_content, flags))
            # if still no output matches, use the extracted answer as the output
            output_snippet = output_matches[-1].group(1) if output_matches else extracted_content
            return output_snippet

    def _get_data_dict(self, data_item: DataProtoItem, problem_type: str, executor, banned_words: List[str], uid: str, banned_assertion_keywords: List[str]) -> Dict:
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch['extra_info']
        non_special_tokens_sequences_str = self.tokenizer.decode(self.tokenizer.encode(sequences_str), skip_special_tokens=True)
        
        generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')
        extracted_content = extract_answer(generation, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)
        thought = extract_thought(generation)

        data_dict = {
            'generation': generation,
            'data_source': data_source,
            'ground_truth': ground_truth,
            'extra_info': extra_info,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        if problem_type.startswith('gen'):
            data_dict['references'] = [ref['snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if problem_type != 'gen_code_f':
                data_dict['composite_functions'] = data_item.non_tensor_batch['extra_info']['composite_functions'].tolist()
            else:
                data_dict['imports'] = [ref['imports'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if self.use_original_code_as_ref:
                data_dict['original_references'] = [ref['original_snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
        elif problem_type.startswith('pred') and 'code_f' not in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['input'] = data_item.non_tensor_batch['extra_info']['input']
            data_dict['output'] = data_item.non_tensor_batch['extra_info']['output']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])
        elif problem_type.startswith('pred') and 'code_f' in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['given_inputs'] = data_item.non_tensor_batch['extra_info']['given_inputs']
            data_dict['given_outputs'] = data_item.non_tensor_batch['extra_info']['given_outputs']
            data_dict['hidden_inputs'] = data_item.non_tensor_batch['extra_info']['hidden_inputs']
            data_dict['hidden_outputs'] = data_item.non_tensor_batch['extra_info']['hidden_outputs']
            data_dict['message'] = data_item.non_tensor_batch['extra_info']['message']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])

        # if QA task, we only need to check the format
        if problem_type is None:
            format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
            data_dict['format_score'] = format_score
            return data_dict
        # first go through, we only checking the format
        elif problem_type.startswith('gen') and 'code_f' not in problem_type:
            success, result = parse_code_input_output(
                extracted_content,
                parse_output=False,
                remove_after_return=self.generation_reward_config.remove_after_return and self.split == 'train',
                remove_comments=self.generation_reward_config.remove_comments and self.split == 'train',
                remove_print=self.generation_reward_config.remove_print and self.split == 'train',
                reject_multiple_functions=self.generation_reward_config.reject_multiple_functions,
                f_replace_location=self.generation_reward_config.f_replace_location,
                reject_test_input_in_code=self.generation_reward_config.reject_test_input_in_code,
                code_location=self.generation_reward_config.code_location,
            )
            if len(data_dict['composite_functions']) > 0 and success:
                # first, check if the composite function names are redefined in the code, which we do not allow
                success = check_no_definitions(result['code'], [f'g_{i}' for i in range(len(data_dict['composite_functions']))])
                if not success: # if the composite function names are redefined, we do not allow the code
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict

                composite_imports = '\n'.join(
                    '\n'.join(list(d['imports'])) if list(d['imports']) else '' for d in data_dict['composite_functions']
                ).strip()

                composite_snippets = '\n\n'.join(d['snippet'] for d in data_dict['composite_functions']).strip()

                # cache the original code
                result['original_code'] = result['code']

                result['code'] = f"{composite_imports}\n\n{composite_snippets}\n\n{result['code']}".strip()
                # TODO: composite function check
                success = check_composite_function(
                    code = result['code'],
                    composite_functions = [d['snippet'] for d in data_dict['composite_functions']],
                )
            if success:
                code_validity, output = executor.check_all(
                    code=result['code'],
                    inputs=result['input'],
                    banned_keywords=banned_words,
                    check_determinism=True,
                    imports=list(set(result['imports'])),
                    check_error=problem_type == 'gen_code_e',
                    banned_keywords_for_errors_and_exceptions=banned_assertion_keywords,
                )
                if not code_validity:
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict
                # means the code is valid, we append any good programs, but we eval format separately
                data_dict['answer'] = {
                    'snippet': result['code'],
                    'original_snippet': result['original_code'] if 'original_code' in result else result['code'],
                    'input': result['input'],
                    'output': output,
                    'imports': result['imports'],
                    'thought': thought,
                    'composite_functions': data_dict['composite_functions']
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        elif problem_type == 'gen_code_f':
            success, result = parse_inputs_message(
                extracted_content,
                num_inputs=self.num_inputs,
            )
            if success and len(result['inputs']) == self.num_inputs: # for code_f, we need to ensure the number of inputs is correct
                outputs = []
                for inpt in result['inputs']:
                    code_validity, output = executor.check_all(
                        code=data_dict['references'][0],
                        inputs=inpt,
                        banned_keywords=[],
                        check_determinism=True,
                        imports=data_dict['imports'][0],
                        check_error=False,
                        banned_keywords_for_errors_and_exceptions=[],
                    )
                    if not code_validity:
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                    outputs.append(output)
                data_dict['answer'] = {
                    'snippet': data_dict['references'][0],
                    'inputs': result['inputs'],
                    'outputs': outputs,
                    'message': result['message'],
                    'imports': data_dict['imports'][0],
                    'thought': thought,
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        # if prediction is the task
        elif problem_type.startswith('pred'):
            # Check required blocks
            if problem_type.endswith('code_i'): # parse input
                input_snippet = self.extract_input_output(extracted_content, return_input=True, return_output=False) \
                    if self.extract_code_block else extracted_content
                if input_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = input_snippet
                return data_dict
            elif problem_type.endswith('code_o') or problem_type.endswith('code_e'): #  parse output, code_e format is same as code_o
                output_snippet = self.extract_input_output(extracted_content, return_input=False, return_output=True) \
                    if self.extract_code_block else extracted_content
                if output_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = output_snippet
                return data_dict
            elif problem_type.endswith('code_f'):
                success, code_snippet = parse_code_function(extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = {
                    'snippet': code_snippet,
                    'given_inputs': data_dict['given_inputs'],
                    'given_outputs': data_dict['given_outputs'],
                    'hidden_inputs': data_dict['hidden_inputs'],
                    'hidden_outputs': data_dict['hidden_outputs'],
                    'message': data_dict['message'],
                    'imports': data_dict['imports'],
                    'thought': thought,
                    'gold_program': data_dict['program'],
                }
                return data_dict
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = defaultdict(list)
        data_dicts = []
        valid_programs = [] # for gen tasks, we need to store the valid programs for later use, ignore this if prediction task
        correct_predictions = []
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info']['metric'] for d in data]
            problem_type = 'pred' # dummy set
        else:
            problem_types = [problem_type] * len(data)
        PrettyPrinter.section_header("Getting Data Dicts")
        for i in range(len(data)): # get format score
            data_dict = self._get_data_dict(data[i], problem_types[i], executor, banned_words, uids[i], banned_assertion_keywords)
            data_dicts.append(data_dict)

        if problem_type.startswith('gen') and rollout_actor_wg is not None: # get generation rewards
            PrettyPrinter.section_header("Generating Rewards for Generation Tasks")
            rewards, valid_programs = self._get_problem_generator_rewards_and_valid_programs(
                data_dicts=data_dicts,
                problem_type=problem_type,
                n_samples=n_samples,
                rollout_actor_wg=rollout_actor_wg,
                executor=executor,
                input_type_counters=input_type_counters,
                output_type_counters=output_type_counters,
                error_type_counters=error_type_counters,
            )
            PrettyPrinter.section_header("Combining Rewards for Generation Tasks")
            for i in range(len(data_dicts)):
                uid = data_dicts[i]['uid']
                valid_response_length = data_dicts[i]['valid_response_length']
                acc_reward = rewards[uid]['accuracy']
                format_reward = data_dicts[i]['format_score']
                if format_reward > 0:
                    if acc_reward > 0:
                        # Helper function for safe reward combination
                        def _combine_rewards(acc, intrinsic_components, method):
                            components = [c for c in intrinsic_components if c is not None]

                            if method == 'sum':
                                return acc + sum(components) if components else acc
                            elif method == 'multiply':
                                return acc * np.prod([c for c in components]) if components else acc
                            elif method == 'sum_multiply':
                                return acc + np.prod([c for c in components]) if components else acc
                            elif method == 'multiply_sum':
                                return acc * sum(components) if components else acc
                            else:
                                raise ValueError(f"Unknown combination method: {method}")

                        intrinsic_reward_components = []
                        if problem_type.endswith('code_f'):
                            if self.generation_reward_config.f_input_answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_input_answer_diversity_reward.coef * rewards[uid]['input_type_counts'],
                                    self.generation_reward_config.f_input_answer_diversity_reward.max))
                            if self.generation_reward_config.f_output_answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.f_output_answer_diversity_reward.coef * rewards[uid]['output_type_counts'],
                                    self.generation_reward_config.f_output_answer_diversity_reward.max))
                        else:
                            if self.generation_reward_config.complexity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.complexity_reward.coef * rewards[uid]['complexity'],
                                    self.generation_reward_config.complexity_reward.max))
                            if self.generation_reward_config.mean_edit_distance_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.mean_edit_distance_reward.coef * rewards[uid]['mean_edit_distance'],
                                    self.generation_reward_config.mean_edit_distance_reward.max))
                            if self.generation_reward_config.halstead_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.halstead_reward.coef * rewards[uid]['halstead'],
                                    self.generation_reward_config.halstead_reward.max))
                            if self.generation_reward_config.answer_diversity_reward.enabled:
                                intrinsic_reward_components.append(min(self.generation_reward_config.answer_diversity_reward.coef * rewards[uid]['type_counts'],
                                    self.generation_reward_config.answer_diversity_reward.max))

                        final_reward = _combine_rewards(acc_reward, intrinsic_reward_components, self.generation_reward_config.intrinsic_combine_method)
                        reward_tensor[i, valid_response_length - 1] = final_reward
                    else:
                        reward_tensor[i, valid_response_length - 1] = -0.5
                else:
                    reward_tensor[i, valid_response_length - 1] = -1.0
            all_scores['accuracy'] = [rewards[uid]['accuracy'] for uid in rewards]
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            if 'code_f' not in problem_type:
                all_scores['answer_diversity'] = [rewards[uid]['type_counts'] for uid in rewards]
                all_scores['complexity'] = [rewards[uid]['complexity'] for uid in rewards]
                all_scores['mean_edit_distance'] = [rewards[uid]['mean_edit_distance'] for uid in rewards]
                all_scores['halstead'] = [rewards[uid]['halstead'] for uid in rewards]
            else:
                all_scores['input_answer_diversity'] = [rewards[uid]['input_type_counts'] for uid in rewards]
                all_scores['output_answer_diversity'] = [rewards[uid]['output_type_counts'] for uid in rewards]
        elif problem_type.startswith('pred'): # get prediction rewards
            PrettyPrinter.section_header("Getting Prediction Rewards")
            all_scores['none_count'] = 0
            acc_rewards = []
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                imports = data_dict['imports']
                if not problem_type.endswith('code_f'):
                    answer = data_dict['answer']
                    gold_input = data_dict['input']
                    gold_output = data_dict['output']
                    program = data_dict['program']
                else:
                    hidden_inputs = data_dict['hidden_inputs']
                    hidden_outputs = data_dict['hidden_outputs']
                if not data_dicts[i]['format_score']: # early stop if the format is not correct
                    acc_reward = 0.
                elif problem_types[i].endswith('code_i'):
                    acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_i, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_o'):
                    acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                    # problematic, but we did not encounter too much of this
                    if acc_reward is None:
                        all_scores['none_count'] += 1
                        acc_reward = 0.
                        print(f"error in pred_code_o, not in [0, 1], acc_reward={acc_reward}\nprogram:\n{program}\n---\nanswer:\n{answer}\n---\nimports:\n{imports}\n---\n")
                    if acc_reward > 0.0:
                        correct_predictions.append(data_dict)
                elif problem_types[i].endswith('code_e'): # string matching for errors
                    answer = answer.split(' ')[0].split(':')[0]
                    if answer.lower() == gold_output.lower():
                        acc_reward = 1.0
                        correct_predictions.append(data_dict)
                    else:
                        acc_reward = 0.0
                elif problem_types[i].endswith('code_f'):
                    input_output_accs = []
                    program = data_dict['answer']['snippet']
                    for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                        input_output_acc = executor.eval_input_prediction(
                            code=program,
                            gold_output=outpt,
                            agent_input=inpt,
                            imports=list(set(imports)),
                        )
                        if input_output_acc is not None:
                            input_output_accs.append(input_output_acc)
                    acc_reward = np.mean(input_output_accs) if input_output_accs else 0.0
                    if self.code_f_reward_type == 'binary':
                        acc_reward = 1.0 if acc_reward == 1.0 else 0.0
                    elif self.code_f_reward_type == 'if_one_correct':
                        acc_reward = 1.0 if acc_reward > 0 else 0.0
                    # note that if code_f_reward_type==accuracy, it is already handled in the above
                    if acc_reward > 0:
                        correct_predictions.append(data_dict)
                else:
                    raise ValueError(f"Invalid problem type: {problem_types[i]}")

                if self.split == 'train':
                    if data_dicts[i]['format_score'] > 0:
                        if acc_reward > 0:
                            reward_tensor[i, valid_response_length - 1] = acc_reward
                        else:
                            reward_tensor[i, valid_response_length - 1] = -0.5
                    else:
                        reward_tensor[i, valid_response_length - 1] = -1.0
                elif self.split == 'test': # only acc reward for eval
                    if acc_reward > 0:
                        reward_tensor[i, valid_response_length - 1] = 1.0
                    else:
                        reward_tensor[i, valid_response_length - 1] = 0.0
                acc_rewards.append(acc_reward)
            all_scores['accuracy'] = acc_rewards
            all_scores['format_score'] = [data_dicts[i]['format_score'] for i in range(len(data))]
            all_scores['none_ratio'] = all_scores['none_count'] / len(data)
        return reward_tensor, all_scores, valid_programs, correct_predictions

    def _get_problem_generator_rewards_and_valid_programs(
        self,
        data_dicts: List[Dict],
        problem_type: str,
        n_samples: int,
        rollout_actor_wg,
        executor,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
        """This function uses samples to estimate the accuracy reward for each program, also computes the code complexity and mean edit distance of generated programs.
            Also returns the valid programs using filters.
            Args:
                data_dicts: List[Dict]: A list of data dictionaries.
                problem_type: str: The type of problem.
                n_samples: int: The number of samples to use.
                rollout_actor_wg: RolloutActorWG: The rollout actor.
                executor: PythonExecutor/CodeBoxExecutor: The executor.
                type_counters: Dict[str, Dict[str, int]]: The type counters.
            Returns:
               rewards: Dict[str, Dict[str, float]]: A dictionary of rewards for each program.
               valid_programs: List[Dict[str, str]]: A list of valid programs.
        """
        if problem_type.endswith('code_i'):
            type_counters = input_type_counters
        elif problem_type.endswith('code_o'):
            type_counters = output_type_counters
        elif problem_type.endswith('code_e'):
            type_counters = error_type_counters
        valid_data_dicts = [data_dict for data_dict in data_dicts if data_dict['code_validity']]
        uid2valid_dict_idx = {data_dict['uid']: i for i, data_dict in enumerate(valid_data_dicts)}
        valid_uids = [data_dict['uid'] for data_dict in data_dicts if data_dict['code_validity']]
        invalid_uids = [data_dict['uid'] for data_dict in data_dicts if not data_dict['code_validity']]
        assert len(valid_uids) + len(invalid_uids) == len(data_dicts)
        accuracies = {uid: 1.0 for uid in invalid_uids} # for invalid uids, we give maximum accuracy to the model
        rewards = defaultdict(dict)
        valid_programs = []
        if len(valid_uids) > 0:
            if self.reward_fn_extraction_type.startswith('boxed'):
                instruction_template = boxed_instruction
            elif self.reward_fn_extraction_type.startswith('answer'):
                instruction_template = instruction_following
            elif self.reward_fn_extraction_type.startswith('none'):
                instruction_template = '{}'
            else:
                raise ValueError(f"Invalid instruction type: {self.reward_fn_extraction_type}")
            prompts = []
            if problem_type.endswith('code_i'):
                pt = 'code_i'
            elif problem_type.endswith('code_o'):
                pt = 'code_o'
            elif problem_type.endswith('code_e'):
                pt = 'code_e'
            elif problem_type.endswith('code_f'):
                pt = 'code_f'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in valid_data_dicts:
                if pt == 'code_f':
                    num_given_inputs = len(data_dict['answer']['inputs']) // 2
                    num_given_outputs = len(data_dict['answer']['outputs']) // 2
                    data_dict['answer']['given_inputs'] = data_dict['answer']['inputs'][:num_given_inputs]
                    data_dict['answer']['given_outputs'] = data_dict['answer']['outputs'][:num_given_outputs]
                    data_dict['answer']['hidden_inputs'] = data_dict['answer']['inputs'][num_given_inputs:]
                    data_dict['answer']['hidden_outputs'] = data_dict['answer']['outputs'][num_given_outputs:]
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=problem_type,
                            snippet=data_dict['answer']['snippet'],
                            message=data_dict['answer']['message'],
                            input_output_pairs=zip(data_dict['answer']['given_inputs'], data_dict['answer']['given_outputs']),
                        )
                    )
                else:
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=pt,
                            snippet=data_dict['answer']['snippet'],
                            input_args=data_dict['answer']['input'],
                            output=data_dict['answer']['output'],
                        )
                    )
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': io_prompt}],
                    'uid': data_dict['uid'],
                    'problem': data_dict['answer'],
                    'data_source': data_dict['data_source'],
                    'ground_truth': data_dict['answer']['output'] if pt != 'code_f' else data_dict['answer']['snippet'],
                    'extra_info': data_dict['extra_info'],
                    'program': data_dict['answer']['snippet'],
                    'imports': data_dict['answer']['imports'],
                    'references': data_dict['references'],
                }
                if pt == 'code_f':
                    prompts_dict.update({
                        'given_inputs': data_dict['answer']['given_inputs'],
                        'given_outputs': data_dict['answer']['given_outputs'],
                        'hidden_inputs': data_dict['answer']['hidden_inputs'],
                        'hidden_outputs': data_dict['answer']['hidden_outputs'],
                        'message': data_dict['answer']['message'],
                    })
                else:
                    prompts_dict.update({
                        'input': data_dict['answer']['input'],
                        'output': data_dict['answer']['output'],
                        'original_program': data_dict['answer']['original_snippet'],
                        'composite_functions': data_dict['answer']['composite_functions'],
                    })
                prompts.append(prompts_dict)

            # sampling to estimate the accuracy
            PrettyPrinter.section_header("Sampling to Estimate Accuracy")
            prompts = prompts * n_samples # repeat the prompts n_samples times
            pd.DataFrame(prompts).to_parquet(f'{self.output_path}/temp.parquet') # RLHFDataset expects parquet
            temp_data = RLHFDataset(
                parquet_files=f'{self.output_path}/temp.parquet',
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(f'{self.output_path}/temp.parquet') # we do not need this file after we load in the dataset
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)

            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            assert len(dataloader) == 1
            data = next(iter(dataloader))
            batch = DataProto.from_single_dict(data)
            gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': False,
            }
            # pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            # unpad
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            batch = batch.union(output_gen_batch)
            batched_responses = []
            for b in batch:
                batch_dict = {
                        'extracted_answers': extract_answer(
                            self.tokenizer.decode(b.batch['responses'], skip_special_tokens=True),
                            self.reward_fn_extraction_type,
                            boxed_retry=self.boxed_retry,
                        ),
                        'uid': b.non_tensor_batch['uid'],
                        'problem': b.non_tensor_batch['problem'],
                        'data_source': b.non_tensor_batch['data_source'],
                        'extra_info': b.non_tensor_batch['extra_info'],
                        'program': b.non_tensor_batch['program'],
                        'references': b.non_tensor_batch['references'],
                        'imports': b.non_tensor_batch['imports'],
                    }
                if pt == 'code_f':
                    batch_dict.update({
                        'given_inputs': b.non_tensor_batch['given_inputs'],
                        'given_outputs': b.non_tensor_batch['given_outputs'],
                        'hidden_inputs': b.non_tensor_batch['hidden_inputs'],
                        'hidden_outputs': b.non_tensor_batch['hidden_outputs'],
                        'message': b.non_tensor_batch['message'],
                    })
                else:
                    batch_dict.update({
                        'input': b.non_tensor_batch['input'],
                        'output': b.non_tensor_batch['output'],
                        'original_program': b.non_tensor_batch['original_program'],
                        'composite_functions': b.non_tensor_batch['composite_functions'].tolist(),
                    })
                batched_responses.append(batch_dict)
            df = pd.DataFrame(batched_responses)

            # estimating accuracy using python executor
            PrettyPrinter.section_header("Estimating Accuracy Using Python Executor")
            for valid_uid in valid_uids:
                df_valid = df[df['uid'] == valid_uid]
                if df_valid.empty: # the prompt got filtered out TODO: check
                    accuracies[valid_uid] = 0.0
                    continue
                if pt != 'code_f':
                    answers = [self.extract_input_output(
                        answer,
                        return_input=problem_type.endswith('code_i'),
                        return_output=(problem_type.endswith('code_o') or problem_type.endswith('code_e')) # code_e output format is same as code_o
                    ) for answer in df_valid['extracted_answers'].tolist()]
                else:
                    answers = [parse_code_function(answer) for answer in df_valid['extracted_answers'].tolist()]
                answer_cache = {} # for the same uid, the answer is the same and the program is assumed to be deterministic, therefore we cache the answer -> accuracy mapping
                if pt == 'code_f':
                    hidden_outputs = df_valid['hidden_outputs'].tolist()[0].tolist()
                    hidden_inputs = df_valid['hidden_inputs'].tolist()[0].tolist()
                else:
                    gold_output = df_valid['output'].tolist()[0]
                    program = df_valid['program'].tolist()[0]
                    # gold_input = df_valid['input'].tolist()[0]
                imports = df_valid['imports'].tolist()[0]
                problem_accuracies = []
                if problem_type.endswith('code_i'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                                answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen I batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_o'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                                answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen O batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_e'): # string matching for errors
                    for answer in answers:
                        answer = answer.split(' ')[0].split(':')[0]
                        if answer.lower() == gold_output.lower():
                            problem_accuracies.append(1.0)
                        else:
                            problem_accuracies.append(0.0)
                elif problem_type.endswith('code_f'):
                    for parsed, answer in answers: # for each input/output set, we sampled n codes to estimate the accuracy
                        if not parsed: # the code answer is not parsed, we assume the code is not valid
                            problem_accuracies.append(0.0)
                            continue
                        code_accuracies = []
                        for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                            code_accuracies.append(executor.eval_input_prediction(code=answer, gold_output=outpt, agent_input=inpt, imports=list(set(imports))))
                        answer_acc = np.mean([a for a in code_accuracies if a is not None]) if code_accuracies else 0.0
                        if self.code_f_reward_type == 'binary':
                            problem_accuracies.append(1.0 if answer_acc == 1.0 else 0.0)
                        elif self.code_f_reward_type == 'if_one_correct':
                            problem_accuracies.append(1.0 if answer_acc > 0 else 0.0)
                        elif self.code_f_reward_type == 'accuracy':
                            problem_accuracies.append(answer_acc)
                        else:
                            raise ValueError(f"Invalid code_f_reward_type: {self.code_f_reward_type}")
                accuracies[valid_uid] = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0

                # filtering valid programs
                if self.valid_program_filter == 'all':
                    valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_one':
                    if accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_extremes':
                    if accuracies[valid_uid] > 0.0 and accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                else:
                    raise ValueError(f"Invalid valid program filter: {self.valid_program_filter}")

        # getting other rewards
        PrettyPrinter.section_header("Getting Other Rewards")
        # outputting rewards
        for d in data_dicts:
            uid = d['uid']
            if self.generation_reward_config.generation_accuracy_convertion == 'one_minus':
                rewards[uid]['accuracy'] = (1 - accuracies[uid]) if accuracies[uid] > 0 else 0.0
            elif self.generation_reward_config.generation_accuracy_convertion == 'inverse':
                rewards[uid]['accuracy'] = 1 - accuracies[uid]
            else:
                raise ValueError(f"Invalid generation accuracy convertion: {self.generation_reward_config.generation_accuracy_convertion}")

        if not problem_type.endswith('code_f'):
            code_key = 'original_snippet' if self.use_original_code_as_ref else 'snippet'
            reference_key = 'original_references' if self.use_original_code_as_ref else 'references'
            if problem_type.endswith('code_i'):
                type_counter_key = 'input'
            elif problem_type.endswith('code_o'):
                type_counter_key = 'output'
            elif problem_type.endswith('code_e'):
                type_counter_key = 'error'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['complexity'] = get_code_complexity_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['mean_edit_distance'] = np.mean([ast_edit_distance(data_dict['answer'][code_key], ref) for ref in data_dict[reference_key]]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['halstead'] = get_halstead_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['type_counts'] = get_type_counts_reward(
                    data_dict['answer'][type_counter_key],
                    type_counters,
                    hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                ) if 'answer' in data_dict else 0.0
            if self.debug:
                for data_dict in data_dicts:
                    if 'answer' in data_dict:
                        continue
        else:
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['input_type_counts'] = []
                rewards[data_dict['uid']]['output_type_counts'] = []
                if 'answer' in data_dict:
                    for inpt, outpt in zip(data_dict['answer']['inputs'], data_dict['answer']['outputs']):
                        rewards[data_dict['uid']]['input_type_counts'].append(get_type_counts_reward(
                            inpt,
                            input_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                        rewards[data_dict['uid']]['output_type_counts'].append(get_type_counts_reward(
                            outpt,
                            output_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                    rewards[data_dict['uid']]['input_type_counts'] = np.mean(rewards[data_dict['uid']]['input_type_counts'])
                    rewards[data_dict['uid']]['output_type_counts'] = np.mean(rewards[data_dict['uid']]['output_type_counts'])
                else:
                    rewards[data_dict['uid']]['input_type_counts'] = 0.0
                    rewards[data_dict['uid']]['output_type_counts'] = 0.0

        # turn into normal dict
        rewards = dict(rewards)
        return rewards, valid_programs

from functools import partial
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoTokenizer
from openai import OpenAI
import json
import numpy as np
import random

class GeneralIORewardManager:
    """The reward manager for GeneralIO tasks."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        eval_reward_config: Dict[str, Any],
        model_name: str,
        max_prompt_length: int = 8192,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        stream: bool = True,
        boxed_retry: bool = False,
        judge_with_actor: bool = False,
        train_judge: bool = False,
        infer_together: bool = False,
        prompt_manager=None,
        normalize_scores_in_batch: bool = False,
        use_format_reward: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.split = split
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.splitter = splitter
        self.output_path = output_path
        self.generation_reward_config = generation_reward_config
        self.eval_reward_config = eval_reward_config
        self.model_name = model_name
        self.max_prompt_length = max_prompt_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.boxed_retry = boxed_retry
        self.judge_with_actor = judge_with_actor
        self.train_judge = train_judge
        self.infer_together = infer_together
        self.prompt_manager = prompt_manager
        self.normalize_scores_in_batch = normalize_scores_in_batch
        self.use_format_reward = use_format_reward

        assert not self.train_judge or self.judge_with_actor, "judge_with_actor must be activated if train_judge is True"

    def set_prompt_manager(self, prompt_manager):
        """Set or update the prompt_manager for this reward manager."""
        self.prompt_manager = prompt_manager
        
        # Initialize the external LLM client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )

    def extract_score_from_tags(self, text: str) -> List[float]:
        """
        Extract numerical scores from <score></score> tags with priority:
        1. Integer numbers
        2. Floating point numbers  
        3. Fractions (e.g., 3/5, 7/10)
        
        Args:
            text (str): The input text containing <score></score> tags
            
        Returns:
            List[float]: List of extracted numerical values as floats
        """
        import re
        from fractions import Fraction
        
        # Find all content within <score></score> tags (case insensitive)
        score_tags = re.findall(r'<score>(.*?)</score>', text, re.IGNORECASE | re.DOTALL)
        
        extracted_scores = []
        
        for tag_content in score_tags:
            tag_content = tag_content.strip()
            score_value = None
            
            # Priority 1: Try to extract fraction first (most specific)
            fraction_match = re.search(r'\b(\d+)/(\d+)\b', tag_content)
            if fraction_match:
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                if denominator != 0:
                    score_value = float(Fraction(numerator, denominator))
            else:
                # Priority 2: Try to extract floating point number
                float_match = re.search(r'\b(\d+\.\d+)\b', tag_content)
                if float_match:
                    score_value = float(float_match.group(1))
                else:
                    # Priority 3: Try to extract integer
                    integer_match = re.search(r'\b(\d+)\b', tag_content)
                    if integer_match:
                        score_value = float(integer_match.group(1))
            
            if score_value is not None:
                extracted_scores.append(score_value)
        
        return extracted_scores

    def _generate_prompt_for_judge(self, question: str = None, answer: str = None, prompt_type: str = "answer") -> str:
        """Generate a prompt for the LLM to judge the quality of a question and response."""
        assert prompt_type in ["answer", "question", "together"], f"Invalid prompt type: {prompt_type}"
        
        # Use prompt_manager if available, otherwise fall back to hardcoded prompts
        if hasattr(self, 'prompt_manager') and self.prompt_manager is not None:
            # Determine the correct judge type based on prompt_type
            if prompt_type == "answer":
                judge_type = "judge_answer"
            elif prompt_type == "question":
                judge_type = "judge_question"  
            elif prompt_type == "together":
                judge_type = "judge_together"
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type}")
            
            # Get optimized prompt from prompt_manager
            judge_prompt_template = self.prompt_manager.get_template(judge_type)
            
            # Fill in the template variables
            if prompt_type == "answer":
                prompt = judge_prompt_template.format(question=question, answer=answer)
            elif prompt_type == "question":
                prompt = judge_prompt_template.format(question=question)
            elif prompt_type == "together":
                prompt = judge_prompt_template.format(question=question, answer=answer)
                
            return prompt
        
        # Fallback to hardcoded prompts if prompt_manager is not available
        prompt = ""
        if prompt_type == "answer":
            prompt = f"""Please evaluate the following solution to a question/problem.

Question/Problem: {question}

Generated Solution: {answer}

First, analyze the solution in the <think> and </think> tags below:

<think>
Consider the following criteria when evaluating:
- Is the solution correct and accurate?
- Is it complete and comprehensive?
- Does it properly address the question?
- Is the reasoning clear and logical?
- Analyze any strengths and weaknesses
- Determine what score is most appropriate

[Write your detailed analysis here]
</think>

Then provide a score from 1 to 10 between <score> and </score> where:
- 10 means the solution is perfect, complete, and correct
- 8-9 means the solution is mostly correct but may have minor issues  
- 5-7 means the solution is partially correct but has significant issues
- 2-4 means the solution has some merit but is largely incorrect
- 1 means the solution is completely wrong or irrelevant

<score>X</score> (where X is an integer from 1 to 10)
"""
        elif prompt_type == "question":
            prompt = f"""Please evaluate the quality of the following question generation.
Question: {question}

First, analyze the question in the <think> tags below:

<think>
Consider the following criteria when evaluating:
- Is the question clear and well-formed?
- Is it complete and understandable?
- Does it make logical sense?
- Is it relevant and appropriate?
- Analyze any strengths and weaknesses
- Determine what score is most appropriate

[Write your detailed analysis here]
</think>

Then provide a score from 1 to 10 between <score> and </score> where:
- 10 means the question is perfect, complete, and clear
- 8-9 means the question is mostly clear but may have minor issues
- 5-7 means the question is partially clear but has significant issues
- 2-4 means the question has some merit but is largely unclear or irrelevant
- 1 means the question is completely wrong or irrelevant (Also rate as 1 if the question is not a valid question)

<score>X</score> (where X is an integer from 1 to 10)
"""

        elif prompt_type == "together":
            prompt = f"""Please evaluate the quality of the following question and answer pair.
Question: {question}

Provided Answer: {answer}

First, analyze the question in the <think> tags below:

<think>
Consider the following criteria when evaluating:
- Is the question clear and well-formed?
- Is it complete and understandable?
- Does it make logical sense?
- Is it relevant and appropriate?
- Analyze any strengths and weaknesses
- Determine what score is most appropriate

[Write your detailed analysis here]
</think>

Then provide a score from 1 to 10 between <score> and </score> for the question where:
- 10 means the question is perfect, complete, and clear
- 8-9 means the question is mostly clear but may have minor issues
- 5-7 means the question is partially clear but has significant issues
- 2-4 means the question has some merit but is largely unclear or irrelevant
- 1 means the question is completely wrong or irrelevant (Also rate as 1 if the question is not a valid question)

<score>X</score> (where X is an integer from 1 to 10)

Then analyze the answer in the <think> and </think> tags below:

<think>
Consider the following criteria when evaluating:
- Is the answer correct and accurate?
- Is it complete and comprehensive?
- Does it properly address the question?
- Is it well-structured and clear?
- Analyze any strengths and weaknesses
- Determine what score is most appropriate

[Write your detailed analysis here]
</think>

Finally provide a score from 1 to 10 between <score> and </score> for the answerwhere:
- 10 means the answer is perfect, complete, and correct
- 8-9 means the answer is mostly correct but may have minor issues
- 5-7 means the answer is partially correct but has significant issues
- 2-4 means the answer has some merit but is largely incorrect
- 1 means the answer is completely wrong or irrelevant

<score>X</score> (where X is an integer from 1 to 10)

Please make sure that your response contains only two pairs of <score> and </score> tags, one for the question and one for the answer. The question score always comes first, followed by the answer score.

When you reference your own scores, you do not use the <score> and </score> tags. You only use these tags to provide the final scores for the question and answer.
"""
        return prompt
    
    def rollout_with_actors(self, dataset_file: str, rollout_actor_wg) -> DataProto:
        dataset = RLHFDataset(
            parquet_files=dataset_file,
            tokenizer=self.tokenizer,
            prompt_key='prompt',
            max_prompt_length=self.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=False,
            truncation='error'
        )
        if os.path.exists(dataset_file):
            os.remove(dataset_file)

        sampler = torch.utils.data.SequentialSampler(data_source=dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        data = next(iter(dataloader))
        batch = DataProto.from_single_dict(data)
        gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
        gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': True,
        }

        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
        out_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
        out_gen_batch = unpad_dataproto(out_gen_batch_padded, pad_size=pad_size)
        batch = batch.union(out_gen_batch)

        return batch
    
    def normalize_scores_in_batch(self, scores):
        """
            Normalize scores in a batch so that they fall into a normal distribution
            Normalize scores that are below 8, higher scores often means success and has no need to be normalized
            Case study needed here
        """
        scores = np.array(scores)
        if np.std(scores) < 1e-5:
            return [0.5] * len(scores)
        normalized_scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-5)
        normalized_scores = 0.5 + 0.1 * normalized_scores
        normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
        normalized_scores = np.round(normalized_scores)
        return normalized_scores.tolist()

    def _get_all_scores(self, data_dicts: List[Dict], rollout_actor_wg, n_samples: int, problem_type: str) -> List[float]:
        """
        Get all scores for both gen and pred.
        """
        
        avg_gen_scores = []
        avg_pred_scores = []

        if problem_type.startswith("judge"):
            try:           
                def count_score_tags(text: str) -> tuple:
                    """Count <score> and </score> tags in the text."""
                    score_open_count = text.count('<score>')
                    score_close_count = text.count('</score>')
                    return score_open_count, score_close_count
                
                # Calculate combined rewards for each response
                judge_scores = []
                for data_dict in data_dicts:
                    # Get the response text from data_dict
                    response_text = data_dict.get('generation', '')
                    
                    if response_text == "":
                        judge_scores.append(0.5)  # Neutral score for empty responses
                        continue
                        
                    open_tags, close_tags = count_score_tags(response_text)
                    
                    # Perfect score if exactly one <score> and one </score> tag
                    if open_tags == 1 and close_tags == 1:
                        tag_score = 1.0
                    elif open_tags == close_tags:
                        if  open_tags >= 2:
                            tag_score = 0.5
                        else:
                            tag_score = 0.0
                    else:
                        tag_score = 0.0
                    
                    judge_scores.append(tag_score)
                
                avg_gen_scores = judge_scores
                avg_pred_scores = []  # No pred scores for judge tasks
                
            except Exception as e:
                print(f"Error in judge reward computation: {e}")
                avg_gen_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores
                avg_pred_scores = []
            
            return avg_gen_scores, avg_pred_scores
        
        if problem_type.startswith("pred"):
            try:
                if not self.judge_with_actor:
                    for data_dict in data_dicts:
                        avg_pred_scores.append(self._generate_llm_response(self._generate_prompt_for_pred(data_dict, self.infer_together))[0])
                    if self.normalize_scores_in_batch:
                        avg_pred_scores = self.normalize_scores_in_batch(avg_pred_scores)
                    return avg_gen_scores, avg_pred_scores

                if rollout_actor_wg is None:
                    return [0.5] * len(data_dicts), [0.5] * len(data_dicts)  # Default neutral difficulty score
                
                # Build evaluation prompts (one per answer)
                eval_prompts = []
                for data_dict in data_dicts:
                    eval_text = self._generate_prompt_for_pred(data_dict, self.infer_together)
                    eval_prompts.append({
                        'prompt': [{'role': 'user', 'content': eval_text}],
                        'uid': data_dict['uid'],
                    })

                # Optionally repeat judgments (n_samples) if desired
                eval_prompts = eval_prompts  # could multiply by another factor if multi-judging needed
                # [TODO] Add n_samples here maybe

                temp_judge_file = f'{self.output_path}/temp_generalio_judge.parquet'
                pd.DataFrame(eval_prompts).to_parquet(temp_judge_file)

                judge_batch = self.rollout_with_actors(
                    dataset_file=temp_judge_file,
                    rollout_actor_wg=rollout_actor_wg
                )

                # Create uid to prompt mapping for dumping
                uid_to_prompt = {}
                for ep in eval_prompts:
                    uid_to_prompt[ep['uid']] = ep['prompt'][0]['content']
                
                # Collect raw judge outputs
                uid2_a_scores = defaultdict(list)
                
                # Open file for dumping evaluation results
                eval_file = open("eval2.txt", "a")

                for jb in judge_batch:
                    uid = jb.non_tensor_batch['uid']
                    text = self.tokenizer.decode(jb.batch['responses'], skip_special_tokens=True)
                    scores = self.extract_score_from_tags(text)
                    
                    # Dump prompt and evaluation response to file separated by ===
                    eval_file.write(uid_to_prompt.get(uid, "Prompt not found"))
                    eval_file.write("\n==================\n")
                    eval_file.write(text)
                    eval_file.write("\n==================\n\n")
                    eval_file.flush()
                    
                    print("Actor evaluation response:", text)
                    try:
                        # assert len(scores) == 1, f"Expected one score in the response, got: {text}"
                        if self.infer_together:
                            # assert len(scores) == 2, f"Expected two scores in the response for together, got: {text}"
                            a = (scores[1] - 1) / 9.0
                            uid2_a_scores[uid].append(min(1.0, max(0.0, a)))
                            continue
                        else:
                            a = (scores[0] - 1) / 9.0
                            uid2_a_scores[uid].append(min(1.0, max(0.0, a)))
                    except:
                        print("Falling back to neutral scores.")
                        pass
                
                # Close the evaluation file
                eval_file.close()

                # Aggregate per original data_dict
                for data_dict in data_dicts:
                    uid = data_dict['uid']
                    if uid2_a_scores.get(uid):
                        avg_pred_scores.append(float(np.mean(uid2_a_scores[uid])))
                    else:
                        avg_pred_scores.append(0.5)

            except Exception as e:
                print(f"Error in pred score computation: {e}")
                avg_pred_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores

            if self.normalize_scores_in_batch:
                avg_pred_scores = self.normalize_scores_in_batch(avg_pred_scores)
            return avg_gen_scores, avg_pred_scores
        
        if problem_type.startswith("gen") and not self.infer_together:
            try:
                if not self.judge_with_actor:
                    for data_dict in data_dicts:
                        avg_gen_scores.append(self._generate_llm_response(self._generate_prompt_for_gen(data_dict))[0])
                else:

                    if rollout_actor_wg is None:
                        return [0.5] * len(data_dicts), [0.5] * len(data_dicts)  # Default neutral difficulty score

                    # Build evaluation prompts (one per answer)
                    eval_prompts = []
                    for data_dict in data_dicts:
                        eval_text = self._generate_prompt_for_gen(data_dict)
                        eval_prompts.append({
                            'prompt': [{'role': 'user', 'content': eval_text}],
                            'uid': data_dict['uid'],
                        })

                    # Optionally repeat judgments (n_samples) if desired
                    eval_prompts = eval_prompts  # could multiply by another factor if multi-judging needed

                    temp_judge_file = f'{self.output_path}/temp_generalio_judge.parquet'
                    pd.DataFrame(eval_prompts).to_parquet(temp_judge_file)

                    judge_batch = self.rollout_with_actors(
                        dataset_file=temp_judge_file,
                        rollout_actor_wg=rollout_actor_wg
                    )

                    # Create uid to prompt mapping for dumping
                    uid_to_prompt = {}
                    for ep in eval_prompts:
                        uid_to_prompt[ep['uid']] = ep['prompt'][0]['content']

                    # Collect raw judge outputs
                    uid2_q_scores = defaultdict(list)

                    # Open file for dumping evaluation results
                    eval_file = open("eval2.txt", "a")
                    for jb in judge_batch:
                        uid = jb.non_tensor_batch['uid']
                        text = self.tokenizer.decode(jb.batch['responses'], skip_special_tokens=True)
                        scores = self.extract_score_from_tags(text)
                        
                        # Dump prompt and evaluation response to file separated by ===
                        eval_file.write(uid_to_prompt.get(uid, "Prompt not found"))
                        eval_file.write("\n==================\n")
                        eval_file.write(text)
                        eval_file.write("\n==================\n\n")
                        eval_file.flush()
                        
                        print("Actor evaluation response:", text)
                        try:
                            # assert len(scores) == 1, f"Expected one score in the response, got: {text}"
                            q = (scores[0] - 1) / 9.0
                            uid2_q_scores[uid].append(min(1.0, max(0.0, q)))
                        except:
                            print("Falling back to neutral scores.")
                            pass

                    # Close evaluation file
                    eval_file.close()

                    # Aggregate per original data_dict
                    for data_dict in data_dicts:
                        uid = data_dict['uid']
                        if uid2_q_scores.get(uid):
                            avg_gen_scores.append(float(np.mean(uid2_q_scores[uid])))
                        else:
                            avg_gen_scores.append(0.5)
            except Exception as e:
                print(f"Error in gen score computation: {e}")
                avg_gen_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores

        # check rollout actor for gen and together problems
        if rollout_actor_wg is None:
            print(f"[DEBUG] rollout_actor_wg is None, returning default scores for problem_type: {problem_type}")
            return [0.5] * len(data_dicts), [0.5] * len(data_dicts)  # Default neutral difficulty score

        print(f"[DEBUG] rollout_actor_wg is not None, proceeding with actual scoring for problem_type: {problem_type}")

        try:
            # Define helper function for extracting questions
            import re
            def extract_question(text):
                pattern = r'<question>(.*?)</question>'
                matches = re.findall(pattern, text, re.DOTALL)
                return matches
            
            # Create prompts for sampling
            prompts = []
            for data_dict in data_dicts:
                question = extract_question(data_dict.get('generation', '<question></question>').split("[Your designed task]")[-1])
                if question != []:
                    question = question[-1]
                else:
                    question = "The question is a invalid question"
                    PrettyPrinter.status("No question tags found in response", "", "warning")
                
                #question = data_dict.get('question', '')
                prompt_text = f"Please solve the following question/problem:\n\n{question}"
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': prompt_text}],
                    'uid': data_dict['uid'],
                    'question': question,
                }
                PrettyPrinter.section_header(f"Creating prompt for question: {question}")
                prompts.append(prompts_dict)

            # Repeat prompts n_samples times for sampling
            repeated_prompts = prompts * n_samples
            
            # Create temporary parquet file for sampling
            temp_file = f'{self.output_path}/temp_generalio_sampling.parquet'
            pd.DataFrame(repeated_prompts).to_parquet(temp_file)
            
            batch = self.rollout_with_actors(
                dataset_file=temp_file,
                rollout_actor_wg=rollout_actor_wg
            )

            batched_responses = []
            for b in batch:
                response_text = self.tokenizer.decode(b.batch['responses'], skip_special_tokens=True)
                batched_responses.append({
                    'response': response_text,
                    'uid': b.non_tensor_batch['uid'],
                    'question': b.non_tensor_batch['question'],
                })
            
            # Group responses by UID and evaluate with LLM
            responses_by_uid = defaultdict(list)
            for response in batched_responses:
                responses_by_uid[response['uid']].append(response)

            if self.judge_with_actor:
                # Build evaluation prompts (one per generated answer)
                eval_prompts = []
                for uid, resp_list in responses_by_uid.items():
                    for idx, resp in enumerate(resp_list):
                        eval_text = self._generate_prompt_for_judge(resp["question"], resp["response"], prompt_type="together" if self.infer_together else "answer")
                        if self.infer_together:
                            PrettyPrinter.section_header(f"Creating prompt for actor evaluation of question and answer for difficulty score: {resp['question']}\n\n{resp['response']}")
                        else:
                            PrettyPrinter.section_header(f"Creating prompt for actor evaluation of answer for difficulty score:\n\n[Question]: {resp['question']}\n\n[Answer]: {resp['response']}")
                        eval_prompts.append({
                            'prompt': [{'role': 'user', 'content': eval_text}],
                            'uid': uid,
                        })

                # Optionally repeat judgments (n_samples) if desired
                eval_prompts = eval_prompts  # could multiply by another factor if multi-judging needed

                temp_judge_file = f'{self.output_path}/temp_generalio_judge.parquet'
                pd.DataFrame(eval_prompts).to_parquet(temp_judge_file)

                judge_batch = self.rollout_with_actors(
                    dataset_file=temp_judge_file,
                    rollout_actor_wg=rollout_actor_wg
                )

                # Collect raw judge outputs
                uid2_q_scores = defaultdict(list)
                uid2_a_scores = defaultdict(list)

                for jb in judge_batch:
                    uid = jb.non_tensor_batch['uid']
                    text = self.tokenizer.decode(jb.batch['responses'], skip_special_tokens=True)
                    scores = self.extract_score_from_tags(text)
                    print("Actor evaluation response:", text)
                    try:
                        if self.infer_together:
                            # assert len(scores) == 2, f"Expected two scores in the response, got: {text}"
                            q = (scores[0] - 1) / 9.0
                            a = (scores[1] - 1) / 9.0
                            uid2_q_scores[uid].append(min(1.0, max(0.0, q)))
                            uid2_a_scores[uid].append(min(1.0, max(0.0, a)))
                        else:
                            # assert len(scores) == 1, f"Expected one score in the response, got: {text}"
                            a = (scores[0] - 1) / 9.0
                            uid2_a_scores[uid].append(min(1.0, max(0.0, a)))
                    except:
                        print("Falling back to neutral scores.")
                        pass

                # Aggregate per original data_dict
                for data_dict in data_dicts:
                    uid = data_dict['uid']
                    if uid2_a_scores.get(uid):
                        avg_pred_scores.append(float(np.mean(uid2_a_scores[uid])))
                    else:
                        avg_pred_scores.append(0.5)
                    if self.infer_together:
                        if uid2_q_scores.get(uid):
                            avg_gen_scores.append(float(np.mean(uid2_q_scores[uid])))
                        else:
                            avg_gen_scores.append(0.5)
            else:
                # Calculate average scores for each question
                for data_dict in data_dicts:
                    uid = data_dict['uid']
                    if uid in responses_by_uid:
                        gen_scores = []
                        pred_scores = []
                        for response_data in responses_by_uid[uid]:
                            # Create evaluation prompt
                            eval_prompt = self._generate_prompt_for_judge(
                                response_data["question"],
                                response_data["response"],
                                prompt_type = "together" if self.infer_together else "answer",
                            )
                            if self.infer_together:
                                PrettyPrinter.section_header(f"Creating prompt for actor evaluation of question and answer for difficulty score:: {response_data['question']}\n\n{response_data['response']}")
                            else:
                                PrettyPrinter.section_header(f"Creating prompt for actor evaluation of answer for difficulty score:\n\n[Question]: {response_data['question']}\n\n[Answer]: {response_data['response']}")
                            
                            scores = self._generate_llm_response(eval_prompt)
                            try:
                                if not self.infer_together:
                                    # assert len(scores) == 1, f"Expected one score in the response, got: {scores}"
                                    pred_scores.append(scores[0])
                                else:
                                    # assert len(scores) == 2, f"Expected two scores in the response, got: {scores}"
                                    gen_scores.append(scores[0])
                                    pred_scores.append(scores[1])
                            except:
                                pass
                        
                        avg_pred_score = np.mean(pred_scores) if pred_scores else 0.5
                        avg_pred_scores.append(avg_pred_score)
                        if self.infer_together:
                            avg_gen_score = np.mean(gen_scores) if gen_scores else 0.5
                            avg_gen_scores.append(avg_gen_score)
                    else:
                        avg_pred_scores.append(0.5)
                        if self.infer_together:
                            avg_gen_scores.append(0.5)
                        
        except Exception as e:
            import traceback
            print(f"Error in solver score computation: {e}")
            print(f"Error traceback: {traceback.format_exc()}")
            avg_pred_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores
            if self.infer_together:
                avg_gen_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores
        
        if self.normalize_scores_in_batch:
            avg_gen_scores = self.normalize_scores_in_batch(avg_gen_scores)
            avg_pred_scores = self.normalize_scores_in_batch(avg_pred_scores)

        return avg_gen_scores, avg_pred_scores
     
    def _generate_llm_response(self, prompt: str) -> List[float]:
        """Call the external LLM for evaluation."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=self.stream
            )

            result = ""
            
            if self.stream:
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                result = result.strip()
            else:
                result = completion.choices[0].message.content.strip()
                
            print(f"LLM Response: {result}")  # Debugging output
                
            # Use the new extract_score_from_tags function
            extracted_scores = self.extract_score_from_tags(result)
            if extracted_scores:
                # Convert scores to 0-1 range
                normalized_scores = []
                for score in extracted_scores:
                    # Assume scores are in 1-10 range, normalize to 0-1
                    if score >= 1 and score <= 10:
                        normalized_score = (score - 1) / 9.0
                        normalized_scores.append(min(1.0, max(0.0, normalized_score)))
                    else:
                        # If score is already normalized or in different range, keep as is
                        normalized_scores.append(min(1.0, max(0.0, score)))
                return normalized_scores
            else:
                # Fallback: try to extract any number between 1-10
                print("Falling back to match any number between 1-10.")
                import re
                fallback_match = re.findall(r'(\d+)', result)
                if fallback_match:
                    score_list = []
                    for score in fallback_match:
                        score = int(score)
                        if 1 <= score <= 10:
                            score = (score - 1) / 9.0
                            score_list.append(min(1.0, max(0.0, score)))
                    if score_list:
                        return score_list
                return [0.0]
        except Exception as e:
            print(f"Error in LLM response generation: {e}")
            return [0.0]

    def _generate_prompt_for_gen(self, data_dict: Dict, answer=None) -> str:
        """Generate the LLM as judge prompt for evaluating the question generation quality."""
        def extract_question(text):
            pattern = r'<question>(.*?)</question>'
            matches = re.findall(pattern, text, re.DOTALL)
            return matches
        question = extract_question(data_dict.get('generation', '').split("[Your designed task]")[-1])
        if question != []:
            question = question[-1].strip()
        else:
            question = "This is not a valid question."

        if answer:
            prompt = self._generate_prompt_for_judge(question, answer, prompt_type="together")
            PrettyPrinter.code_block(f"Generated prompt for question generation evaluation with paired answer:\n{prompt}")
        else:
            prompt = self._generate_prompt_for_judge(question, None, prompt_type="question")
            PrettyPrinter.code_block(f"Generated prompt for question generation evaluation:\n{prompt}")
        return prompt

    def _generate_prompt_for_pred(self, data_dict: Dict, infer_together=False) -> str:
        """Generate the LLM prompt for 'pred' type problems."""
        question = data_dict.get('question', '')
        answer = data_dict.get('answer', data_dict.get('generation', '')).split('[Your final answer to the question, structured and clear, without restating the question]')[-1]

        if infer_together:
            prompt = self._generate_prompt_for_judge(question, answer, prompt_type="together")
            PrettyPrinter.code_block(f"Generated prompt for answer evaluation with question:\n{prompt}")
        else:
            prompt = self._generate_prompt_for_judge(question, answer, prompt_type="answer")
            PrettyPrinter.code_block(f"Generated prompt for answer evaluation:\n{prompt}")
        return prompt

    def _compute_score_for_gen(self, data_dict: Dict, external_llm_score: float, solver_avg_score: float) -> float:
        """For 'gen' problem type, combine LLM score and solver score."""
        return 0.5 * external_llm_score + 0.5 * (1 - solver_avg_score)

    def _compute_score_for_pred(self, external_llm_score: float) -> float:
        """For 'pred' problem type, use the LLM score as the final score."""
        return external_llm_score

    def _evaluate_gen(self, data_dict: Dict, solver_avg_score: float, rollout_actor_wg=None) -> float:
        """Evaluate a 'gen' problem type."""
        prompt = self._generate_prompt_for_gen(data_dict)
        external_llm_score = self._get_evaluation_score(prompt, rollout_actor_wg)
        final_score = self._compute_score_for_gen(data_dict, external_llm_score, solver_avg_score)
        return final_score

    def _evaluate_pred(self, data_dict: Dict, rollout_actor_wg=None) -> float:
        """Evaluate a 'pred' problem type."""
        prompt = self._generate_prompt_for_pred(data_dict)
        external_llm_score = self._get_evaluation_score(prompt, rollout_actor_wg)
        final_score = self._compute_score_for_pred(external_llm_score)
        return final_score
        final_score = self._compute_score_for_pred(external_llm_score)
        return final_score

    def _get_data_dict(self, data_item: DataProtoItem, problem_type: str, banned_words: List[str], uid: str, banned_assertion_keywords: List[str]) -> Dict:
        """
        Extract data dictionary for GeneralIO tasks.
        This method is simplified compared to CodeIORewardManager since GeneralIO tasks
        don't require code execution validation.
        """
        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        # Extract relevant information from non_tensor_batch
        # For GeneralIO tasks, the ground_truth is typically the answer to the question
        ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', '')
        if not ground_truth:
            ground_truth = data_item.non_tensor_batch.get('ground_truth', '')
        
        # The problem field contains the question for GeneralIO tasks
        question = data_item.non_tensor_batch.get('problem', '')
        if not question:
            # Fallback to extracting from reward_model or other locations
            question = data_item.non_tensor_batch.get('question', '')
        answer = data_item.non_tensor_batch.get('answer', '')
        
        data_source = data_item.non_tensor_batch.get('data_source', '')
        extra_info = data_item.non_tensor_batch.get('extra_info', {})
        
        non_special_tokens_sequences_str = self.tokenizer.decode(
            self.tokenizer.encode(sequences_str), skip_special_tokens=True
        )
        
        # Extract generation from response
        try:
            generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')
        except (IndexError, AttributeError):
            generation = non_special_tokens_sequences_str.strip()
        
        extracted_content = extract_answer(generation, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)
        thought = extract_thought(generation)

        data_dict = {
            'generation': generation,
            'question': question,  # The question/problem to solve
            'answer': answer,
            'ground_truth': ground_truth,  # The expected answer
            'data_source': data_source,
            'extra_info': extra_info,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        
        # Set answer for evaluation
        if problem_type is None or problem_type.startswith('pred'):
            data_dict['answer'] = extracted_content if extracted_content else generation
        elif problem_type.startswith('gen'):
            data_dict['answer'] = generation
        
        return data_dict
        

    def __call__(
        self,
        data: DataProto,
        problem_type: str = None,
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
        general_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict, List[Dict]]:
        """
        Main method for computing rewards for GeneralIO tasks.
        
        For 'gen' type: Uses LLM judge score + difficulty score (1 - solver average)
        For 'pred' type: Uses LLM judge score directly
        
        Returns:
            reward_tensor: Tensor of rewards for each sequence
            all_scores: Dictionary containing various scores and metrics
            valid_questions: List of valid questions/responses for dataset expansion
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = defaultdict(list)
        data_dicts = []
        valid_data = []  # For GeneralIO tasks, we track valid questions or question and answer pairs
        uids = np.array([str(uuid.uuid4()) for _ in range(len(data))], dtype=object)
        
        if problem_type is None:
            problem_types = [d.non_tensor_batch['extra_info'].get('metric', 'pred') for d in data]
            problem_type = 'pred'  # dummy set
        else:
            problem_types = [problem_type] * len(data)
            
        PrettyPrinter.section_header("Getting Data Dicts for GeneralIO")
        for i in range(len(data)):
            data_dict = self._get_data_dict(data[i], problem_types[i], banned_words, uids[i], banned_assertion_keywords)
            data_dicts.append(data_dict)

        if problem_type.startswith('gen') and rollout_actor_wg is not None:
            PrettyPrinter.section_header("Computing Generation Rewards for GeneralIO Tasks")
            
            # Step 1: Get solver average scores and llm scores(possibly) from actor model
            print(f"[DEBUG] Getting all scores for problem_type: {problem_type}, rollout_actor_wg is None: {rollout_actor_wg is None}")
            llm_scores, solver_avg_scores = self._get_all_scores(data_dicts, rollout_actor_wg, n_samples, problem_type)
            print(f"[DEBUG] Got scores - LLM scores: {llm_scores[:3]}..., Solver avg scores: {solver_avg_scores[:3]}...")
            
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                
                def extract_question(text):
                    pattern = r'<question>(.*?)</question>'
                    matches = re.findall(pattern, text, re.DOTALL)
                    return [m.strip() for m in matches]
                question = extract_question(data_dict.get('generation', '<question></question>'))
                if len(question) > 3:
                    # The exact number depends on the prompt given
                    question = question[-1]
                else:
                    question = None
                # First check validity and then add
                # If validity check not passed, all should be reset

                if question:
                    difficulty_score = 1 - solver_avg_scores[i]
                    final_score = 0.5 * llm_scores[i] + 0.5 * difficulty_score
                    
                    print(f"[DEBUG] Item {i}: solver_avg={solver_avg_scores[i]:.4f}, difficulty={difficulty_score:.4f}, llm={llm_scores[i]:.4f}, final={final_score:.4f}")
                    
                    reward_tensor[i, valid_response_length - 1] = final_score
                    all_scores['llm_judge_score'].append(llm_scores[i])
                    all_scores['difficulty_score'].append(difficulty_score)
                    all_scores['combined_score'].append(final_score)
                    if llm_scores[i] > 0.3:
                        # Only add question to dataset if it is valid
                        # This only works when using strict prompt for evaluating questions
                        valid_data.append({
                            'question': question,
                            'generation': data_dict.get('generation', ''),
                            'thought': data_dict.get('thought', ''),
                            'answer': data_dict.get('generation', ''),
                            'uid': data_dict['uid'],
                        })
                    else:
                        # Override scores if not valid
                        # Dump bad question to file
                        with open('bad_question.txt', 'a') as f:
                            f.write(f"Question: {question}\n")
                            f.write("==============================================\n")
                            if 'thought' in data_dict:
                                f.write(f"Thought: {data_dict['thought']}\n")
                                f.write("==============================================\n")
                            if 'generation' in data_dict:
                                f.write(f"Generation: {data_dict['generation']}\n")
                                f.write("==============================================\n")
                            f.write(f"LLM Score: {llm_scores[i]}\n")
                            f.write("==============================================\n")
                            f.write("\n")
                        reward_tensor[i, valid_response_length - 1] = llm_scores[i]
                        all_scores['difficulty_score'][-1] = llm_scores[i]
                        all_scores['combined_score'][-1] = llm_scores[i]
                else:
                    print("Question format failed. Penalized and falling back")
                    reward_tensor[i, valid_response_length - 1] = 0.0
                    all_scores['llm_judge_score'].append(0.0)
                    all_scores['difficulty_score'].append(0.0)
                    all_scores['combined_score'].append(0.0)

            all_scores['solver_avg_scores'] = solver_avg_scores
        elif problem_type.startswith('pred'):
            PrettyPrinter.section_header("Computing Prediction Rewards for GeneralIO Tasks")
            
            llm_scores = []
            _, llm_scores = self._get_all_scores(data_dicts, rollout_actor_wg, n_samples, problem_type)
            
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                
                if self.split == 'train':
                    reward_tensor[i, valid_response_length - 1] = llm_scores[i]
                    valid_data.append({
                        'question': data_dict.get('question', ''),
                        'answer': data_dict.get('answer', ''),
                        'thought': data_dict.get('thought', ''),
                        'reward_model': {
                            'ground_truth': data_dict.get('ground_truth', ''),
                        },
                        'uid': data_dict['uid'],
                    })
                elif self.split == 'test':
                    reward_tensor[i, valid_response_length - 1] = llm_scores[i]
                    # test split pairs not saved
                    # valid_data.append({
                    #     'question': data_dict.get('question', ''),
                    #     'answer': data_dict.get('answer', ''),
                    #     'thought': data_dict.get('thought', ''),
                    #     'uid': data_dict['uid'],
                    # })
                
                all_scores['llm_judge_score'].append(llm_scores[i])
            
            all_scores['accuracy'] = all_scores['llm_judge_score']  # For compatibility
        elif problem_type.startswith("judge"):
            PrettyPrinter.section_header("Computing Judge Rewards for GeneralIO Tasks")

            rewards, _ = self._get_all_scores(data_dicts, rollout_actor_wg, n_samples, problem_type)

            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                reward_tensor[i, valid_response_length - 1] = rewards[i]

            all_scores['format_reward'] = rewards  # Default neutral score for judge tasks
        else:
            # For other cases or when rollout_actor_wg is None
            PrettyPrinter.section_header("Computing Default Rewards for GeneralIO Tasks")
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                reward_tensor[i, valid_response_length - 1] = 0.5  # Default neutral reward
            
            all_scores['accuracy'] = [0.5] * len(data_dicts)

        return reward_tensor, all_scores, valid_data


def evaluate_single_item(args):
    """
    Independent function to evaluate a single item using LLM.
    This function can be called in parallel processes.
    """
    (item_data, model_name, temperature, max_tokens, top_p, stream) = args
    
    try:
        # Create a new OpenAI client for this process
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        
        model_answer = item_data['model_answer']
        ground_truth = item_data['ground_truth']
        metric_type = item_data['metric_type']
        
        # Generate evaluation prompt
        if metric_type == "math_accuracy":
            prompt = f"""Please evaluate whether the following mathematical answer is correct.

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze both answers in the <think> tags below:

<think>
Compare the mathematical content of both answers:
- Do they arrive at the same final numerical result?
- Is the reasoning correct in the model's answer?
- Are there any mathematical errors?
- Do they represent the same solution even if expressed differently?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model's answer is mathematically correct and matches the ground truth
- FALSE means the model's answer is mathematically incorrect or doesn't match the ground truth

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "multiple_choice_accuracy":
            prompt = f"""Please evaluate whether the multiple choice answer is correct.

Correct Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model select the same choice letter as the ground truth?
(If the model did not provide a letter, count it as incorrect)

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model selected the exact correct choice
- FALSE means the model selected the wrong choice

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "truthfulness_accuracy":
            prompt = f"""Please evaluate the truthfulness and accuracy of the answer.

Expected Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Evaluate the truthfulness:
- Is the model's answer factually accurate?
- Does it align with the expected truthful response?
- Are there any misleading or false statements?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is truthful and accurate:
- TRUE means the answer is truthful and accurate
- FALSE means the answer contains false or misleading information

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        else:
            # General accuracy evaluation
            prompt = f"""Please evaluate whether the answer is correct and appropriate.

Expected Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model's answer provide accurate content?
- Is the content appropriate and relevant?
- How well does it match the expected response?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the answer is correct and appropriate
- FALSE means the answer is incorrect or inappropriate

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        # Call LLM for evaluation
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream
        )
        
        if stream:
            result = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content
        else:
            result = completion.choices[0].message.content.strip()
        
        # Extract TRUE/FALSE from <answer></answer> tags
        answer_match = re.search(r'<answer>(TRUE|FALSE)</answer>', result, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).upper()
            score = 1.0 if answer == "TRUE" else 0.0
        else:
            # Fallback: look for TRUE/FALSE anywhere in the response
            if re.search(r'\bTRUE\b', result, re.IGNORECASE):
                score = 1.0
            elif re.search(r'\bFALSE\b', result, re.IGNORECASE):
                score = 0.0
            else:
                # If no clear TRUE/FALSE found, default to FALSE (incorrect)
                score = 0.0
        
        return {
            'index': item_data['index'],
            'score': score,
            'evaluation_result': result
        }
        
    except Exception as e:
        print(f"Error evaluating item {item_data.get('index', 'unknown')}: {e}")
        return {
            'index': item_data.get('index', -1),
            'score': 0.0,
            'evaluation_result': f"Error: {str(e)}"
        }


class BenchmarkEvaluationRewardManager:
    """Reward manager for evaluating on standard benchmarks like MATH, GSM8K, HellaSwag, etc."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model_name: str = "meta/llama-3.1-405b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_p: float = 0.95,
        stream: bool = False,
        max_workers: int = 10,  # Number of parallel processes
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.max_workers = min(max_workers, mp.cpu_count())  # Don't exceed CPU count
        
        # Initialize the external LLM client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        
    def _generate_llm_evaluation(self, model_answer: str, ground_truth: str, metric_type: str) -> float:
        """Use LLM to evaluate if model answer matches ground truth."""
        
        if metric_type == "math_accuracy":
            prompt = f"""Please evaluate whether the following mathematical answer is correct.

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze both answers in the <think> tags below:

<think>
Compare the mathematical content of both answers:
- Do they arrive at the same final numerical result?
- Is the reasoning correct in the model's answer?
- Are there any mathematical errors?
- Do they represent the same solution even if expressed differently?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model's answer is mathematically correct and matches the ground truth
- FALSE means the model's answer is mathematically incorrect or doesn't match the ground truth

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "multiple_choice_accuracy":
            prompt = f"""Please evaluate whether the multiple choice answer is correct.

Correct Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model select the same choice letter as the ground truth?
(If the model did not provide a letter, count it as incorrect)

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model selected the exact correct choice
- FALSE means the model selected the wrong choice

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "truthfulness_accuracy":
            prompt = f"""Please evaluate the truthfulness and accuracy of the answer.

Expected Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Evaluate the truthfulness:
- Is the model's answer factually accurate?
- Does it align with the expected truthful response?
- Are there any misleading or false statements?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is truthful and accurate:
- TRUE means the answer is truthful and accurate
- FALSE means the answer contains false or misleading information

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        else:
            # General accuracy evaluation
            prompt = f"""Please evaluate whether the answer is correct and appropriate.

Expected Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model's answer provide accurate content?
- Is the content appropriate and relevant?
- How well does it match the expected response?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the answer is correct and appropriate
- FALSE means the answer is incorrect or inappropriate

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        try:
            PrettyPrinter.code_block(f"Generated LLM Evaluation Prompt:\n{prompt}")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=self.stream
            )
            
            if self.stream:
                result = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
            else:
                result = completion.choices[0].message.content.strip()
            
            PrettyPrinter.code_block(f"LLM Evaluation Result:\n{result}")
            # Extract TRUE/FALSE from <answer></answer> tags
            answer_match = re.search(r'<answer>(TRUE|FALSE)</answer>', result, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).upper()
                return 1.0 if answer == "TRUE" else 0.0
            else:
                # Fallback: look for TRUE/FALSE anywhere in the response
                if re.search(r'\bTRUE\b', result, re.IGNORECASE):
                    return 1.0
                elif re.search(r'\bFALSE\b', result, re.IGNORECASE):
                    return 0.0
                else:
                    # If no clear TRUE/FALSE found, default to FALSE (incorrect)
                    return 0.0
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return 0.0
    
    def _extract_model_answer(self, generation: str) -> str:
        """Extract the model's answer from the generation."""
        # Try to extract answer from common patterns
        generation = generation.strip()
        
        # Look for final answer patterns
        patterns = [
            r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so),?\s*(.+?)(?:\n|$)",
            r"\$\$(.+?)\$\$",  # LaTeX math
            r"####\s*(.+?)(?:\n|$)",  # GSM8K format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generation, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no specific pattern found, use the last line or last sentence
        lines = generation.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('(') and len(line) < 200:
                return line
        
        # Fallback to first 100 characters
        return generation[:100] + "..." if len(generation) > 100 else generation
    
    def _get_question_from_prompt(self, prompt_data: List[Dict]) -> str:
        """Extract question from prompt data."""
        if prompt_data and len(prompt_data) > 0:
            return prompt_data[0].get('content', '')
        return ''
    
    def __call__(
        self,
        data: DataProto,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Evaluate model generations against benchmark ground truths.
        
        Returns:
            reward_tensor: Tensor of evaluation scores
            metrics: Dictionary of evaluation metrics
        """
        
        try:
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            
            all_scores = defaultdict(list)
            correct_predictions = []
            
            PrettyPrinter.section_header("Benchmark Evaluation (Multi-process)")
            
            data_length = len(data)
            PrettyPrinter.status("Info", f"Processing {data_length} items with {self.max_workers} workers", "info")
            
            # Prepare evaluation tasks
            evaluation_tasks = []
            item_info = []  # Store original item info for later processing
            
            for i in range(data_length):
                try:
                    data_item = data[i]
                    
                    # Extract information
                    prompt_data = data_item.non_tensor_batch.get('prompt', [])
                    question = self._get_question_from_prompt(prompt_data)
                    ground_truth = data_item.non_tensor_batch.get('answer', '')
                    data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
                    extra_info = data_item.non_tensor_batch.get('extra_info', {})
                    metric_type = extra_info.get('metric', 'general_accuracy')
                    
                    # Get model generation
                    response_ids = data_item.batch['responses']
                    generation = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    model_answer = self._extract_model_answer(generation)
                    
                    # Store item info for later processing
                    valid_response_length = data_item.batch['attention_mask'][len(data_item.batch['prompts']):].sum()
                    item_info.append({
                        'index': i,
                        'question': question,
                        'model_answer': model_answer,
                        'ground_truth': ground_truth,
                        'data_source': data_source,
                        'metric_type': metric_type,
                        'valid_response_length': valid_response_length
                    })
                    
                    # Prepare task for multiprocessing
                    task_data = {
                        'index': i,
                        'model_answer': model_answer,
                        'ground_truth': ground_truth,
                        'metric_type': metric_type
                    }
                    
                    evaluation_tasks.append((
                        task_data,
                        self.model_name,
                        self.temperature,
                        self.max_tokens,
                        self.top_p,
                        self.stream
                    ))
                    
                except Exception as e:
                    PrettyPrinter.status("Error", f"Failed to prepare item {i}: {str(e)}", "error")
                    # Add default task to maintain indexing
                    task_data = {
                        'index': i,
                        'model_answer': '',
                        'ground_truth': '',
                        'metric_type': 'general_accuracy'
                    }
                    evaluation_tasks.append((
                        task_data,
                        self.model_name,
                        self.temperature,
                        self.max_tokens,
                        self.top_p,
                        self.stream
                    ))
                    item_info.append({
                        'index': i,
                        'question': '',
                        'model_answer': '',
                        'ground_truth': '',
                        'data_source': 'unknown',
                        'metric_type': 'general_accuracy',
                        'valid_response_length': 1
                    })
                    continue
            
            # Execute evaluations in parallel
            PrettyPrinter.status("Info", f"Starting parallel evaluation with {len(evaluation_tasks)} tasks", "info")
            
            evaluation_results = {}
            if self.max_workers > 1:
                # Use multiprocessing
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_index = {executor.submit(evaluate_single_item, task): task[0]['index'] for task in evaluation_tasks}
                    
                    for future in as_completed(future_to_index):
                        try:
                            result = future.result()
                            evaluation_results[result['index']] = result
                        except Exception as e:
                            index = future_to_index[future]
                            PrettyPrinter.status("Error", f"Evaluation failed for item {index}: {str(e)}", "error")
                            evaluation_results[index] = {
                                'index': index,
                                'score': 0.0,
                                'evaluation_result': f"Process error: {str(e)}"
                            }
            else:
                # Single-threaded fallback
                for task in evaluation_tasks:
                    result = evaluate_single_item(task)
                    evaluation_results[result['index']] = result
            
            # Process results and populate reward tensor
            for info in item_info:
                i = info['index']
                result = evaluation_results.get(i, {'score': 0.0})
                score = result['score']
                
                # Store score in reward tensor
                valid_response_length = info['valid_response_length']
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = score
                else:
                    reward_tensor[i, -1] = score
                
                # Track metrics
                all_scores['accuracy'].append(score)
                all_scores[f'accuracy_{info["data_source"]}'].append(score)
                all_scores[f'accuracy_{info["metric_type"]}'].append(score)
                
                # Count as correct if score is 1.0 (TRUE)
                if score == 1.0:
                    correct_predictions.append({
                        'question': info['question'],
                        'model_answer': info['model_answer'],
                        'ground_truth': info['ground_truth'],
                        'score': score,
                        'data_source': info['data_source']
                    })
                
                PrettyPrinter.status(
                    f"Sample {i+1}", 
                    f"Source: {info['data_source']}, Correct: {'✓' if score == 1.0 else '✗'}", 
                    "success" if score == 1.0 else "warning"
                )
            
            # Calculate overall metrics
            overall_accuracy = np.mean(all_scores['accuracy']) if all_scores['accuracy'] else 0.0
            
            # Calculate per-source accuracies
            source_accuracies = {}
            for key in all_scores:
                if key.startswith('accuracy_') and key != 'accuracy':
                    source_name = key.replace('accuracy_', '')
                    source_accuracies[f'val/benchmark_accuracy/{source_name}'] = np.mean(all_scores[key])
            
            metrics = {
                'val/benchmark_accuracy/overall': overall_accuracy,
                'val/benchmark_correct_count': len(correct_predictions),
                'val/benchmark_total_count': len(data),
                **source_accuracies
            }
            
            PrettyPrinter.status(
                "Evaluation Complete", 
                f"Overall Accuracy: {overall_accuracy:.3f} ({len(correct_predictions)}/{len(data)}) using {self.max_workers} workers",
                "success"
            )
            
            return reward_tensor, metrics
            
        except Exception as e:
            PrettyPrinter.status("Error", f"Benchmark evaluation failed: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            
            # Return empty results on error
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            return reward_tensor, {}
