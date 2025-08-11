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
                'validate': True,
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
        
        # Initialize the external LLM client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        
    def _generate_llm_response(self, prompt: str) -> float:
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
            
            if self.stream:
                result = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        result += chunk.choices[0].delta.content
                
                # Extract score from result
                result = result.strip()
                print(f"LLM Response: {result}")  # Debugging output
                
                # Try to extract score from <score></score> tags
                import re
                score_match = re.search(r'<score>(\d+)</score>', result, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    # Convert from 1-10 scale to 0-1 scale
                    score = (score - 1) / 9.0  # Maps 1->0, 10->1
                    return min(1.0, max(0.0, score))
                else:
                    # Fallback: try to extract any number between 1-10
                    fallback_match = re.search(r'(\d+)', result)
                    if fallback_match:
                        score = int(fallback_match.group(1))
                        if 1 <= score <= 10:
                            score = (score - 1) / 9.0
                            return min(1.0, max(0.0, score))
                    return 0.0
            else:
                result = completion.choices[0].message.content.strip()
                print(f"LLM Response: {result}")  # Debugging output
                
                # Try to extract score from <score></score> tags
                import re
                score_match = re.search(r'<score>(\d+)</score>', result, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    # Convert from 1-10 scale to 0-1 scale
                    score = (score - 1) / 9.0  # Maps 1->0, 10->1
                    return min(1.0, max(0.0, score))
                else:
                    # Fallback: try to extract any number between 1-10
                    fallback_match = re.search(r'(\d+)', result)
                    if fallback_match:
                        score = int(fallback_match.group(1))
                        if 1 <= score <= 10:
                            score = (score - 1) / 9.0
                            return min(1.0, max(0.0, score))
                    return 0.0
        except Exception as e:
            print(f"Error in LLM response generation: {e}")
            return 0.0

    def _generate_prompt_for_gen(self, data_dict: Dict) -> str:
        """Generate the LLM as judge prompt for evaluating the question genration quality."""
        def extract_question(text):
            pattern = r'<question>(.*?)</question>'
            matches = re.findall(pattern, text, re.DOTALL)
            return matches
        question = extract_question(data_dict.get('generation', '').split("[Your designed task]")[-1])
        if question != []:
            question = question[-1].strip()
        else:
            question = "This is not a valid question."

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

<score>X</score> (where X is an integer from 1 to 10)"""
        PrettyPrinter.code_block(f"Generated prompt for question generation evaluation:\n{prompt}")
        return prompt

    def _generate_prompt_for_pred(self, data_dict: Dict) -> str:
        """Generate the LLM prompt for 'pred' type problems."""
        question = data_dict.get('question', '')
        PrettyPrinter.code_block(f"Generated prompt for question evaluation:\n{question}")
        answer = data_dict.get('answer', data_dict.get('generation', '')).split('[Your final answer to the question, structured and clear, without restating the question]')[-1]
        PrettyPrinter.code_block(f"Generated answer for question evaluation:\n{answer}") 
        prompt = f"""Please evaluate the following answer to a question/problem.

Question/Problem: {question}

Provided Answer: {answer}

First, analyze the answer in the <think> and </think> tags below:

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

Then provide a score from 1 to 10 between <score> and </score> where:
- 10 means the answer is perfect, complete, and correct
- 8-9 means the answer is mostly correct but may have minor issues
- 5-7 means the answer is partially correct but has significant issues
- 2-4 means the answer has some merit but is largely incorrect
- 1 means the answer is completely wrong or irrelevant

<score>X</score> (where X is an integer from 1 to 10)"""
        return prompt

    def _compute_score_for_gen(self, data_dict: Dict, external_llm_score: float, solver_avg_score: float) -> float:
        """For 'gen' problem type, combine LLM score and solver score."""
        return 0.5 * external_llm_score + 0.5 * (1 - solver_avg_score)

    def _compute_score_for_pred(self, external_llm_score: float) -> float:
        """For 'pred' problem type, use the LLM score as the final score."""
        return external_llm_score

    def _get_solver_scores_from_actor(self, data_dicts: List[Dict], rollout_actor_wg, n_samples: int) -> List[float]:
        """
        Get solver scores by having the actor model generate n_samples responses for each question,
        then evaluate them with LLM-as-a-judge and compute average scores.
        """
        if rollout_actor_wg is None:
            return [0.5] * len(data_dicts)  # Default neutral difficulty score
        
        solver_avg_scores = []
        
        try:
            # Create prompts for sampling
            prompts = []
            for data_dict in data_dicts:
                def extract_question(text):
                    pattern = r'<question>(.*?)</question>'
                    matches = re.findall(pattern, text, re.DOTALL)
                    return matches
                question = extract_question(data_dict.get('generation', '<question></question>').split("[Your designed task]")[-1])
                if question != []:
                    question = question[-1]
                else:
                    question = "The question is a invalid question"

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
            
            # Create dataset for sampling
            temp_data = RLHFDataset(
                parquet_files=temp_file,
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Create data loader
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)
            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            
            # Generate responses
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
            
            # Pad and generate
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            
            # Process generated responses
            batch = batch.union(output_gen_batch)
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
            
            # Calculate average scores for each question
            for data_dict in data_dicts:
                uid = data_dict['uid']
                if uid in responses_by_uid:
                    scores = []
                    for response_data in responses_by_uid[uid]:
                        # Create evaluation prompt
                        eval_prompt = f"""Please evaluate the following solution to a question/problem.

Question/Problem: {response_data['question']}

Generated Solution: {response_data['response']}

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

<score>X</score> (where X is an integer from 1 to 10)"""
                        
                        score = self._generate_llm_response(eval_prompt)
                        scores.append(score)
                    
                    avg_score = np.mean(scores) if scores else 0.5
                    solver_avg_scores.append(avg_score)
                else:
                    solver_avg_scores.append(0.5)  # Default if no responses generated
                    
        except Exception as e:
            print(f"Error in solver score computation: {e}")
            solver_avg_scores = [0.5] * len(data_dicts)  # Fallback to neutral scores
        
        return solver_avg_scores

    def _evaluate_gen(self, data_dict: Dict, solver_avg_score: float) -> float:
        """Evaluate a 'gen' problem type."""
        prompt = self._generate_prompt_for_gen(data_dict)
        external_llm_score = self._generate_llm_response(prompt)
        final_score = self._compute_score_for_gen(data_dict, external_llm_score, solver_avg_score)
        return final_score

    def _evaluate_pred(self, data_dict: Dict) -> float:
        """Evaluate a 'pred' problem type."""
        prompt = self._generate_prompt_for_pred(data_dict)
        external_llm_score = self._generate_llm_response(prompt)
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
        valid_questions = []  # For GeneralIO tasks, we track valid questions instead of programs
        correct_predictions = []
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
            
            # Step 1: Get solver average scores from actor model
            solver_avg_scores = self._get_solver_scores_from_actor(data_dicts, rollout_actor_wg, n_samples)
            
            # Step 2: Evaluate each generation with LLM judge and combine with difficulty score
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                
                # Get LLM judge score for the actual generation
                external_llm_score = self._generate_llm_response(self._generate_prompt_for_gen(data_dict))
                
                # Compute combined score: LLM judge + difficulty (1 - solver average)
                difficulty_score = 1 - solver_avg_scores[i]
                final_score = 0.5 * external_llm_score + 0.5 * difficulty_score
                
                reward_tensor[i, valid_response_length - 1] = final_score
                all_scores['llm_judge_score'].append(external_llm_score)
                all_scores['difficulty_score'].append(difficulty_score)
                all_scores['combined_score'].append(final_score)
                def extract_question(text):
                    pattern = r'<question>(.*?)</question>'
                    matches = re.findall(pattern, text, re.DOTALL)
                    return matches
                question = extract_question(data_dict.get('generation', '<question></question>').split("[Your designed task]")[-1])
                if question != []:
                    question = question[-1]
                else:
                    question = None
                # For gen tasks, add to valid_questions
                if question!=None:
                    valid_questions.append({
                        'question': question,
                        'generation': data_dict.get('generation', ''),
                        'thought': data_dict.get('thought', ''),
                        'answer': data_dict.get('generation', ''),
                        'uid': data_dict['uid'],
                    })
                else:
                    PrettyPrinter.section_header(f"Processing Question {i+1}/{len(data_dicts)}")
                    PrettyPrinter.status(f"Question: {data_dict.get('question', '')}", "", "info")
                    PrettyPrinter.status(f"Generation: {data_dict.get('generation', '')}", "", "info")
                    PrettyPrinter.status(f"Thought: {data_dict.get('thought', '')}", "", "info")
                    PrettyPrinter.status(f"LLM Judge Score: {external_llm_score:.4f}", f"Difficulty Score: {difficulty_score:.4f}", "info")
                    PrettyPrinter.status(f"Combined Score: {final_score:.4f}", "", "info")
                    print("\n" + "-"*80 + "\n")
            all_scores['solver_avg_scores'] = solver_avg_scores
            
        elif problem_type.startswith('pred'):
            PrettyPrinter.section_header("Computing Prediction Rewards for GeneralIO Tasks")
            
            # For prediction tasks, use LLM judge score directly
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                
                # Get LLM judge score directly
                llm_score = self._generate_llm_response(self._generate_prompt_for_pred(data_dict))
                
                if self.split == 'train':
                    if llm_score > 0.5:  # Consider scores > 0.5 as correct
                        reward_tensor[i, valid_response_length - 1] = llm_score
                        correct_predictions.append({
                            'question': data_dict.get('question', ''),
                            'answer': data_dict.get('answer', ''),
                            'thought': data_dict.get('thought', ''),
                            'uid': data_dict['uid'],
                        })
                    else:
                        reward_tensor[i, valid_response_length - 1] = llm_score
                elif self.split == 'test':
                    reward_tensor[i, valid_response_length - 1] = llm_score
                    if llm_score > 0.5:
                        correct_predictions.append({
                            'question': data_dict.get('question', ''),
                            'answer': data_dict.get('answer', ''),
                            'thought': data_dict.get('thought', ''),
                            'uid': data_dict['uid'],
                        })
                
                all_scores['llm_judge_score'].append(llm_score)
            
            all_scores['accuracy'] = all_scores['llm_judge_score']  # For compatibility
        
        else:
            # For other cases or when rollout_actor_wg is None
            PrettyPrinter.section_header("Computing Default Rewards for GeneralIO Tasks")
            for i, data_dict in enumerate(data_dicts):
                valid_response_length = data_dict['valid_response_length']
                reward_tensor[i, valid_response_length - 1] = 0.5  # Default neutral reward
            
            all_scores['accuracy'] = [0.5] * len(data_dicts)

        return reward_tensor, all_scores, valid_questions


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
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        
        # Initialize the external LLM client
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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"
        )
        
    def _generate_llm_evaluation(self, question: str, model_answer: str, ground_truth: str, metric_type: str) -> float:
        """Use LLM to evaluate if model answer matches ground truth."""
        
        if metric_type == "math_accuracy":
            prompt = f"""Please evaluate whether the following mathematical answer is correct.

Question: {question}

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

Question: {question}

Correct Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model select the same choice letter as the ground truth?
- Is the model's reasoning sound?
- Does the answer make sense in context?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model selected the exact correct choice
- FALSE means the model selected the wrong choice

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "truthfulness_accuracy":
            prompt = f"""Please evaluate the truthfulness and accuracy of the answer.

Question: {question}

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

Question: {question}

Expected Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the answers:
- Does the model's answer address the question correctly?
- Is the content accurate and appropriate?
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
            
            PrettyPrinter.section_header("Benchmark Evaluation")
            
            data_length = len(data)
            PrettyPrinter.status("Debug", f"Data length: {data_length}", "info")
            PrettyPrinter.status("Debug", f"Data type: {type(data)}", "info")
            
            for i in range(data_length):
                try:
                    PrettyPrinter.status("Debug", f"Processing item {i}", "info")
                    data_item = data[i]
                    PrettyPrinter.status("Debug", f"Data item type: {type(data_item)}", "info")
                    
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
                    
                    # Evaluate using LLM
                    score = self._generate_llm_evaluation(question, model_answer, ground_truth, metric_type)
                    
                    # Store score in reward tensor (at the last position)
                    valid_response_length = data_item.batch['attention_mask'][len(data_item.batch['prompts']):].sum()
                    if valid_response_length > 0:
                        reward_tensor[i, valid_response_length - 1] = score
                    else:
                        reward_tensor[i, -1] = score
                    
                    # Track metrics
                    all_scores['accuracy'].append(score)
                    all_scores[f'accuracy_{data_source}'].append(score)
                    all_scores[f'accuracy_{metric_type}'].append(score)
                    
                    # Count as correct if score is 1.0 (TRUE)
                    if score == 1.0:
                        correct_predictions.append({
                            'question': question,
                            'model_answer': model_answer,
                            'ground_truth': ground_truth,
                            'score': score,
                            'data_source': data_source
                        })
                    
                    PrettyPrinter.status(
                        f"Sample {i+1}", 
                        f"Source: {data_source}, Correct: {'✓' if score == 1.0 else '✗'}", 
                        "success" if score == 1.0 else "warning"
                    )
                    
                except Exception as e:
                    PrettyPrinter.status("Error", f"Failed to process item {i}: {str(e)}", "error")
                    import traceback
                    traceback.print_exc()
                    continue
            
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
                f"Overall Accuracy: {overall_accuracy:.3f} ({len(correct_predictions)}/{len(data)})",
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
