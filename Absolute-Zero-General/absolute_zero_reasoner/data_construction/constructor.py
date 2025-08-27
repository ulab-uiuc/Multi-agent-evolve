from typing import List, Dict

from numpy import random
import pandas as pd
from transformers import AutoTokenizer

from absolute_zero_reasoner.data_construction.prompts import get_code_problem_generator_prompt, get_code_problem_predictor_prompt, get_general_generator_prompt,get_general_generation_with_reference_prompt, get_general_predictor_prompt, get_general_judger_prompt
from absolute_zero_reasoner.data_construction.process_data import boxed_instruction, instruction_following
from absolute_zero_reasoner.utils.code_utils.parsers import replace_main_function_name

def extract_question(text: str) -> str:
    """
    Extract the question part from the text.
    Assumes the question is enclosed in <question> tags.
    """
    start = text.find('<question>') + len('<question>')
    end = text.find('</question>', start)
    return text[start:end].strip() if start != -1 and end != -1 else text.strip()

import numpy as np
import pandas as pd

def get_gen_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    io_n: int,
    output_path: str,
    split: str,
    tokenizer,  # 不强依赖类型声明，避免导入问题
    weights: List[float] = None,
    include_references: bool = True,
    prompt_manager = None,  # Add prompt manager parameter
):
    return_io_data = []
    
    # Use dynamic prompt if prompt_manager is available
    if prompt_manager:
        instruction_template = prompt_manager.get_proposer_instruction()
        print(f"[DEBUG] get_gen_general_io_data: Using dynamic proposer instruction")
    else:
        instruction_template = '{}'
        print(f"[DEBUG] get_gen_general_io_data: Using default instruction template")

    # 兜底：空数据直接写空表并返回
    if not io_data:
        pd.DataFrame(return_io_data).to_parquet(output_path)
        return

    # 概率分布
    if weights is None:
        probabilities = np.full(len(io_data), 1.0 / len(io_data))
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0 or len(w) != len(io_data) or not np.isfinite(s):
            probabilities = np.full(len(io_data), 1.0 / len(io_data))
        else:
            probabilities = w / s

    idx = 0
    max_attempts = max(5 * target_data_len, 100)  # 防止无限循环
    attempts = 0

    while len(return_io_data) < target_data_len and attempts < max_attempts:
        attempts += 1

        if not include_references:
            chosen_references = []
        else:
            k = min(io_n, len(io_data))
            # 用 numpy 的 choice，并转成 Python list
            chosen_indices = np.random.choice(len(io_data), size=k, replace=False, p=probabilities)
            chosen_references = [io_data[i] for i in chosen_indices]
        if not chosen_references:
            if prompt_manager:
                # Use enhanced proposer instruction with empty references
                io_prompt = f"{instruction_template}\n\n{get_general_generator_prompt(reference_questions=chosen_references)}"
            else:
                io_prompt = instruction_template.format(
                    get_general_generator_prompt(reference_questions=chosen_references)
                )
        else:
            if prompt_manager:
                # Use enhanced proposer instruction with references
                io_prompt = f"{instruction_template}\n\n{get_general_generation_with_reference_prompt(reference_questions=chosen_references)}"
            else:
                io_prompt = instruction_template.format(
                    get_general_generation_with_reference_prompt(reference_questions=chosen_references)
                )
        # 提取 question
        # question = extract_question(io_prompt.split('### Your Task:')[1].strip())
        question = io_prompt
        # if not question:
        #     # print("No question found in the generated prompt, skipping this item.")
        #     continue

        # 过滤过长样本
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_general',
                "prompt": [{"role": "user", "content": io_prompt}],
                "question": question,
                "answer": "",
                "ability": "general",
                "reward_model": {"style": "rule", "ground_truth": ''},
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_general',
                    # 确保可序列化
                    'chosen_references': chosen_references,
                }
            }
            return_io_data.append(io_item)
            idx += 1

    # 不足就上采样补齐（前提：io_data 非空）
    while len(return_io_data) < target_data_len:
        j = np.random.randint(0, len(io_data))  # 上界开区间，不会越界
        return_io_data.append(io_data[j])

    # 输出到 parquet
    pd.DataFrame(return_io_data).to_parquet(output_path)

def get_pred_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    prompt_manager = None,  # Add prompt manager parameter
):
    return_io_data = []
    
    # Use dynamic prompt if prompt_manager is available
    if prompt_manager:
        instruction_template = prompt_manager.get_solver_instruction("{}")
        print(f"[DEBUG] get_pred_general_io_data: Using dynamic solver instruction")
    else:
        instruction_template = '{}'
        print(f"[DEBUG] get_pred_general_io_data: Using default instruction template")

    for idx, io_item in enumerate(io_data):
        if prompt_manager:
            # Use prompt manager to get enhanced solver instruction
            io_prompt = prompt_manager.get_solver_instruction(io_item['question'])
        else:
            # Use traditional template approach
            io_prompt = instruction_template.format(
                get_general_predictor_prompt(
                    question=io_item['question'],
                )
            )
        print(f"Generated prompt: {io_prompt}")
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": io_item['question'],
                "answer": "",
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": io_item.get('answer',''),
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_general',
                }
            }
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

def get_judge_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    prompt_manager=None,
):
    return_io_data = []
    instruction_template = '{}'

    for idx, io_item in enumerate(io_data):
        io_prompt = instruction_template.format(
            get_general_judger_prompt(
                question=io_item['question'],
                answer=io_item['answer'],
                prompt_manager=prompt_manager,
            )
        )
        print(f"Generated prompt: {io_prompt}")
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'judge_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": io_item['question'],
                "answer": io_item['answer'],
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": io_item['reward_model']['ground_truth'],
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'judge_general',
                }
            }
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

def get_gen_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    io_n: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    banned_keywords: List[str],
    banned_assertion_keywords: List[str],
    weights: List[float] = None,
    enable_composite_function: bool = False,
    composite_function_n_min: int = -1,
    composite_function_n_max: int = -1,
    composite_chance: float = 0.5,
    remove_after_return: bool = False,
    num_inputs: int = 10,
    remove_input_from_snippet: bool = False,
    include_references: bool = True,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    if weights is None:
        probabilities = [1.0 / len(io_data)] * len(io_data)
    else:
        # Normalize weights to form a probability distribution
        probabilities = [float(w)/sum(weights) for w in weights]
    
    idx = 0

    while len(return_io_data) < target_data_len:
        if not include_references and problem_type != 'code_f':
            chosen_references = []
        else:
            chosen_references = random.choice(io_data, size=min(io_n, len(io_data)), replace=False, p=probabilities)
        # composite functions is not used for code_f problem type
        if problem_type != 'code_f' and composite_function_n_max > 0 and enable_composite_function and random.random() <= composite_chance and len(chosen_references) > composite_function_n_max:
            # TODO: we only allow composite to sample from code snippets without composite functions
            io_without_composite_function_indices = [i for i in range(len(io_data)) if not io_data[i]['composite_functions']]
            io_without_composite_function_data = [io_data[i] for i in io_without_composite_function_indices]
            io_without_composite_function_weights = [probabilities[i] for i in io_without_composite_function_indices]
            # normalize the weights
            io_without_composite_function_probabilities = [w / sum(io_without_composite_function_weights) for w in io_without_composite_function_weights]
            # number of composite functions to sample is either fixed or random
            composite_function_n = composite_function_n_min if composite_function_n_min == composite_function_n_max else random.randint(composite_function_n_min, composite_function_n_max)
            composite_functions = random.choice(io_without_composite_function_data, size=composite_function_n, replace=False, p=io_without_composite_function_probabilities)
            for i, composite_function in enumerate(composite_functions):
                # TODO: need to also replace recursively called composite functions, ignore functions that have f as the last letter, only for function call f()
                composite_functions[i]['snippet'] = replace_main_function_name(composite_function['snippet'], 'f', f'g_{i}')
            imports = []
        else:
            composite_functions = []
            if include_references:
                imports = chosen_references[0]['imports']
            else:
                imports = []
        io_prompt = instruction_template.format(
            get_code_problem_generator_prompt(
                problem_type=problem_type,
                reference_snippets=chosen_references,
                banned_keywords=banned_keywords,
                banned_assertion_keywords=banned_assertion_keywords,
                composite_functions=composite_functions,
                remove_after_return=remove_after_return,
                num_inputs=num_inputs,
                remove_input_from_snippet=remove_input_from_snippet,
            )
        )
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "problem": '',
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": '',
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_' + problem_type,
                    'chosen_references': chosen_references,
                    'composite_functions': composite_functions,
                    'imports': imports,
                }
            }
            return_io_data.append(io_item)
            idx += 1

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = io_data[random.randint(0, len(io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)


def get_pred_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    for idx, io_item in enumerate(io_data):
        if problem_type == 'code_i':
            ground_truth = io_item['input']
        elif problem_type == 'code_o':
            ground_truth = io_item['output']
        elif problem_type == 'code_e':
            ground_truth = io_item['output']
        elif problem_type == 'code_f':
            ground_truth = io_item['snippet']
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")
        if problem_type == 'code_f':
            num_given_inputs = len(io_item['inputs']) // 2
            num_given_outputs = len(io_item['outputs']) // 2
            given_inputs = list(io_item['inputs'][:num_given_inputs])
            given_outputs = list(io_item['outputs'][:num_given_outputs])
            hidden_inputs = list(io_item['inputs'][num_given_inputs:])
            hidden_outputs = list(io_item['outputs'][num_given_outputs:])
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    message=io_item['message'],
                    input_output_pairs=zip(given_inputs, given_outputs),
                )
            )
        else:
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    input_args=io_item['input'],
                    output=io_item['output'],
                )
            )
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "problem": io_item['snippet'],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_' + problem_type,
                    'imports': io_item['imports'],
                }
            }
            if problem_type == 'code_f': # for code_f, we need to split the inputs and outputs into given and hidden, only show part of the inputs and outputs to the model
                output_io_item['extra_info']['given_inputs'] = given_inputs
                output_io_item['extra_info']['given_outputs'] = given_outputs
                output_io_item['extra_info']['hidden_inputs'] = hidden_inputs
                output_io_item['extra_info']['hidden_outputs'] = hidden_outputs
                output_io_item['extra_info']['message'] = io_item['message']
            else:
                output_io_item['extra_info']['input'] = io_item['input']
                output_io_item['extra_info']['output'] = io_item['output']
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

