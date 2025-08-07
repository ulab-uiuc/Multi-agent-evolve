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
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
from glob import glob
import argparse

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


METRIC_MAP = {
        'aime2024': 'math',
        'aime2025': 'math',
        'gpqa': 'mc',
        'amc2023': 'math',
        'math500': 'math',
        'minerva': 'math',
        'olympiadbench': 'math',
        'math': 'math',
        'orz': 'math',
        'simplerl': 'math',
        'hmmt_2025': 'math',
        'hmmt_2024': 'math',
        'live_math_bench': 'math',
        'big_math': 'math',
        'deepscaler': 'math',
        "math3to5": 'math',
        'dapo': 'math',
    }

instruction_following = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>"
boxed_instruction = "{}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


# add a row to each data item that represents a unique id
def make_map_fn(split, question_key, answer_key, do_extract_solution, reward_fn_extraction_type, nothink = False):

    def process_fn(example, idx):
        question = example.pop(question_key)

        if reward_fn_extraction_type == 'answer':
            formatted_question = (instruction_following if not nothink else instruction_following.strip(' <think>')).format(question)
        elif reward_fn_extraction_type == 'boxed':
            formatted_question = boxed_instruction.format(question)
        elif reward_fn_extraction_type == 'none':
            formatted_question = question
        # gpqa has this string in the question
        if reward_fn_extraction_type != 'boxed':
            remove_string = "\n\nPlease reason step-by-step and put your choice letter without any other text with \\boxed{} in the end."
            replacement_string = '\n\nPlease reason step-by-step and put your choice letter without any other text with <answer> </answer> in the end.'
            formatted_question = formatted_question.replace(remove_string, replacement_string)

        answer = example.pop(answer_key)
        if do_extract_solution:
            solution = extract_solution(answer)
        else:
            solution = answer
        data_source = example.pop('data_source')
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": formatted_question
            }],
            "problem": question,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'metric': METRIC_MAP[data_source],
            }
        }
        return data

    return process_fn


def process_data(args):
    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    if args.train_set == 'math':
        dataset = datasets.load_dataset('DigitalLearningGmbH/MATH-lighteval', trust_remote_code=True)
    elif args.train_set == 'orz':
        dataset = datasets.load_dataset('json', data_files='data/orz_math_57k_collected.json')
        dataset = dataset.map(lambda x: {'problem': x['0']['value'], 'solution': x['1']['ground_truth']['value']})
    elif args.train_set == 'simplerl':
        dataset = datasets.load_dataset('json', data_files='data/math_level3to5_data_processed_with_qwen_prompt.json')
        dataset = dataset.map(lambda x: {'problem': x['input'].replace('<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n', '').replace('<|im_end|>\n<|im_start|>assistant', ''), 'solution': x['gt_answer']})
    elif args.train_set == 'big_math':
        dataset = datasets.load_dataset('SynthLabsAI/Big-Math-RL-Verified')
        dataset = dataset.rename_column('answer', 'solution')
    elif args.train_set == 'deepscaler':
        dataset = datasets.load_dataset('agentica-org/DeepScaleR-Preview-Dataset')
        dataset = dataset.remove_columns(['solution'])
        dataset = dataset.rename_column('answer', 'solution')
    elif args.train_set == 'dapo':
        remove_string = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
        remove_string_2 = "\n\nRemember to put your answer on its own line after \"Answer:\"."
        dataset = datasets.load_dataset('YouJiacheng/DAPO-Math-17k-dedup')
        dataset = dataset.map(lambda x: {'problem': x['prompt'][0]['content'].replace(remove_string, '').replace(remove_string_2, '').strip(), 'solution': x['reward_model']['ground_truth']})
    else:
        raise ValueError(f"Invalid train_set: {args.train_set}")

    if not args.test_only:
        train_dataset = dataset['train']
        train_dataset = train_dataset.add_column('data_source', [args.train_set] * len(train_dataset))
        if args.filter_key is not None and args.filter_value is not None:
            train_dataset = train_dataset.filter(lambda x: x[args.filter_key] == args.filter_value)
        train_dataset = train_dataset.remove_columns([k for k in train_dataset.column_names if k not in ['problem', 'solution', 'data_source']])

    test_datasources = glob('data/*.jsonl')
    test_datasets = []
    for test_datasource in test_datasources:
        if 'seed_io' in test_datasource or 'MbppPlus' in test_datasource or 'HumanEvalPlus' in test_datasource:
            continue
        temp_ds = datasets.load_dataset('json', data_files=test_datasource, split='train')
        if 'question' in temp_ds.column_names and 'problem' not in temp_ds.column_names:
            temp_ds = temp_ds.rename_column('question', 'problem')
        temp_ds = temp_ds.remove_columns([col for col in temp_ds.column_names if col not in ['problem', 'answer']])
        temp_ds = temp_ds.add_column('data_source', [test_datasource.split('/')[-1].split('.')[0]] * len(temp_ds))
        temp_ds = temp_ds.cast_column('problem', datasets.Value('string'))
        temp_ds = temp_ds.cast_column('answer', datasets.Value('string'))
        temp_ds = temp_ds.cast_column('data_source', datasets.Value('string'))
        test_datasets.append(temp_ds)
    live_math_bench_datasets = ['v202412_AMC_en', 'v202412_CCEE_en', 'v202412_CNMO_en', 'v202412_WLPMC_en', 'v202412_hard_en']
    for dataset_name in live_math_bench_datasets:
        live_math_bench_ds = datasets.load_dataset('opencompass/LiveMathBench', dataset_name)['test']
        live_math_bench_ds = live_math_bench_ds.rename_column('question', 'problem')
        live_math_bench_ds = live_math_bench_ds.remove_columns([col for col in live_math_bench_ds.column_names if col not in ['problem', 'answer']])
        live_math_bench_ds = live_math_bench_ds.add_column('data_source', ['live_math_bench'] * len(live_math_bench_ds))
        test_datasets.append(live_math_bench_ds)
    test_dataset = datasets.concatenate_datasets(test_datasets)

    if not args.test_only:
        train_dataset = train_dataset.map(
            function=make_map_fn(args.train_split_key, 'problem', 'solution', args.train_set == 'math', args.reward_fn_extraction_type, args.nothink),
            with_indices=True, num_proc=16,
        )
    test_dataset = test_dataset.map(
        function=make_map_fn(args.eval_split_key, 'problem', 'answer', False, args.reward_fn_extraction_type, args.nothink),
        with_indices=True, num_proc=16,
    )

    if args.length_limit != -1 and not args.test_only:
        train_dataset = train_dataset.select(range(args.length_limit))
        test_dataset = test_dataset.select(range(args.length_limit))

    local_dir = args.local_dir + f'/{args.train_set}{"_nothink" if args.nothink else ""}'
    hdfs_dir = args.hdfs_dir

    if args.filter_key is not None:
        filter_key = f"_{args.filter_key}_{args.filter_value}"
    else:
        filter_key = ""

    if not args.test_only:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_{args.reward_fn_extraction_type}{"" if args.length_limit == -1 else f"_{args.length_limit}"}{filter_key}.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'test_{args.reward_fn_extraction_type}{"_ood" if args.ood_testsets else ""}{"" if args.length_limit == -1 else f"_{args.length_limit}"}{filter_key}.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data')
    parser.add_argument(
        '--reward_fn_extraction_type',
        default='answer',
        choices=['answer', 'boxed', 'none']
    )
    parser.add_argument('--length_limit', default=-1, type=int)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_set', default='math', choices=['math', 'orz', 'simplerl', 'big_math', 'deepscaler', 'dapo'])
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--train_split_key', default='train', type=str)
    parser.add_argument('--eval_split_key', default='test', type=str)
    parser.add_argument('--filter_key', default=None, type=str)
    parser.add_argument('--filter_value', default=None, type=str)
    parser.add_argument('--nothink', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    process_data(args)