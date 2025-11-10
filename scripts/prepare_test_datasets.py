#!/usr/bin/env python3
"""
Script to prepare test datasets for evaluation in general tasks.
Supports MATH, HellaSwag, GSM8K, and other common benchmarks.
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Any
import argparse
from pathlib import Path


def load_math_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MATH dataset."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['problem']}"}],
            "ground_truth": item['solution'],
            "answer": item['answer'].split('The answer is')[-1].strip() if 'The answer is' in item['answer'] else item['answer'],
            "data_source": "math",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_gsm8k_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Extract the final answer
        answer = item['answer'].split('####')[-1].strip() if '####' in item['answer'] else item['answer']
        
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['question']}"}],
            "ground_truth": item['answer'],
            "answer": answer,
            "data_source": "gsm8k",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_hellaswag_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load HellaSwag dataset."""
    dataset = load_dataset("Rowan/hellaswag", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = item['endings']
        choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
        prompt_text = f"""Complete the following scenario by choosing the most likely continuation:

Context: {item['ctx']}

Choices:
{choice_text}

Choose the most appropriate continuation (A, B, C, or D):"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": chr(65 + int(item['label'])),  # Convert 0,1,2,3 to A,B,C,D
            "answer": chr(65 + int(item['label'])),
            "data_source": "hellaswag",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data


def load_mmlu_dataset(subject: str = "abstract_algebra", split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MMLU dataset for a specific subject."""
    dataset = load_dataset("cais/mmlu", subject, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = item['choices']
        choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
        
        prompt_text = f"""Answer the following multiple choice question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": chr(65 + item['answer']),  # Convert 0,1,2,3 to A,B,C,D
            "answer": chr(65 + item['answer']),
            "data_source": f"mmlu_{subject}",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
        # print(f"Loaded {subject} sample {i+1}: {item['question']}")
        # print(f"Choices: {choice_text}")
        # print(f"Ground truth: {data[-1]['ground_truth']}")
        # print(f"Answer: {data[-1]['answer']}")
        # print("-" * 40)
    return data


def load_arc_dataset(split: str = "test", challenge: bool = True, num_samples: int = None) -> List[Dict]:
    """Load ARC dataset."""
    config = "ARC-Challenge" if challenge else "ARC-Easy"
    dataset = load_dataset("allenai/ai2_arc", config, split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        # Format choices
        choices = [choice for choice in item['choices']['text']]
        labels = item['choices']['label']
        choice_text = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
        
        prompt_text = f"""Answer the following science question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer:"""
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": item['answerKey'],
            "answer": item['answerKey'],
            "data_source": f"arc_{'challenge' if challenge else 'easy'}",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })
    
    return data

def load_aime24_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load AIME 2024 dataset."""
    dataset = load_dataset("HuggingFaceH4/aime_2024", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{item['problem']}"}],
            "ground_truth": item['answer'],
            "answer": item['answer'],
            "data_source": "aime24",
            "extra_info": {"metric": "math_accuracy"}
        })
    
    return data


def load_truthfulqa_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load TruthfulQA dataset."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
            
        data.append({
            "prompt": [{"role": "user", "content": f"Answer the following question truthfully:\n\n{item['question']}"}],
            "ground_truth": item['best_answer'],
            "answer": item['best_answer'],
            "data_source": "truthfulqa",
            "extra_info": {"metric": "truthfulness_accuracy"}
        })
    
    return data

def load_gpqa_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load GPQA dataset."""
    dataset = load_dataset("Idavidrein/gpqa", 'gpqa_main', split=split)
    
    data = []
    for i, item in enumerate(dataset):
        # if num_samples and i >= num_samples:
        #     break
            
        # Format the question and choices
        question = item['Question']
        correct_answer = item['Correct Answer']
        incorrect_answers = [
            item['Incorrect Answer 1'],
            item['Incorrect Answer 2'], 
            item['Incorrect Answer 3']
        ]
        
        # Combine all answers and shuffle them
        all_answers = [correct_answer] + incorrect_answers
        import random
        random.shuffle(all_answers)
        
        # Find which position the correct answer is in
        correct_index = all_answers.index(correct_answer)
        correct_letter = chr(65 + correct_index)  # Convert to A, B, C, D
        
        # Format choices
        choice_text = "\n".join([f"{chr(65+j)}. {answer}" for j, answer in enumerate(all_answers)])
        
        prompt_text = f"""Answer the following scientific question:

Question: {question}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
        
        # Get additional metadata
        subdomain = item.get('Subdomain', '')
        high_level_domain = item.get('High-level domain', '')
        difficulty = item.get("Writer's Difficulty Estimate", '')
        
        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": correct_answer,
            "answer": correct_letter,
            "data_source": "gpqa",
            "extra_info": {
                "metric": "multiple_choice_accuracy",
            }
        })
        
        print(f"Loaded GPQA sample {i+1}: {question[:100]}...")
        print(f"Answer: {data[-1]['answer']}")
    
    return data

def load_mmlu_dataset_as_a_whole(split: str = "test", num_samples: int = None) -> List[Dict]:
    # enumerate all subjects in MMLU
    MMLU_SUBJECTS = [
        "abstract_algebra",
        "anatomy", 
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry", 
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics", 
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law", 
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]
    
    data = []
    total_collected = 0
    
    for subject in MMLU_SUBJECTS:
        if num_samples and total_collected >= num_samples:
            break
            
        try:
            dataset = load_dataset("cais/mmlu", subject, split=split)
            
            subject_samples = 0
            for i, item in enumerate(dataset):
                if num_samples and total_collected >= num_samples:
                    break
                    
                choices = item['choices']
                choice_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
                
                prompt_text = f"""Answer the following multiple choice question:

Question: {item['question']}

Choices:
{choice_text}

Choose the correct answer (A, B, C, or D):"""
                
                data.append({
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "ground_truth": chr(65 + item['answer']),  # Convert 0,1,2,3 to A,B,C,D
                    "answer": chr(65 + item['answer']),
                    "data_source": f"mmlu",
                    "extra_info": {
                        "metric": "multiple_choice_accuracy",
                    }
                })
                
                total_collected += 1
                subject_samples += 1
                
            print(f"Loaded {subject_samples} samples from MMLU subject: {subject}")
            
        except Exception as e:
            print(f"Failed to load MMLU subject {subject}: {e}")
            continue
    
    print(f"Total MMLU samples loaded: {len(data)}")
    return data


def load_winogrande_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load WinoGrande dataset from coref-data/winogrande_raw."""
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format the sentence with the blank
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        correct_answer = item['answer']  # This is "1" or "2"

        # Create prompt with two options
        prompt_text = f"""Complete the following sentence by choosing the most appropriate option:

Sentence: {sentence}

Options:
A. {option1}
B. {option2}

Choose the correct option (A or B):"""

        # Convert answer to letter format
        answer_letter = "A" if correct_answer == "1" else "B"

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer_letter,
            "answer": answer_letter,
            "data_source": "winogrande",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })

    return data


def load_ifeval_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load IFEval dataset for instruction following evaluation."""
    dataset = load_dataset("google/IFEval", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the prompt and instruction details
        prompt_text = item['prompt']
        instruction_id_list = item['instruction_id_list']
        kwargs = item['kwargs'] if 'kwargs' in item else {}

        # Create a formatted instruction following prompt
        formatted_prompt = f"""Follow the instructions below carefully:

{prompt_text}

Instructions to follow:
{', '.join(instruction_id_list)}"""

        data.append({
            "prompt": [{"role": "user", "content": formatted_prompt}],
            "ground_truth": "",  # IFEval doesn't have ground truth answers, it's evaluated by instruction compliance
            "answer": "",  # Will be filled by the model's response
            "data_source": "ifeval",
            "extra_info": {
                "metric": "instruction_following",
                "instruction_id_list": instruction_id_list,
                "kwargs": kwargs,
                "key": item.get('key', i)
            }
        })

    return data


def load_minerva_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load Minerva math dataset for mathematical reasoning evaluation."""
    dataset = load_dataset("math-ai/minervamath", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the question and answer
        question = item['question']
        answer = item['answer']

        # Create a formatted mathematical reasoning prompt
        prompt_text = f"""Solve the following mathematical problem step by step:

{question}

Provide your final answer as a numerical value."""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "minerva",
            "extra_info": {"metric": "math_accuracy"}
        })

    return data


def load_amc_dataset(split: str = "train", num_samples: int = None) -> List[Dict]:
    """Load AMC dataset from AI-MO/aimo-validation-amc for American Mathematics Competitions evaluation."""
    dataset = load_dataset("AI-MO/aimo-validation-amc", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the problem and answer
        problem = item['problem']
        answer = str(item['answer'])  # Convert to string for consistency
        problem_url = item.get('url', '')

        # Create a formatted AMC problem prompt
        prompt_text = f"""Solve the following AMC (American Mathematics Competition) problem step by step:

{problem}

Provide your final answer as a numerical value."""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "amc",
            "extra_info": {
                "metric": "math_accuracy",
                "problem_id": item.get('id', i),
                "source_url": problem_url
            }
        })

    return data


def load_olympiad_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load Olympiad dataset from KbsdJames/Omni-MATH for mathematical olympiad evaluation."""
    dataset = load_dataset("KbsdJames/Omni-MATH", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the problem details
        problem = item['problem']
        solution = item.get('solution', '')
        answer = item['answer']
        domain = item.get('domain', '')
        difficulty = item.get('difficulty', 0)
        source = item.get('source', '')

        # Create a formatted Olympiad problem prompt
        prompt_text = f"""Solve the following Mathematical Olympiad problem step by step:

{problem}

Provide your final answer clearly."""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "olympiad",
            "extra_info": {
                "metric": "math_accuracy",
                "domain": domain,
                "difficulty": difficulty,
                "source": source,
                "solution": solution
            }
        })

    return data


def load_livebench_reasoning_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load LiveBench reasoning dataset for logical reasoning evaluation."""
    dataset = load_dataset("livebench/reasoning", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the reasoning problem details
        question_id = item.get('question_id', f"question_{i}")
        category = item.get('category', '')
        ground_truth = item['ground_truth']
        turns = item.get('turns', [])
        task = item.get('task', '')
        level = item.get('level', 0)

        # Extract the problem from turns (usually the first turn contains the problem)
        if turns and len(turns) > 0:
            problem_text = turns[0]
        else:
            problem_text = "No problem text available"

        # Create a formatted reasoning problem prompt
        prompt_text = f"""Solve the following logical reasoning problem step by step:

{problem_text}

Think through the constraints systematically and provide your final answer."""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": ground_truth,
            "answer": ground_truth,
            "data_source": "livebench_reasoning",
            "extra_info": {
                "metric": "reasoning_accuracy",
                "question_id": question_id,
                "category": category,
                "task": task,
                "level": level
            }
        })

    return data


def load_mmlu_pro_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load MMLU-Pro dataset for enhanced multi-task language understanding evaluation."""
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract the problem details
        question_id = item.get('question_id', i)
        question = item['question']
        options = item['options']
        answer = item['answer']
        answer_index = item.get('answer_index', 0)
        category = item.get('category', '')
        source = item.get('src', '')
        cot_content = item.get('cot_content', '')

        # Format choices (MMLU-Pro typically has 10 options, labeled A-J)
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        choice_text = "\n".join([f"{choice_labels[j]}. {option}" for j, option in enumerate(options[:len(choice_labels)])])

        # Determine the correct answer letter
        if answer_index < len(choice_labels):
            correct_letter = choice_labels[answer_index]
        else:
            # Fallback: find the answer in options
            try:
                correct_index = options.index(answer)
                correct_letter = choice_labels[correct_index] if correct_index < len(choice_labels) else 'A'
            except ValueError:
                correct_letter = 'A'  # Default fallback

        # Create a formatted MMLU-Pro problem prompt
        prompt_text = f"""Answer the following multiple choice question from the {category} domain. Think step by step and use chain-of-thought reasoning:

Question: {question}

Choices:
{choice_text}

Choose the correct answer (A-J):"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": correct_letter,
            "answer": correct_letter,
            "data_source": "mmlu_pro",
            "extra_info": {
                "metric": "multiple_choice_accuracy",
                "question_id": question_id,
                "category": category,
                "source": source,
                "cot_content": cot_content,
                "raw_answer": answer
            }
        })

    return data


def load_commonsenseqa_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load CommonsenseQA dataset for commonsense reasoning evaluation."""
    dataset = load_dataset("tau/commonsense_qa", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract question and answer choices
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_key = item['answerKey']

        # Format choices
        choice_text = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])

        # Create a formatted commonsense reasoning prompt
        prompt_text = f"""Answer the following commonsense reasoning question:

Question: {question}

Choices:
{choice_text}

Choose the correct answer:"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer_key,
            "answer": answer_key,
            "data_source": "commonsenseqa",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })

    return data


def load_openbookqa_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load OpenBookQA dataset for open book question answering."""
    dataset = load_dataset("allenai/openbookqa", "main", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract question and choices
        question_stem = item['question_stem']
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_key = item['answerKey']

        # Format choices
        choice_text = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])

        # Create a formatted OpenBookQA prompt
        prompt_text = f"""Answer the following science question that requires reasoning with basic facts:

Question: {question_stem}

Choices:
{choice_text}

Choose the correct answer:"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer_key,
            "answer": answer_key,
            "data_source": "openbookqa",
            "extra_info": {"metric": "multiple_choice_accuracy"}
        })

    return data


def load_naturalquestions_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load Natural Questions dataset for open domain question answering."""
    # Use the NQ-Open dataset which is a simplified version
    dataset = load_dataset("nq_open", split=split if split != "test" else "validation")

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract question and answer
        question = item['question']
        # NQ-Open has a list of acceptable answers
        answers = item['answer'] if isinstance(item['answer'], list) else [item['answer']]
        answer = answers[0] if answers else ""

        if not answer:
            continue

        prompt_text = f"""Answer the following question concisely:

Question: {question}

Provide a brief, factual answer:"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "naturalquestions",
            "extra_info": {
                "metric": "exact_match",
                "all_answers": answers  # Store all acceptable answers
            }
        })

    return data


def load_triviaqa_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load TriviaQA dataset for trivia question answering."""
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract question and answer
        question = item['question']
        # TriviaQA has multiple possible answers, we'll use the first one
        answer = item['answer']['value'] if item['answer'] else ""

        prompt_text = f"""Answer the following trivia question:

Question: {question}

Provide a brief, factual answer:"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "triviaqa",
            "extra_info": {
                "metric": "exact_match",
                "aliases": item['answer']['aliases'] if item['answer'] else []
            }
        })

    return data


def load_squad_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load SQuAD v2 dataset for reading comprehension."""
    dataset = load_dataset("squad_v2", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract context, question and answer
        context = item['context']
        question = item['question']

        # Check if the question is answerable
        # SQuAD v2 has both answerable and unanswerable questions
        if item['answers'] and item['answers'].get('text'):
            answers = item['answers']['text']
            answer = answers[0] if answers else ""
        else:
            # Skip unanswerable questions for simplicity
            continue

        if not answer:
            continue

        prompt_text = f"""Read the following passage and answer the question based on it.

Context: {context}

Question: {question}

Answer:"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "squad",
            "extra_info": {
                "metric": "exact_match",
                "all_answers": answers if 'answers' in locals() else [answer]
            }
        })

    return data


def load_boolq_dataset(split: str = "validation", num_samples: int = None) -> List[Dict]:
    """Load BoolQ dataset for yes/no question answering."""
    dataset = load_dataset("google/boolq", split=split)

    data = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Extract passage, question and answer
        passage = item['passage']
        question = item['question']
        answer = "yes" if item['answer'] else "no"

        prompt_text = f"""Read the following passage and answer the yes/no question.

Passage: {passage}

Question: {question}

Answer with only 'yes' or 'no':"""

        data.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "ground_truth": answer,
            "answer": answer,
            "data_source": "boolq",
            "extra_info": {"metric": "exact_match"}
        })

    return data


def load_bbh_dataset(split: str = "test", num_samples: int = None) -> List[Dict]:
    """Load BBH dataset."""
    BBH_SUBSETS = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_three_objects",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "movie_recommendation",
        'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
    ]
    data = []
    for subset in BBH_SUBSETS:
        # dataset = load_dataset("SaylorTwift/bbh", split=split)
        dataset = load_dataset("SaylorTwift/bbh", subset, split=split)
        for i, item in enumerate(dataset):
            # if num_samples and i >= num_samples:
            #     break
            question = item['input']
            correct_answer = item['target']

            data.append({
                "prompt": [{"role": "user", "content": question}],
                "ground_truth": correct_answer,
                "answer": correct_answer,
                "data_source": "bbh",
                "extra_info": {
                    "metric": "multiple_choice_accuracy",
                }
            })

            print(f"Loaded BBH sample {i+1}: {question[:100]}...")
            # print(f"Loaded BBH sample {i+1}: {question[:100]}...")
            # print(f"Choices: {choice_text}")
            print(f"Ground truth: {data[-1]['ground_truth']}")
            print(f"Answer: {data[-1]['answer']}")
            print("-" * 40)

    return data

def save_dataset_to_parquet(data: List[Dict], output_path: str, dataset_name: str):
    """Save dataset to parquet format."""
    os.makedirs(output_path, exist_ok=True)
    df = pd.DataFrame(data)
    output_file = os.path.join(output_path, f"{dataset_name}_test.parquet")
    df.to_parquet(output_file)
    print(f"Saved {len(data)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare test datasets for evaluation")
    parser.add_argument("--output_dir", type=str, default="./validation_datasets", 
                       help="Output directory for validation datasets")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to load per dataset (None for all)")
    parser.add_argument("--datasets", nargs="+",
                       choices=["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa", "aime24", "gpqa", "winogrande", "ifeval", "minerva", "amc", "olympiad", "livebench_reasoning", "mmlu_pro", "bbh", "commonsenseqa", "openbookqa", "naturalquestions", "triviaqa", "squad", "boolq", "all"],
                       default=["all"], help="Datasets to prepare")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets_to_load = args.datasets
    if "all" in datasets_to_load:
        datasets_to_load = ["math", "gsm8k", "hellaswag", "mmlu", "arc", "truthfulqa", "aime24", "gpqa", "winogrande", "ifeval", "minerva", "amc", "olympiad", "livebench_reasoning", "mmlu_pro", "bbh", "commonsenseqa", "openbookqa", "naturalquestions", "triviaqa", "squad", "boolq"]
    
    print(f"Preparing datasets: {datasets_to_load}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples per dataset: {args.num_samples or 'All'}")
    
    # Load and save datasets
    if "math" in datasets_to_load:
        print("\nLoading MATH dataset...")
        math_data = load_math_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(math_data, args.output_dir, "math")
    
    if "gsm8k" in datasets_to_load:
        print("\nLoading GSM8K dataset...")
        gsm8k_data = load_gsm8k_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(gsm8k_data, args.output_dir, "gsm8k")
    
    if "hellaswag" in datasets_to_load:
        print("\nLoading HellaSwag dataset...")
        hellaswag_data = load_hellaswag_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(hellaswag_data, args.output_dir, "hellaswag")
    
    if "mmlu" in datasets_to_load:
        print("\nLoading MMLU dataset (all subjects)...")
        
        # First get the MMLU_SUBJECTS list from the function
        # def get_mmlu_subjects():
        #     return [
        #         "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        #         "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        #         "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        #         "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        #         "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        #         "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        #         "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        #         "high_school_physics", "high_school_psychology", "high_school_statistics",
        #         "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        #         "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        #         "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        #         "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        #         "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        #         "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
        #     ]
        
        # subjects = get_mmlu_subjects()
        # for subject in subjects:
        #     mmlu_data = load_mmlu_dataset(subject=subject, num_samples=args.num_samples)
        #     save_dataset_to_parquet(mmlu_data, args.output_dir, f"mmlu_{subject}")
        mmlu_data = load_mmlu_dataset_as_a_whole(num_samples=args.num_samples)
        save_dataset_to_parquet(mmlu_data, args.output_dir, f"mmlu")
    
    if "arc" in datasets_to_load:
        print("\nLoading ARC dataset...")
        arc_challenge_data = load_arc_dataset(challenge=True, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_challenge_data, args.output_dir, "arc_challenge")
        
        arc_easy_data = load_arc_dataset(challenge=False, num_samples=args.num_samples)
        save_dataset_to_parquet(arc_easy_data, args.output_dir, "arc_easy")
    
    if "truthfulqa" in datasets_to_load:
        print("\nLoading TruthfulQA dataset...")
        truthfulqa_data = load_truthfulqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(truthfulqa_data, args.output_dir, "truthfulqa")

    if "aime24" in datasets_to_load:
        print("\nLoading AIME 2024 dataset...")
        aime24_data = load_aime24_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(aime24_data, args.output_dir, "aime24")

    if "gpqa" in datasets_to_load:
        print("\nLoading GPQA dataset...")
        gpqa_data = load_gpqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(gpqa_data, args.output_dir, "gpqa")

    if "winogrande" in datasets_to_load:
        print("\nLoading WinoGrande dataset...")
        winogrande_data = load_winogrande_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(winogrande_data, args.output_dir, "winogrande")

    if "ifeval" in datasets_to_load:
        print("\nLoading IFEval dataset...")
        ifeval_data = load_ifeval_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(ifeval_data, args.output_dir, "ifeval")

    if "minerva" in datasets_to_load:
        print("\nLoading Minerva dataset...")
        minerva_data = load_minerva_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(minerva_data, args.output_dir, "minerva")

    if "amc" in datasets_to_load:
        print("\nLoading AMC dataset...")
        amc_data = load_amc_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(amc_data, args.output_dir, "amc")

    if "olympiad" in datasets_to_load:
        print("\nLoading Olympiad dataset...")
        olympiad_data = load_olympiad_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(olympiad_data, args.output_dir, "olympiad")

    if "livebench_reasoning" in datasets_to_load:
        print("\nLoading LiveBench reasoning dataset...")
        livebench_data = load_livebench_reasoning_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(livebench_data, args.output_dir, "livebench_reasoning")

    if "mmlu_pro" in datasets_to_load:
        print("\nLoading MMLU-Pro dataset...")
        mmlu_pro_data = load_mmlu_pro_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(mmlu_pro_data, args.output_dir, "mmlu_pro")

    if "bbh" in datasets_to_load:
        print("\nLoading BBH dataset...")
        bbh_data = load_bbh_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(bbh_data, args.output_dir, "bbh")

    if "commonsenseqa" in datasets_to_load:
        print("\nLoading CommonsenseQA dataset...")
        commonsenseqa_data = load_commonsenseqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(commonsenseqa_data, args.output_dir, "commonsenseqa")

    if "openbookqa" in datasets_to_load:
        print("\nLoading OpenBookQA dataset...")
        openbookqa_data = load_openbookqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(openbookqa_data, args.output_dir, "openbookqa")

    if "naturalquestions" in datasets_to_load:
        print("\nLoading Natural Questions dataset...")
        naturalquestions_data = load_naturalquestions_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(naturalquestions_data, args.output_dir, "naturalquestions")

    if "triviaqa" in datasets_to_load:
        print("\nLoading TriviaQA dataset...")
        triviaqa_data = load_triviaqa_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(triviaqa_data, args.output_dir, "triviaqa")

    if "squad" in datasets_to_load:
        print("\nLoading SQuAD dataset...")
        squad_data = load_squad_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(squad_data, args.output_dir, "squad")

    if "boolq" in datasets_to_load:
        print("\nLoading BoolQ dataset...")
        boolq_data = load_boolq_dataset(num_samples=args.num_samples)
        save_dataset_to_parquet(boolq_data, args.output_dir, "boolq")

    print(f"\nAll datasets prepared and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
