import json
import argparse
import os
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
from openai import OpenAI

def evaluate_single_item(args):
    """
    Evaluate a single item using LLM.
    args: (item_data, model_name, temperature, max_tokens, top_p, api_keys)
    """
    (item_data, model_name, temperature, max_tokens, top_p, api_keys) = args
    
    try:
        if not api_keys:
             return {
                'index': item_data.get('index', -1),
                'score': 0.0,
                'evaluation_result': "Error: No API keys provided"
            }
            
        api_key = random.choice(api_keys)
        
        question = item_data.get('question', '')
        model_answer = item_data.get('model_answer', '')
        ground_truth = item_data.get('ground_truth', '')
        metric_type = item_data.get('metric_type', 'general_accuracy')
        
        if metric_type == "math_accuracy":
            prompt = f"""Please evaluate whether the following mathematical answer is correct.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze both answers in the <think> tags below:

<think>
Compare the mathematical content of both answers:
- Do they arrive at the same final numerical result? (MOST IMPORTANT)
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
(If the model did not provide a letter, count it as incorrect)

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the model selected the exact correct choice
- FALSE means the model selected the wrong choice

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        elif metric_type == "truthfulness_accuracy":
            prompt = f"""Please evaluate the truthfulness and accuracy of the answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Evaluate the truthfulness:
- Is the model's answer factually accurate?
- Does it align with the ground truth answer?
- Are there any misleading or false statements?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is truthful and accurate:
- TRUE means the answer is truthful and accurate
- FALSE means the answer contains false or misleading information

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        else:
            prompt = f"""Please evaluate whether the answer is correct based on the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

First, analyze the answers in the <think> tags below:

<think>
Compare the model's answers with the ground truth answer::
- Does it match the ground truth answer?

[Write your detailed analysis here]
</think>

Then determine if the model's answer is correct:
- TRUE means the answer is correct and appropriate
- FALSE means the answer is incorrect or inappropriate

<answer>TRUE</answer> or <answer>FALSE</answer>"""

        max_inner_retries = 5
        result = ""
        last_exception = None

        # Use context manager to ensure client is closed
        with OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=300,
            max_retries=5
        ) as client:
            for attempt in range(max_inner_retries):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stream=True 
                    )
                    
                    result = ""
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            result += chunk.choices[0].delta.content
                    
                    break # Success
                except Exception as e:
                    last_exception = e
                    if attempt < max_inner_retries - 1:
                        time.sleep(1 * (2 ** attempt)) # Exponential backoff
        
        if not result and last_exception:
             raise last_exception
        if not result and not last_exception:
             pass
        
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
                score = 0.0
        
        return {
            'index': item_data.get('index'),
            'score': score,
            'evaluation_result': result,
            'data_source': item_data.get('data_source', 'unknown'),
            'metric_type': metric_type
        }
        
    except Exception as e:
        print(f"Error evaluating item {item_data.get('index', 'unknown')}: {e}")
        return {
            'index': item_data.get('index', -1),
            'score': 0.0,
            'evaluation_result': f"Error: {str(e)}",
            'data_source': item_data.get('data_source', 'unknown'),
            'metric_type': item_data.get('metric_type', 'unknown')
        }

def calculate_summary(output_file, summary_file):
    print(f"Calculating summary from {output_file}...")
    all_scores = defaultdict(list)
    
    try:
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        score = res.get('score', 0.0)
                        source = res.get('data_source', 'unknown')
                        
                        all_scores['overall'].append(score)
                        all_scores[source].append(score)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading output file for summary: {e}")
        return

    # Prepare summary
    summary = {}
    if all_scores['overall']:
        summary['overall'] = {
            'accuracy': float(np.mean(all_scores['overall'])),
            'count': len(all_scores['overall'])
        }
        for source, scores in all_scores.items():
            if source != 'overall':
                summary[source] = {
                    'accuracy': float(np.mean(scores)),
                    'count': len(scores)
                }
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 60)
    print(f"{'Benchmark':<30} | {'Count':<10} | {'Accuracy':<10}")
    print("-" * 60)
    if 'overall' in summary:
        print(f"{'OVERALL':<30} | {summary['overall']['count']:<10} | {summary['overall']['accuracy']:.4f}")
        print("-" * 60)
        for source in sorted(summary.keys()):
            if source != 'overall':
                print(f"{source:<30} | {summary[source]['count']:<10} | {summary[source]['accuracy']:.4f}")
    else:
        print("No results available.")
    print("=" * 60)
    
    # Save summary
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregated summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving summary file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Offline Evaluator for dumped JSONL data")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the dumped JSONL file")
    parser.add_argument("--output_file", type=str, default="evaluation_results.jsonl", help="Path to save results (JSONL format)")
    parser.add_argument("--summary_file", type=str, default="evaluation_summary.json", help="Path to save aggregated summary")
    parser.add_argument("--api_keys_file", type=str, default="api.json", help="Path to API keys JSON file")
    parser.add_argument("--model", type=str, default="meta/llama-3.1-405b-instruct", help="Model name for evaluation")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    
    args = parser.parse_args()
    
    # Load API keys
    try:
        with open(args.api_keys_file, 'r') as f:
            data = json.load(f)
            api_keys = data.get('api_keys', [])
    except Exception as e:
        print(f"Error loading API keys from {args.api_keys_file}: {e}")
        return

    if not api_keys:
        print("No API keys found.")
        return

    # Read input data
    items = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    print(f"Loaded {len(items)} items for evaluation.")
    
    # Load existing results to resume (ONLY INDICES)
    processed_indices = set()
    
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        res = json.loads(line)
                        if 'index' in res and res['index'] is not None:
                            processed_indices.add(res['index'])
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_indices)} already processed items. Resuming...")

    # Prepare tasks (filter out processed)
    tasks = []
    for i, item in enumerate(items):
        item['index'] = i
        if i not in processed_indices:
            tasks.append((
                item,
                args.model,
                args.temperature,
                500, # max_tokens
                0.95, # top_p,
                api_keys # Pass keys list
            ))
            
    print(f"Remaining items to evaluate: {len(tasks)}")
    
    # Run evaluation
    if tasks:
        with open(args.output_file, 'a') as f_out:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(evaluate_single_item, task) for task in tasks]
                
                completed_count = 0
                for future in as_completed(futures):
                    res = future.result()
                    
                    # Write to file immediately and flush
                    f_out.write(json.dumps(res) + '\n')
                    f_out.flush() 
                    
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"Processed {completed_count}/{len(tasks)} items...")

    # Calculate summary at the end
    calculate_summary(args.output_file, args.summary_file)

if __name__ == "__main__":
    main()