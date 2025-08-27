"""
Configuration for benchmark evaluation datasets.
"""

import os
from typing import Dict, List, Optional

class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    
    def __init__(self, validation_dir: str = "./validation_datasets"):
        self.validation_dir = validation_dir
        self.available_benchmarks = self._get_available_benchmarks()
    
    def _get_available_benchmarks(self) -> Dict[str, str]:
        """Get available benchmark files."""
        benchmarks = {}
        if os.path.exists(self.validation_dir):
            for file in os.listdir(self.validation_dir):
                if file.endswith('.parquet'):
                    name = file.replace('_test.parquet', '')
                    benchmarks[name] = os.path.join(self.validation_dir, file)
        return benchmarks
    
    def get_benchmark_files(self, benchmarks: Optional[List[str]] = None) -> List[str]:
        """Get benchmark files for evaluation."""
        if benchmarks is None:
            return list(self.available_benchmarks.values())
        
        files = []
        for benchmark in benchmarks:
            if benchmark in self.available_benchmarks:
                files.append(self.available_benchmarks[benchmark])
            else:
                print(f"Warning: Benchmark '{benchmark}' not found. Available: {list(self.available_benchmarks.keys())}")
        
        return files
    
    def get_benchmark_info(self) -> Dict[str, Dict]:
        """Get information about available benchmarks."""
        info = {}
        
        benchmark_details = {
            'math': {
                'description': 'MATH dataset - High school competition mathematics',
                'metric': 'math_accuracy',
                'task_type': 'mathematical_reasoning'
            },
            'gsm8k': {
                'description': 'GSM8K - Grade school math word problems',
                'metric': 'math_accuracy', 
                'task_type': 'mathematical_reasoning'
            },
            'hellaswag': {
                'description': 'HellaSwag - Commonsense reasoning',
                'metric': 'multiple_choice_accuracy',
                'task_type': 'commonsense_reasoning'
            },
            'arc_challenge': {
                'description': 'ARC Challenge - Science questions',
                'metric': 'multiple_choice_accuracy',
                'task_type': 'scientific_reasoning'
            },
            'arc_easy': {
                'description': 'ARC Easy - Science questions (easier)',
                'metric': 'multiple_choice_accuracy',
                'task_type': 'scientific_reasoning'
            },
            'truthfulqa': {
                'description': 'TruthfulQA - Truthfulness evaluation',
                'metric': 'truthfulness_accuracy',
                'task_type': 'truthfulness'
            }
        }
        
        # Add MMLU subjects
        mmlu_subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge'
        ]
        for subject in mmlu_subjects:
            benchmark_details[f'mmlu_{subject}'] = {
                'description': f'MMLU {subject.replace("_", " ").title()}',
                'metric': 'multiple_choice_accuracy',
                'task_type': 'knowledge_reasoning'
            }
        
        for benchmark_name in self.available_benchmarks:
            if benchmark_name in benchmark_details:
                info[benchmark_name] = {
                    'file_path': self.available_benchmarks[benchmark_name],
                    **benchmark_details[benchmark_name]
                }
            else:
                info[benchmark_name] = {
                    'file_path': self.available_benchmarks[benchmark_name],
                    'description': f'Custom benchmark: {benchmark_name}',
                    'metric': 'general_accuracy',
                    'task_type': 'general'
                }
        
        return info
    
    def print_available_benchmarks(self):
        """Print information about available benchmarks."""
        info = self.get_benchmark_info()
        
        print("Available Benchmarks:")
        print("=" * 50)
        
        for name, details in info.items():
            print(f"Name: {name}")
            print(f"Description: {details['description']}")
            print(f"Metric: {details['metric']}")
            print(f"Task Type: {details['task_type']}")
            print(f"File: {details['file_path']}")
            print("-" * 30)


# Example usage and default configuration
DEFAULT_BENCHMARK_CONFIG = {
    'validation_dir': './validation_datasets',
    # 'default_benchmarks': ['math', 'gsm8k', 'truthfulqa', 'arc_challenge','arc_easy', 'mmlu_abstract_algebra', 'mmlu_anatomy', 'mmlu_astronomy', 'mmlu_business_ethics', 'mmlu_clinical_knowledge'],
    'default_benchmarks': ['math', 'mmlu_clinical_knowledge'],
    'evaluation_frequency': 100,  # Evaluate every 100 steps
    'max_samples_per_benchmark': 10,  # Limit samples for faster evaluation
}
