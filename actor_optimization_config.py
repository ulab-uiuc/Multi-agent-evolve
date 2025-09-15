"""
Configuration for actor-driven prompt optimization system
"""

# Enable actor prompt optimization in trainer config
ACTOR_OPTIMIZATION_CONFIG = {
    'enable_actor_prompt_optimization': True,  # Enable actor-driven optimization
    'optimization_frequency': 2,  # Run optimization every N validation steps
    'max_prompt_improvements': 5,  # Maximum improvements per prompt type
    'optimization_temperature': 0.7,  # Temperature for actor generation
    'optimization_max_tokens': 1024,  # Max tokens for optimization prompt
}

# Protected regions configuration - these regions CANNOT be modified by actor
PROTECTED_REGIONS_CONFIG = {
    'solver': {
        'chat_format': {
            'pattern': r'A conversation between User and Assistant\.',
            'description': 'Core conversation structure'
        },
        'xml_tags': {
            'pattern': r'<think>.*?</think>.*?<answer>.*?</answer>',
            'description': 'Required XML tags for reasoning and answer'
        },
        'user_placeholder': {
            'pattern': r'User:\s*\{.*?\}',
            'description': 'User question placeholder'
        },
        'assistant_start': {
            'pattern': r'Assistant:\s*<think>',
            'description': 'Assistant response start marker'
        }
    },
    'judge': {
        'core_instruction': {
            'pattern': r'Evaluate the following answer',
            'description': 'Core evaluation instruction'
        }
    },
    'proposer': {
        'generation_instruction': {
            'pattern': r'Generate a new question',
            'description': 'Core generation instruction'
        }
    }
}

# Example trainer configuration
EXAMPLE_TRAINER_CONFIG = {
    "azr": {
        # ... other azr configs ...
        "enable_actor_prompt_optimization": True,
        "task_type": "general",  # Enable general task optimization
        "prompt_optimization": {
            "frequency": 2,  # Optimize every 2 validation steps
            "min_accuracy_threshold": 0.3,  # Only optimize if accuracy < 30%
            "max_improvements_per_step": 3,  # Limit improvements per step
            "enable_safety_validation": True,  # Enable safety checks
            "backup_prompts": True,  # Keep backup of original prompts
        }
    },
    "trainer": {
        "default_local_dir": "./outputs",
        # ... other trainer configs ...
    }
}

# Safety configuration
SAFETY_CONFIG = {
    'min_prompt_length': 20,  # Minimum prompt length in characters
    'max_prompt_length': 4000,  # Maximum prompt length in characters
    'dangerous_patterns': [
        r'ignore\s+previous\s+instructions',
        r'forget\s+everything',
        r'system\s+prompt',
        r'jailbreak',
        r'hack',
        r'bypass',
    ],
    'required_elements': {
        'solver': ['User:', 'Assistant:', '<think>', '<answer>'],
        'judge': ['Evaluate'],
        'proposer': ['Generate', 'question']
    }
}
