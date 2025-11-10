import json
import re
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter


@dataclass
class PromptTemplate:
    """Container for different prompt types and their improvements"""
    base_template: str
    improvements: List[str]
    last_updated: str
    performance_context: str = ""
    
    def get_template(self) -> str:
        return self.base_template

class PromptManager:
    """Manages dynamic prompt updates based on benchmark analysis"""
    
    def __init__(self, config=None, template_file: Optional[str] = None):
        self.config = config
        
        # New: optional template file path to initialize prompts from JSON
        self.template_file: Optional[str] = template_file
        try:
            if self.template_file is None and config is not None:
                if isinstance(config, dict):
                    self.template_file = config.get('prompt_manager', {}).get('template_file')
                else:
                    pm_cfg = getattr(config, 'prompt_manager', None)
                    if pm_cfg is not None:
                        self.template_file = getattr(pm_cfg, 'template_file', None)
        except Exception:
            self.template_file = template_file
        
        # Initialize templates: prefer JSON file if provided
        try:
            if self.template_file and Path(self.template_file).exists():
                self.templates = self._initialize_from_json(self.template_file)
                print(f"[DEBUG] PromptManager: Initialized templates from JSON: {self.template_file}")
            else:
                # Initialize default templates
                self.templates = self._initialize_from_json("absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json")
                self.template_file = "absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json"
                print(f"[DEBUG] PromptManager: No JSON file found or no template was provided. Using default prompts")
        except Exception as e:
            self.templates = self._initialize_from_json("absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json")
            self.template_file = "absolute_zero_reasoner/data_construction/initial_prompt_templates/default.json"
            print(f"[DEBUG] PromptManager: Failed to init from JSON due to {e}, falling back to defaults")
        
        print(f"[DEBUG] PromptManager initialized with templates: {list(self.templates.keys())}")
        if self.template_file:
            print(f"[DEBUG] PromptManager.template_file = {self.template_file}")
    
    def get_template(self, prompt_type: str, question: str = None) -> str:
        """Get the current template for a specific prompt type"""
        if prompt_type not in self.templates:
            print(f"[DEBUG] PromptManager: Unknown prompt type '{prompt_type}', using base template")
            return question or "{}"
        
        template = self.templates[prompt_type].get_template()
        
        # Format with question if provided
        if question is not None:
            try:
                return template.format(question)
            except (KeyError, ValueError) as e:
                print(f"[DEBUG] PromptManager: Template formatting failed: {e}, falling back to base")
                return f"{template}\n\n{question}"
        
        return template
    
    def get_solver_instruction(self, question: str) -> str:
        """Get solver instruction for a specific question"""
        template = self.get_template('solver')
        
        # Handle both old and new template formats
        if '{}' in template:
            return template.format(question)
        else:
            # If no placeholder, append question at the end
            return f"{template}\n\nUser: {question}\nAssistant: <think>"
    
    def get_judge_instruction(self, prompt_type: str = "answer") -> str:
        """Get judge instruction for evaluation with specific type"""
        # Map the prompt type to the appropriate template
        judge_template_map = {
            "answer": "judge_answer",
            "question": "judge_question"
        }
        
        template_name = judge_template_map.get(prompt_type, "judge_answer")
        
        if template_name not in self.templates:
            print(f"[DEBUG] PromptManager: Judge template '{template_name}' not found, using fallback")
            template_name = "judge"  # Fallback to generic judge
        
        return self.get_template(template_name)
    
    def get_proposer_instruction(self, ref: bool, with_answer_generation: bool=True) -> str:
        """Get proposer instruction for question generation"""
        if ref:
            if with_answer_generation:
                return self.get_template('proposer_with_ref_with_answer_generation')
            else:
                return self.get_template('proposer_with_ref_no_answer_generation')
        else:
            if with_answer_generation:
                return self.get_template('proposer_no_ref_with_answer_generation')
            else:
                return self.get_template('proposer_no_ref_no_answer_generation')
    
    def _initialize_from_json(self, file_path: str) -> Dict[str, PromptTemplate]:
        """Initialize templates from a JSON file.
        JSON schema supported:
        - Flat mapping: { "solver": {"base_template": "...", "performance_context": "..."}, ... }
        - Or with a top-level key "templates": { ...same as above }
        Optional fields per template: improvements (list[str]), performance_context (str)
        last_updated is set to now if not provided.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        payload = data.get('templates', data)
        templates: Dict[str, PromptTemplate] = {}
        now_iso = datetime.now().isoformat()
        
        for name, value in payload.items():
            if isinstance(value, dict) and 'base_template' in value:
                templates[name] = PromptTemplate(
                    base_template=value['base_template'],
                    improvements=value.get('improvements', []),
                    last_updated=value.get('last_updated', now_iso),
                    performance_context=value.get('performance_context', f"Loaded from {Path(file_path).name}")
                )
            elif isinstance(value, str):
                # Simple form: value is the base template string
                templates[name] = PromptTemplate(
                    base_template=value,
                    improvements=[],
                    last_updated=now_iso,
                    performance_context=f"Loaded from {Path(file_path).name}"
                )
        
        # Backward compatibility / aliases
        if 'judge' not in templates and 'judge_answer' in templates:
            templates['judge'] = templates['judge_answer']
        
        return templates