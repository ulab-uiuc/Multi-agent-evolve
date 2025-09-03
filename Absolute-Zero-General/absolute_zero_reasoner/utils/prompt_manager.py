"""
Simple prompt management system for tracking prompt changes.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter


class PromptManager:
    """Simple prompt manager with basic change tracking"""
    
    def __init__(self, config=None, output_dir: str = "./prompt_history"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple change log - just strings with timestamps
        self.change_log = []
        
        # Initialize default templates
        self.templates = self._initialize_default_templates()
        
        self._log_change("PromptManager initialized")
        
        print(f"[DEBUG] PromptManager initialized with templates: {list(self.templates.keys())}")
    
    def _log_change(self, message: str):
        """Simple logging of changes"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.change_log.append(log_entry)
        print(f"[PROMPT_LOG] {log_entry}")
    
    def _initialize_default_templates(self) -> Dict[str, Dict]:
        """Initialize default prompt templates"""
        templates = {}
        
        # Import the original prompts
        try:
            from absolute_zero_reasoner.data_construction.prompts import (
                general_prediction_prompt, 
                general_generation_prompt,
                general_generation_based_on_reference_prompt
            )
        except ImportError:
            print("[DEBUG] PromptManager: Could not import original prompts, using fallback templates")
            return self._initialize_fallback_templates()
        
        # Store templates with their improvements
        templates['solver'] = {
            'base_template': general_prediction_prompt,
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['proposer'] = {
            'base_template': general_generation_prompt,
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # Judge prompt templates
        templates['judge_answer'] = {
            'base_template': self._get_judge_template("answer"),
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge_question'] = {
            'base_template': self._get_judge_template("question"),
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge_together'] = {
            'base_template': self._get_judge_template("together"),
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        # Backward compatibility
        templates['judge'] = templates['judge_answer']
        
    
    def _get_enhanced_template(self, template_dict: Dict) -> str:
        """Get template with applied improvements"""
        if not template_dict['improvements']:
            return template_dict['base_template']
        
        # Combine base template with improvements
        improvement_text = "\n".join([f"- {imp}" for imp in template_dict['improvements']])
        enhanced_template = f"{template_dict['base_template']}\n\nAdditional Instructions:\n{improvement_text}"
        return enhanced_template
    
    def _initialize_fallback_templates(self) -> Dict[str, Dict]:
        """Initialize fallback templates if original imports fail"""
        templates = {}
        
        templates['solver'] = {
            'base_template': "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>",
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['proposer'] = {
            'base_template': "Generate a new question based on the following examples. The question should be challenging but solvable, and follow similar patterns to the reference questions.",
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge_answer'] = {
            'base_template': "Evaluate the following answer to determine if it is correct. Consider mathematical accuracy, logical reasoning, and completeness. Rate from 1-10.",
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge_question'] = {
            'base_template': "Evaluate the following question to determine if it is well-formed and appropriate. Consider clarity, completeness, and relevance. Rate from 1-10.",
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge_together'] = {
            'base_template': "Evaluate the following question and answer pair. Rate both the question quality and answer quality from 1-10.",
            'improvements': [],
            'last_updated': datetime.now().isoformat()
        }
        
        templates['judge'] = templates['judge_answer']
        
        return templates
    
    def _get_judge_template(self, prompt_type: str) -> str:
        # These are the original judge prompt templates used in reward_managers.py
        if prompt_type == "answer":
            return """Please evaluate the following solution to a question/problem.

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
            return """Please evaluate the quality of the following question generation.
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
            return """Please evaluate the quality of the following question and answer pair.
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
        else:
            return "Invalid judge prompt type requested"
    
    def update_prompts_from_analysis(self, improvement_prompts: Dict[str, str], 
                                   performance_context: str = "", step: int = 0):
        """Update prompts based on benchmark analysis"""
        self._log_change(f"Starting prompt update for step {step}")
        
        for prompt_type, improvement_text in improvement_prompts.items():
            if prompt_type == "judge":
                # Apply improvements to all judge types
                judge_types = ["judge_answer", "judge_question", "judge_together"]
                for judge_type in judge_types:
                    if judge_type in self.templates:
                        improvements = self._extract_improvements(improvement_text)
                        if improvements:
                            self.templates[judge_type]['improvements'].extend(improvements)
                            self.templates[judge_type]['last_updated'] = datetime.now().isoformat()
                            self._log_change(f"Updated {judge_type} with {len(improvements)} improvements")
            elif prompt_type in self.templates:
                improvements = self._extract_improvements(improvement_text)
                if improvements:
                    self.templates[prompt_type]['improvements'].extend(improvements)
                    self.templates[prompt_type]['last_updated'] = datetime.now().isoformat()
                    self._log_change(f"Updated {prompt_type} with {len(improvements)} improvements")
                else:
                    self._log_change(f"No improvements found for {prompt_type}")
            else:
                self._log_change(f"Unknown prompt type: {prompt_type}")
        
        # Save change log
        self._save_change_log()
    
    def _extract_improvements(self, improvement_text: str) -> List[str]:
        """Extract actionable improvements from analysis text"""
        improvements = []
        
        # Look for improvement suggestions
        patterns = [
            r"- \*\*.*?\*\*:\s*(.+)",  # - **Category**: suggestion
            r"- (.+)",  # - suggestion
            r"Consider (.+)",  # Consider doing X
            r"Add (.+)",  # Add specific instructions
            r"Improve (.+)",  # Improve X
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, improvement_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                cleaned = match.strip().rstrip('.')
                if len(cleaned) > 20:
                    improvements.append(cleaned)
        
        # Remove duplicates
        unique_improvements = []
        for imp in improvements:
            if imp not in unique_improvements:
                unique_improvements.append(imp)
        
        return unique_improvements[:3]  # Limit to top 3 improvements
    
    def get_template(self, prompt_type: str) -> str:
        """Get the current template for a specific prompt type"""
        if prompt_type not in self.templates:
            self._log_change(f"Unknown prompt type: {prompt_type}")
            return "{}"
        
        return self._get_enhanced_template(self.templates[prompt_type])
    
    def get_solver_instruction(self, question: str) -> str:
        """Get solver instruction for a specific question"""
        template = self.get_template('solver')
        
        if '{}' in template:
            return template.format(question)
        else:
            return f"{template}\n\nUser: {question}\nAssistant: <think>"
    
    def get_judge_instruction(self, prompt_type: str = "answer") -> str:
        """Get judge instruction for evaluation with specific type"""
        judge_template_map = {
            "answer": "judge_answer",
            "question": "judge_question", 
            "together": "judge_together"
        }
        
        template_name = judge_template_map.get(prompt_type, "judge_answer")
        return self.get_template(template_name)
    
    def get_proposer_instruction(self) -> str:
        """Get proposer instruction for question generation"""
        return self.get_template('proposer')
    
    def _save_change_log(self):
        """Save change log to file"""
        try:
            log_file = self.output_dir / "prompt_changes.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("Prompt Change Log\n")
                f.write("=" * 50 + "\n\n")
                for entry in self.change_log:
                    f.write(entry + "\n")
            print(f"[DEBUG] PromptManager: Saved change log to {log_file}")
        except Exception as e:
            print(f"[DEBUG] PromptManager: Error saving change log: {e}")
    
    def view_change_log(self, recent_n: int = 10):
        """Print recent changes"""
        print(f"\n=== Recent Prompt Changes (Last {recent_n}) ===")
        for entry in self.change_log[-recent_n:]:
            print(entry)
        print("=" * 50)
    
    def reset_template(self, prompt_type: str):
        """Reset a template to its default state"""
        if prompt_type in self.templates:
            old_count = len(self.templates[prompt_type]['improvements'])
            self.templates[prompt_type]['improvements'].clear()
            self.templates[prompt_type]['last_updated'] = datetime.now().isoformat()
            self._log_change(f"Reset {prompt_type} template (removed {old_count} improvements)")
        else:
            self._log_change(f"Cannot reset unknown template: {prompt_type}")
    
    def get_prompt_status(self) -> Dict[str, Dict]:
        """Get status of all prompt templates"""
        status = {}
        for name, template in self.templates.items():
            if name != 'judge':  # Skip alias
                status[name] = {
                    'improvements_count': len(template['improvements']),
                    'last_updated': template['last_updated'],
                    'current_improvements': template['improvements']
                }
        return status
