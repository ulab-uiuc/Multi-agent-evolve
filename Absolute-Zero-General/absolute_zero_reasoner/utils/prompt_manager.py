"""
Dynamic prompt management system that applies improvement suggestions from benchmark tracker.
"""

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
    
    def get_enhanced_template(self) -> str:
        """Get template with applied improvements"""
        if not self.improvements:
            return self.base_template
        
        # Combine base template with improvements
        improvement_text = "\n".join([f"- {imp}" for imp in self.improvements])
        enhanced_template = f"{self.base_template}\n\nAdditional Instructions:\n{improvement_text}"
        return enhanced_template


class PromptManager:
    """Manages dynamic prompt updates based on benchmark analysis"""
    
    def __init__(self, config=None, output_dir: str = "./prompt_history"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default templates
        self.templates = self._initialize_default_templates()
        
        # Load existing prompt history if available
        self._load_prompt_history()
        
        print(f"[DEBUG] PromptManager initialized with templates: {list(self.templates.keys())}")
    
    def _initialize_default_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize default prompt templates from original sources"""
        templates = {}
        
        # Import the original prompts from prompts.py for solver and proposer
        try:
            from absolute_zero_reasoner.data_construction.prompts import (
                general_prediction_prompt, 
                general_generation_prompt,
                general_generation_based_on_reference_prompt
            )
        except ImportError:
            print("[DEBUG] PromptManager: Could not import original prompts, using fallback templates")
            return self._initialize_fallback_templates()
        
        # Solver prompt template (based on general_prediction_prompt)
        templates['solver'] = PromptTemplate(
            base_template=general_prediction_prompt,
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Original template from prompts.py"
        )
        
        # Proposer prompt template (based on generation prompts)
        templates['proposer'] = PromptTemplate(
            base_template=general_generation_prompt,
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Original template from prompts.py"
        )
        
        # Judge prompt templates (from reward_managers.py - different types for scoring)
        templates['judge_answer'] = PromptTemplate(
            base_template=self._get_judge_template("answer"),
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Original judge template from reward_managers.py (answer type)"
        )
        
        templates['judge_question'] = PromptTemplate(
            base_template=self._get_judge_template("question"),
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Original judge template from reward_managers.py (question type)"
        )
        
        templates['judge_together'] = PromptTemplate(
            base_template=self._get_judge_template("together"),
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Original judge template from reward_managers.py (together type)"
        )
        
        # Backward compatibility: generic judge template points to answer type
        templates['judge'] = templates['judge_answer']
        
        print(f"[DEBUG] PromptManager: Initialized templates from original sources")
        return templates
    
    def _get_judge_template(self, prompt_type: str) -> str:
        """Extract judge template from reward managers"""
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
        return templates
    
    def _initialize_fallback_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize fallback templates if original imports fail"""
        templates = {}
        
        # Solver prompt template (for answering questions)
        templates['solver'] = PromptTemplate(
            base_template="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}\nAssistant: <think>",
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Fallback template"
        )
        
        # Proposer prompt template (for question generation)
        templates['proposer'] = PromptTemplate(
            base_template="Generate a new question based on the following examples. The question should be challenging but solvable, and follow similar patterns to the reference questions.",
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Fallback template"
        )
        
        # Judge prompt templates (fallback versions)
        templates['judge_answer'] = PromptTemplate(
            base_template="Evaluate the following answer to determine if it is correct. Consider mathematical accuracy, logical reasoning, and completeness. Rate from 1-10.",
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Fallback template"
        )
        
        templates['judge_question'] = PromptTemplate(
            base_template="Evaluate the following question to determine if it is well-formed and appropriate. Consider clarity, completeness, and relevance. Rate from 1-10.",
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Fallback template"
        )
        
        templates['judge_together'] = PromptTemplate(
            base_template="Evaluate the following question and answer pair. Rate both the question quality and answer quality from 1-10.",
            improvements=[],
            last_updated=datetime.now().isoformat(),
            performance_context="Fallback template"
        )
        
        # Backward compatibility: generic judge template points to answer type
        templates['judge'] = templates['judge_answer']
        
        return templates
    
    def update_prompts_from_analysis(self, improvement_prompts: Dict[str, str], 
                                   performance_context: str = "", step: int = 0):
        """Update prompts based on benchmark analysis"""
        print(f"[DEBUG] PromptManager.update_prompts_from_analysis: Updating prompts for step {step}")
        print(f"[DEBUG] PromptManager.update_prompts_from_analysis: Available improvements: {list(improvement_prompts.keys())}")
        
        for prompt_type, improvement_text in improvement_prompts.items():
            # Handle both specific judge types and generic judge
            if prompt_type == "judge":
                # Apply improvements to all judge types
                judge_types = ["judge_answer", "judge_question", "judge_together", "judge"]
                for judge_type in judge_types:
                    if judge_type in self.templates:
                        improvements = self._extract_improvements(improvement_text, judge_type)
                        if improvements:
                            self.templates[judge_type].improvements.extend(improvements)
                            self.templates[judge_type].last_updated = datetime.now().isoformat()
                            self.templates[judge_type].performance_context = performance_context
                            print(f"[DEBUG] PromptManager: Updated {judge_type} with {len(improvements)} improvements")
                            for i, imp in enumerate(improvements, 1):
                                print(f"[DEBUG] PromptManager:   {i}. {imp}")
            elif prompt_type in self.templates:
                # Handle other prompt types normally
                improvements = self._extract_improvements(improvement_text, prompt_type)
                
                if improvements:
                    # Update the template
                    self.templates[prompt_type].improvements.extend(improvements)
                    self.templates[prompt_type].last_updated = datetime.now().isoformat()
                    self.templates[prompt_type].performance_context = performance_context
                    
                    print(f"[DEBUG] PromptManager: Updated {prompt_type} with {len(improvements)} improvements")
                    for i, imp in enumerate(improvements, 1):
                        print(f"[DEBUG] PromptManager:   {i}. {imp}")
                else:
                    print(f"[DEBUG] PromptManager: No actionable improvements found for {prompt_type}")
            else:
                print(f"[DEBUG] PromptManager: Unknown prompt type '{prompt_type}', skipping")
        
        # Save updated prompts
        self._save_prompt_history(step)
    
    def _extract_improvements(self, improvement_text: str, prompt_type: str) -> List[str]:
        """Extract actionable improvements from analysis text"""
        improvements = []
        
        # Look for specific improvement suggestions
        patterns = [
            r"- \*\*.*?\*\*:\s*(.+)",  # - **Category**: suggestion
            r"- (.+)",  # - suggestion
            r"Consider (.+)",  # Consider doing X
            r"Add (.+)",  # Add specific instructions
            r"Improve (.+)",  # Improve X
            r"Check (.+)",  # Check if X
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, improvement_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Clean and filter the improvement
                cleaned = match.strip().rstrip('.')
                # Use more specific skip conditions to avoid over-filtering
                skip_phrases = [
                    'debug', 'error', 'analyze the', 'step 1', 'step 2', 'step:', 
                    'write your', 'detailed analysis here', 'make sure'
                ]
                has_skip_phrase = any(skip_phrase in cleaned.lower() for skip_phrase in skip_phrases)
                
                if len(cleaned) > 20 and not has_skip_phrase:
                    improvements.append(cleaned)
        
        # Deduplicate while preserving order
        seen = set()
        unique_improvements = []
        for imp in improvements:
            if imp not in seen:
                seen.add(imp)
                unique_improvements.append(imp)
        
        return unique_improvements[:3]  # Limit to top 3 improvements
    
    def get_template(self, prompt_type: str, question: str = None) -> str:
        """Get the current template for a specific prompt type"""
        if prompt_type not in self.templates:
            print(f"[DEBUG] PromptManager: Unknown prompt type '{prompt_type}', using base template")
            return question or "{}"
        
        template = self.templates[prompt_type].get_enhanced_template()
        
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
            "question": "judge_question", 
            "together": "judge_together"
        }
        
        template_name = judge_template_map.get(prompt_type, "judge_answer")
        
        if template_name not in self.templates:
            print(f"[DEBUG] PromptManager: Judge template '{template_name}' not found, using fallback")
            template_name = "judge"  # Fallback to generic judge
        
        return self.get_template(template_name)
    
    def get_proposer_instruction(self) -> str:
        """Get proposer instruction for question generation"""
        return self.get_template('proposer')
    
    def _save_prompt_history(self, step: int):
        """Save prompt history to disk"""
        try:
            history_file = self.output_dir / f"prompt_history_step_{step}.json"
            
            # Convert templates to serializable format
            serializable_templates = {}
            for name, template in self.templates.items():
                serializable_templates[name] = asdict(template)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_templates, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] PromptManager: Saved prompt history to {history_file}")
            
        except Exception as e:
            print(f"[DEBUG] PromptManager: Error saving prompt history: {e}")
    
    def _load_prompt_history(self):
        """Load the most recent prompt history"""
        try:
            # Find the most recent history file
            history_files = list(self.output_dir.glob("prompt_history_step_*.json"))
            if not history_files:
                print("[DEBUG] PromptManager: No existing prompt history found")
                return
            
            # Get the most recent file
            latest_file = max(history_files, key=lambda x: int(x.stem.split('_')[-1]))
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                saved_templates = json.load(f)
            
            # Restore templates
            for name, template_data in saved_templates.items():
                if name in self.templates:
                    self.templates[name] = PromptTemplate(**template_data)
            
            print(f"[DEBUG] PromptManager: Loaded prompt history from {latest_file}")
            
        except Exception as e:
            print(f"[DEBUG] PromptManager: Error loading prompt history: {e}")
    
    def get_prompt_status(self) -> Dict[str, Dict]:
        """Get status of all prompt templates"""
        status = {}
        for name, template in self.templates.items():
            status[name] = {
                'improvements_count': len(template.improvements),
                'last_updated': template.last_updated,
                'performance_context': template.performance_context,
                'current_improvements': template.improvements
            }
        return status
    
    def reset_template(self, prompt_type: str):
        """Reset a template to its default state"""
        if prompt_type in self.templates:
            self.templates[prompt_type].improvements.clear()
            self.templates[prompt_type].last_updated = datetime.now().isoformat()
            self.templates[prompt_type].performance_context = "Reset to default"
            print(f"[DEBUG] PromptManager: Reset {prompt_type} template to default")
