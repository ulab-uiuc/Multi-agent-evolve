"""
Actor-driven prompt optimization system that uses the trained model to improve prompts.
"""

import json
import re
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from absolute_zero_reasoner.utils.prompt_manager import PromptManager, PromptTemplate


@dataclass 
class ProtectedRegion:
    """Defines a protected region in a prompt template"""
    name: str
    pattern: str  # Regex pattern to match the protected region
    description: str
    

class ActorPromptOptimizer:
    """Uses the actor model to optimize prompts based on benchmark analysis"""
    
    def __init__(self, model_interface, prompt_manager: PromptManager, output_dir: str = "./actor_prompt_optimization"):
        self.model_interface = model_interface
        self.prompt_manager = prompt_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define protected regions that should not be modified
        # Based on the original prompts in prompts.py
        self.protected_regions = {
            'solver': [
                ProtectedRegion(
                    name='chat_format',
                    pattern=r'A conversation between User and Assistant',
                    description='Core conversation structure'
                ),
                ProtectedRegion(
                    name='thinking_tags',
                    pattern=r'<think>.*?</think>',
                    description='Required thinking process tags'
                ),
                ProtectedRegion(
                    name='answer_tags',
                    pattern=r'<answer>.*?</answer>',
                    description='Required answer tags'
                ),
                ProtectedRegion(
                    name='user_assistant_pattern',
                    pattern=r'User:.*?Assistant:',
                    description='User-Assistant interaction pattern'
                ),
                ProtectedRegion(
                    name='placeholder_format',
                    pattern=r'\{.*?\}',
                    description='Question placeholder format'
                )
            ],
            'judge': [
                ProtectedRegion(
                    name='task_header',
                    pattern=r'## Task:.*?Given Task',
                    description='Task header structure'
                ),
                ProtectedRegion(
                    name='instructions_section',
                    pattern=r'### Instructions:',
                    description='Instructions section marker'
                ),
                ProtectedRegion(
                    name='output_format',
                    pattern=r'no need to restate or reformat the task',
                    description='Output format instructions'
                )
            ],
            'proposer': [
                ProtectedRegion(
                    name='task_creation_header',
                    pattern=r'## Task: Create a.*?Original Task',
                    description='Task creation header'
                ),
                ProtectedRegion(
                    name='output_format_section',
                    pattern=r'### Output Format:',
                    description='Output format section marker'
                ),
                ProtectedRegion(
                    name='think_question_tags',
                    pattern=r'<think>.*?<question>',
                    description='Think-question output tags'
                ),
                ProtectedRegion(
                    name='task_requirements',
                    pattern=r'### Task Requirements:',
                    description='Task requirements section'
                )
            ]
        }
        
        print(f"[DEBUG] ActorPromptOptimizer initialized with protected regions: {list(self.protected_regions.keys())}")
    
    def optimize_prompts_from_analysis(self, benchmark_analysis: str, problematic_questions: Dict[str, List[str]], 
                                     performance_trends: Dict, step: int) -> Dict[str, str]:
        """Use actor model to optimize prompts based on benchmark analysis"""
        
        print(f"[DEBUG] ActorPromptOptimizer: Starting prompt optimization for step {step}")
        
        optimized_prompts = {}
        
        # Optimize each prompt type
        for prompt_type in ['solver', 'judge', 'proposer']:
            print(f"[DEBUG] ActorPromptOptimizer: Optimizing {prompt_type} prompt")
            
            try:
                current_template = self.prompt_manager.get_template(prompt_type)
                optimization_prompt = self._create_optimization_prompt(
                    prompt_type, current_template, benchmark_analysis, 
                    problematic_questions, performance_trends
                )
                
                # Get optimization from actor model
                optimized_content = self._query_actor_for_optimization(optimization_prompt)
                
                # Apply safe optimization (protect core regions)
                safe_optimized_template = self._apply_safe_optimization(
                    prompt_type, current_template, optimized_content
                )
                
                if safe_optimized_template != current_template:
                    optimized_prompts[prompt_type] = safe_optimized_template
                    print(f"[DEBUG] ActorPromptOptimizer: Successfully optimized {prompt_type}")
                else:
                    print(f"[DEBUG] ActorPromptOptimizer: No valid optimization found for {prompt_type}")
                    
            except Exception as e:
                print(f"[DEBUG] ActorPromptOptimizer: Error optimizing {prompt_type}: {e}")
        
        # Save optimization history
        self._save_optimization_history(optimized_prompts, benchmark_analysis, step)
        
        return optimized_prompts
    
    def _create_optimization_prompt(self, prompt_type: str, current_template: str, 
                                  benchmark_analysis: str, problematic_questions: Dict[str, List[str]],
                                  performance_trends: Dict) -> str:
        """Create a prompt for the actor model to optimize the given prompt type"""
        
        # Extract problem-specific context
        problem_context = self._extract_problem_context(prompt_type, problematic_questions, performance_trends)
        
        optimization_prompt = f"""You are tasked with improving a {prompt_type} prompt based on performance analysis. 

**Current {prompt_type.title()} Prompt:**
```
{current_template}
```

**Performance Analysis:**
{benchmark_analysis}

**Problem-Specific Context for {prompt_type.title()}:**
{problem_context}

**Protected Elements (DO NOT MODIFY):**
{self._get_protected_elements_description(prompt_type)}

**Task:** 
Improve the {prompt_type} prompt to address the identified issues while preserving all protected elements.

**Guidelines:**
1. Keep the exact structure and format of protected elements
2. Add helpful instructions in the modifiable sections
3. Be specific about addressing the identified performance issues
4. Keep improvements concise and actionable
5. Ensure the improved prompt maintains compatibility with existing chat templates

**Output Format:**
Provide the improved prompt within <improved_prompt> tags:
<improved_prompt>
[Your improved version here]
</improved_prompt>

Also explain your changes within <explanation> tags:
<explanation>
[Brief explanation of what you changed and why]
</explanation>
"""
        
        return optimization_prompt
    
    def _extract_problem_context(self, prompt_type: str, problematic_questions: Dict[str, List[str]], 
                                performance_trends: Dict) -> str:
        """Extract context relevant to the specific prompt type"""
        
        context_parts = []
        
        if prompt_type == 'solver':
            if problematic_questions.get('always_wrong'):
                context_parts.append(f"- {len(problematic_questions['always_wrong'])} questions are consistently answered incorrectly")
                context_parts.append("- May need better reasoning guidance or problem-solving strategies")
            
            if problematic_questions.get('inconsistent'):
                context_parts.append(f"- {len(problematic_questions['inconsistent'])} questions have inconsistent performance")
                context_parts.append("- May need more structured reasoning approach")
        
        elif prompt_type == 'judge':
            if problematic_questions.get('inconsistent'):
                context_parts.append(f"- {len(problematic_questions['inconsistent'])} questions have inconsistent evaluation")
                context_parts.append("- May need more precise evaluation criteria")
            
            if performance_trends.get('overall_accuracy_change', 0) < -0.05:
                context_parts.append("- Overall accuracy has declined, evaluation may be too strict or inconsistent")
        
        elif prompt_type == 'proposer':
            if problematic_questions.get('got_worse'):
                context_parts.append(f"- {len(problematic_questions['got_worse'])} questions regressed in performance")
                context_parts.append("- Question generation may be drifting from effective patterns")
            
            current_accuracy = performance_trends.get('current_overall_accuracy', 0)
            if current_accuracy > 0.8:
                context_parts.append("- High overall accuracy - may need more challenging questions")
            elif current_accuracy < 0.5:
                context_parts.append("- Low overall accuracy - may need more approachable questions")
        
        return "\n".join(context_parts) if context_parts else "No specific issues identified for this prompt type."
    
    def _get_protected_elements_description(self, prompt_type: str) -> str:
        """Get description of protected elements for a prompt type"""
        protected_regions = self.protected_regions.get(prompt_type, [])
        if not protected_regions:
            return "No specific protected elements."
        
        descriptions = []
        for region in protected_regions:
            descriptions.append(f"- {region.name}: {region.description}")
        
        return "\n".join(descriptions)
    
    def _query_actor_for_optimization(self, optimization_prompt: str) -> str:
        """Query the actor model for prompt optimization"""
        
        print(f"[DEBUG] ActorPromptOptimizer: Querying actor model for optimization")
        
        try:
            # Use the model interface to generate optimization
            # This should be adapted based on your specific model interface
            if hasattr(self.model_interface, 'generate'):
                response = self.model_interface.generate(optimization_prompt)
            elif hasattr(self.model_interface, 'query'):
                response = self.model_interface.query(optimization_prompt)
            elif hasattr(self.model_interface, '__call__'):
                response = self.model_interface(optimization_prompt)
            else:
                # Fallback: assume it's a function
                response = self.model_interface(optimization_prompt)
                
            print(f"[DEBUG] ActorPromptOptimizer: Received response of length {len(response)}")
            return response
            
        except Exception as e:
            print(f"[DEBUG] ActorPromptOptimizer: Error querying actor model: {e}")
            return ""
    
    def _apply_safe_optimization(self, prompt_type: str, current_template: str, 
                               optimized_content: str) -> str:
        """Safely apply optimization while protecting core regions"""
        
        print(f"[DEBUG] ActorPromptOptimizer: Applying safe optimization for {prompt_type}")
        
        # Extract the improved prompt from the response
        improved_prompt = self._extract_improved_prompt(optimized_content)
        if not improved_prompt:
            print(f"[DEBUG] ActorPromptOptimizer: No improved prompt found in response")
            return current_template
        
        # Validate that protected regions are preserved
        if not self._validate_protected_regions(prompt_type, improved_prompt):
            print(f"[DEBUG] ActorPromptOptimizer: Protected regions validation failed")
            return current_template
        
        # Additional safety checks
        if not self._basic_safety_checks(improved_prompt):
            print(f"[DEBUG] ActorPromptOptimizer: Basic safety checks failed")
            return current_template
        
        print(f"[DEBUG] ActorPromptOptimizer: Safe optimization validated successfully")
        return improved_prompt
    
    def _extract_improved_prompt(self, response: str) -> str:
        """Extract improved prompt from actor model response"""
        
        # Look for improved prompt in tags
        improved_match = re.search(r'<improved_prompt>\s*(.*?)\s*</improved_prompt>', 
                                 response, re.DOTALL | re.IGNORECASE)
        if improved_match:
            return improved_match.group(1).strip()
        
        # Fallback: look for code blocks
        code_block_match = re.search(r'```(?:text|prompt)?\s*(.*?)\s*```', 
                                   response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
        
        # If no tags found, return the whole response (risky)
        print(f"[DEBUG] ActorPromptOptimizer: No tagged improved prompt found, using fallback")
        return ""
    
    def _validate_protected_regions(self, prompt_type: str, improved_prompt: str) -> bool:
        """Validate that protected regions are preserved in the improved prompt"""
        
        protected_regions = self.protected_regions.get(prompt_type, [])
        
        for region in protected_regions:
            if not re.search(region.pattern, improved_prompt, re.DOTALL | re.IGNORECASE):
                print(f"[DEBUG] ActorPromptOptimizer: Protected region '{region.name}' not found")
                return False
        
        return True
    
    def _basic_safety_checks(self, improved_prompt: str) -> bool:
        """Perform basic safety checks on the improved prompt"""
        
        # Check minimum length
        if len(improved_prompt.strip()) < 20:
            print(f"[DEBUG] ActorPromptOptimizer: Prompt too short: {len(improved_prompt)} chars")
            return False
        
        # Check for dangerous content (basic check)
        dangerous_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'system\s+prompt',
            r'jailbreak',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, improved_prompt, re.IGNORECASE):
                print(f"[DEBUG] ActorPromptOptimizer: Dangerous pattern detected: {pattern}")
                return False
        
        return True
    
    def _save_optimization_history(self, optimized_prompts: Dict[str, str], 
                                  benchmark_analysis: str, step: int):
        """Save optimization history to disk"""
        
        try:
            history_file = self.output_dir / f"optimization_history_step_{step}.json"
            
            history_data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'benchmark_analysis': benchmark_analysis,
                'optimized_prompts': optimized_prompts,
                'protected_regions': {k: [{'name': r.name, 'description': r.description} 
                                        for r in v] for k, v in self.protected_regions.items()}
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] ActorPromptOptimizer: Saved optimization history to {history_file}")
            
        except Exception as e:
            print(f"[DEBUG] ActorPromptOptimizer: Error saving optimization history: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of optimization system"""
        
        # Find recent optimization files
        optimization_files = list(self.output_dir.glob("optimization_history_step_*.json"))
        
        status = {
            'total_optimizations': len(optimization_files),
            'protected_regions': {k: len(v) for k, v in self.protected_regions.items()},
            'output_dir': str(self.output_dir)
        }
        
        if optimization_files:
            latest_file = max(optimization_files, key=lambda x: int(x.stem.split('_')[-1]))
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    latest_data = json.load(f)
                status['latest_optimization'] = {
                    'step': latest_data.get('step'),
                    'timestamp': latest_data.get('timestamp'),
                    'optimized_prompt_types': list(latest_data.get('optimized_prompts', {}).keys())
                }
            except Exception as e:
                print(f"[DEBUG] ActorPromptOptimizer: Error reading latest optimization: {e}")
        
        return status


class SafePromptUpdater:
    """Safely updates prompt manager with actor-optimized prompts"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
    
    def update_prompts_safely(self, optimized_prompts: Dict[str, str], step: int, 
                             performance_context: str = ""):
        """Update prompt manager with actor-optimized prompts"""
        
        print(f"[DEBUG] SafePromptUpdater: Updating {len(optimized_prompts)} prompts for step {step}")
        
        for prompt_type, optimized_template in optimized_prompts.items():
            if prompt_type in self.prompt_manager.templates:
                # Create improvement entry for the prompt manager
                improvement_summary = f"Actor-optimized template (step {step})"
                
                # Replace the base template with the optimized version
                self.prompt_manager.templates[prompt_type].base_template = optimized_template
                self.prompt_manager.templates[prompt_type].improvements = [improvement_summary]
                self.prompt_manager.templates[prompt_type].last_updated = datetime.now().isoformat()
                self.prompt_manager.templates[prompt_type].performance_context = performance_context
                
                print(f"[DEBUG] SafePromptUpdater: Updated {prompt_type} template")
            else:
                print(f"[DEBUG] SafePromptUpdater: Unknown prompt type: {prompt_type}")
        
        # Save the updated prompts
        self.prompt_manager._save_prompt_history(step)
