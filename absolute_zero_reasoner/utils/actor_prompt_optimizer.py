"""
Actor-driven prompt optimization system that uses the trained model to improve prompts.
"""

import json
import re
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from absolute_zero_reasoner.utils.prompt_manager import PromptManager
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter

@dataclass 
class ProtectedRegion:
    """Defines a protected region in a prompt template"""
    name: str
    pattern: str  # Regex pattern to match the protected region
    description: str
    

class ActorPromptOptimizer:
    """Uses the actor model to optimize prompts based on benchmark analysis"""
    
    def __init__(self, model_interface, prompt_manager: PromptManager, output_dir: str = "./actor_prompt_optimization",
                 infer_together: bool = False):
        self.model_interface = model_interface
        self.prompt_manager = prompt_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configuration for judge type determination
        self.infer_together = infer_together
        
        # Define protected regions that should not be modified
        # Include essential tags, template variables, and key structural elements
        self.protected_regions = {
            'solver': [
                ProtectedRegion(
                    name='question_variable',
                    pattern=r'\{question\}',
                    description='Question template variable - must be present for input as {question}'
                ),
                ProtectedRegion(
                    name='answer_tags',
                    pattern=r'<answer>.*?</answer>',
                    description='Answer output tags for structured response - must include <answer>...</answer>'
                )
            ],
            'judge_answer': [
                ProtectedRegion(
                    name='think_tags',
                    pattern=r'<think>',
                    description='Think opening tags - must include <think>...'
                ),
                ProtectedRegion(
                    name='think_close_tags', 
                    pattern=r'</think>',
                    description='Think closing tags - must include </think>'
                ),
                ProtectedRegion(
                    name='score_tags',
                    pattern=r'<score>',
                    description='Score opening tags - must include <score>...'
                ),
                ProtectedRegion(
                    name='score_close_tags',
                    pattern=r'</score>',
                    description='Score closing tags - must include </score>'
                ),
                ProtectedRegion(
                    name='question_variable',
                    pattern=r'\{question\}',
                    description='Question template variable - must be present as {question}'
                ),
                ProtectedRegion(
                    name='answer_variable',
                    pattern=r'\{answer\}',
                    description='Answer template variable - must be present as {answer}'
                ),
                ProtectedRegion(
                    name='score_range',
                    pattern=r'score from 1 to 10',
                    description='Score range specification (1-10)'
                )
            ],
            'judge_question': [
                ProtectedRegion(
                    name='think_tags',
                    pattern=r'<think>',
                    description='Think opening tags - must include <think>...'
                ),
                ProtectedRegion(
                    name='think_close_tags', 
                    pattern=r'</think>',
                    description='Think closing tags - must include </think>'
                ),
                ProtectedRegion(
                    name='score_tags',
                    pattern=r'<score>',
                    description='Score opening tags - must include <score>...'
                ),
                ProtectedRegion(
                    name='score_close_tags',
                    pattern=r'</score>',
                    description='Score closing tags - must include </score>'
                ),
                ProtectedRegion(
                    name='question_variable',
                    pattern=r'\{question\}',
                    description='Question template variable - must be present as {question}'
                ),
                ProtectedRegion(
                    name='score_range',
                    pattern=r'score from 1 to 10',
                    description='Score range specification (1-10)'
                )
            ],
            'judge_together': [
                ProtectedRegion(
                    name='think_tags',
                    pattern=r'<think>',
                    description='Think opening tags - must include <think>...'
                ),
                ProtectedRegion(
                    name='think_close_tags', 
                    pattern=r'</think>',
                    description='Think closing tags - must include </think>'
                ),
                ProtectedRegion(
                    name='score_tags',
                    pattern=r'<score>',
                    description='Score opening tags - must include <score>...'
                ),
                ProtectedRegion(
                    name='score_close_tags',
                    pattern=r'</score>',
                    description='Score closing tags - must include </score>'
                ),
                ProtectedRegion(
                    name='question_variable',
                    pattern=r'\{question\}',
                    description='Question template variable - must be present as {question}'
                ),
                ProtectedRegion(
                    name='answer_variable',
                    pattern=r'\{answer\}',
                    description='Answer template variable - must be present as {answer}'
                ),
                ProtectedRegion(
                    name='score_range',
                    pattern=r'score from 1 to 10',
                    description='Score range specification (1-10)'
                )
            ],
            # Backward compatibility: generic judge points to judge_answer
            'judge': [
                ProtectedRegion(
                    name='think_tags',
                    pattern=r'<think>',
                    description='Think opening tags - must include <think>...'
                ),
                ProtectedRegion(
                    name='think_close_tags', 
                    pattern=r'</think>',
                    description='Think closing tags - must include </think>'
                ),
                ProtectedRegion(
                    name='score_tags',
                    pattern=r'<score>',
                    description='Score opening tags - must include <score>...'
                ),
                ProtectedRegion(
                    name='score_close_tags',
                    pattern=r'</score>',
                    description='Score closing tags - must include </score>'
                ),
                ProtectedRegion(
                    name='question_variable',
                    pattern=r'\{question\}',
                    description='Question template variable - must be present as {question}'
                ),
                ProtectedRegion(
                    name='answer_variable',
                    pattern=r'\{answer\}',
                    description='Answer template variable - must be present as {answer}'
                ),
                ProtectedRegion(
                    name='score_range',
                    pattern=r'score from 1 to 10',
                    description='Score range specification (1-10)'
                )
            ],
            'proposer': [
                ProtectedRegion(
                    name='think_tags',
                    pattern=r'<think>',
                    description='Think opening tags - must include <think>...'
                ),
                ProtectedRegion(
                    name='think_close_tags',
                    pattern=r'</think>',
                    description='Think closing tags - must include </think> '
                ),
                ProtectedRegion(
                    name='question_tags',
                    pattern=r'<question>',
                    description='Question opening tags - must include <question>...'
                ),
                ProtectedRegion(
                    name='question_close_tags',
                    pattern=r'</question>',
                    description='Question closing tags - must include </question>'
                )
            ]
        }
        
        print(f"[DEBUG] ActorPromptOptimizer initialized with infer_together={self.infer_together}")
        print(f"[DEBUG] ActorPromptOptimizer initialized with protected regions: {list(self.protected_regions.keys())}")
    
    def get_active_judge_types(self) -> List[str]:
        """Get the list of judge types that are actually used based on configuration"""
        if self.infer_together:
            # When infer_together=True, only use judge_together
            return ['judge_together']
        else:
            # When infer_together=False, use separate judge types
            return ['judge_answer', 'judge_question']
    
    def optimize_prompts_from_analysis(self, benchmark_analysis: str, problematic_questions: Dict[str, List[str]], 
                                     performance_trends: Dict, step: int) -> Dict[str, str]:
        """Use actor model to optimize prompts based on benchmark analysis"""
        
        PrettyPrinter.section_header(f"[ACTOR OPTIMIZER DEBUG] Starting Prompt Optimization for Step {step}")
        PrettyPrinter.status("BENCHMARK ANALYSIS", f"Analysis length: {len(benchmark_analysis)} chars", "info")
        PrettyPrinter.status("ANALYSIS CONTENT", benchmark_analysis, "info")
        
        # Show problematic questions details
        if problematic_questions:
            PrettyPrinter.status("PROBLEMATIC QUESTIONS", f"Found {sum(len(v) for v in problematic_questions.values())} total problematic questions", "warn")
            for category, questions in problematic_questions.items():
                PrettyPrinter.status(f"CATEGORY {category.upper()}", f"{len(questions)} questions", "warn")
                for i, question in enumerate(questions[:2]):  # Show first 2 per category
                    PrettyPrinter.status(f"  QUESTION {i+1}", question, "warn")
        
        # Show performance trends
        PrettyPrinter.status("PERFORMANCE TRENDS", f"Trends data: {str(performance_trends)}", "info")
        print("-" * 80)
        
        optimized_prompts = {}
        
        # Get active judge types based on configuration
        active_judge_types = self.get_active_judge_types()
        PrettyPrinter.status("ACTIVE JUDGE TYPES", f"Using: {active_judge_types}", "info")
        
        # Optimize each prompt type (including only active judge types)
        prompt_types_to_optimize = ['solver', 'proposer'] + active_judge_types
        PrettyPrinter.status("OPTIMIZATION PLAN", f"Will optimize: {prompt_types_to_optimize}", "info")
        print("-" * 80)
        for prompt_type in prompt_types_to_optimize:
            PrettyPrinter.section_header(f"[ACTOR OPTIMIZER DEBUG] Optimizing {prompt_type.upper()} Prompt")
            
            try:
                current_template = self.prompt_manager.get_template(prompt_type)
                PrettyPrinter.status("CURRENT TEMPLATE", f"Length: {len(current_template)} chars", "info")
                PrettyPrinter.status("TEMPLATE CONTENT", current_template, "info")
                print("-" * 60)
                
                optimization_prompt = self._create_optimization_prompt(
                    prompt_type, current_template, benchmark_analysis, 
                    problematic_questions, performance_trends
                )
                
                PrettyPrinter.status("OPTIMIZATION PROMPT", f"Generated prompt length: {len(optimization_prompt)} chars", "info")
                PrettyPrinter.status("OPTIMIZATION CONTENT", optimization_prompt, "info")
                print("-" * 40)
                
                # Get optimization from actor model
                PrettyPrinter.status("ACTOR QUERY", f"Sending prompt to actor model for {prompt_type}", "info")
                optimized_content = self._query_actor_for_optimization(optimization_prompt)
                
                PrettyPrinter.status("ACTOR RESPONSE", f"Received response length: {len(optimized_content)} chars", "success")
                PrettyPrinter.status("RESPONSE CONTENT", optimized_content, "success")
                print("-" * 40)
                
                # Apply safe optimization (protect core regions)
                PrettyPrinter.status("SAFETY CHECK", f"Applying safe optimization for {prompt_type}", "info")
                safe_optimized_template = self._apply_safe_optimization(
                    prompt_type, current_template, optimized_content
                )
                
                if safe_optimized_template != current_template:
                    optimized_prompts[prompt_type] = safe_optimized_template
                    PrettyPrinter.status("OPTIMIZATION RESULT", f"Successfully optimized {prompt_type}", "success")
                    PrettyPrinter.status("NEW TEMPLATE", safe_optimized_template, "success")
                else:
                    PrettyPrinter.status("OPTIMIZATION RESULT", f"No valid optimization found for {prompt_type}", "warn")
                    
            except Exception as e:
                PrettyPrinter.status("ERROR", f"Error optimizing {prompt_type}: {e}", "error")
                import traceback
                print(f"[DEBUG] ActorPromptOptimizer: Full traceback: {traceback.format_exc()}")
            
            print("-" * 80)
        
        # Save optimization history
        PrettyPrinter.section_header("[ACTOR OPTIMIZER DEBUG] Saving Optimization History")
        PrettyPrinter.status("SAVE STATUS", f"Saving history for step {step}", "info")
        PrettyPrinter.status("OPTIMIZED PROMPTS", f"Saving {len(optimized_prompts)} optimized prompts: {list(optimized_prompts.keys())}", "info")
        PrettyPrinter.status("BENCHMARK DATA", f"Benchmark analysis length: {len(benchmark_analysis)} chars", "info")
        
        self._save_optimization_history(optimized_prompts, benchmark_analysis, step)
        PrettyPrinter.status("SAVE COMPLETE", "Optimization history save completed", "success")
        
        PrettyPrinter.section_header("[ACTOR OPTIMIZER DEBUG] Final Optimization Results")
        PrettyPrinter.status("TOTAL OPTIMIZED", f"Successfully optimized {len(optimized_prompts)} prompt types", "success")
        for prompt_type in optimized_prompts.keys():
            PrettyPrinter.status("  OPTIMIZED TYPE", prompt_type, "success")
        
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
Improve the {prompt_type} prompt to address the identified issues while preserving all protected elements and don't add any new protected elements.

**Guidelines:**
1. Keep the exact structure and format of protected elements
2. Add helpful instructions in the modifiable sections
3. Be specific about addressing the identified performance issues
4. Keep improvements concise and actionable
5. Ensure the improved prompt maintains compatibility with existing chat templates
6. DO NOT add new template variables like {{your_answer}}, {{user_input}}, etc. - only use existing variables

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
        
        PrettyPrinter.status("ACTOR MODEL QUERY", "Sending optimization prompt to actor model", "info")
        PrettyPrinter.status("INPUT PROMPT", optimization_prompt, "info")
        print("-" * 60)
        
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
                
            PrettyPrinter.status("ACTOR MODEL RESPONSE", f"Received response of length {len(response)}", "success")
            PrettyPrinter.status("FULL RESPONSE", response, "success")
            return response
            
        except Exception as e:
            PrettyPrinter.status("ACTOR MODEL ERROR", f"Error querying actor model: {e}", "error")
            import traceback
            PrettyPrinter.status("ERROR TRACEBACK", traceback.format_exc(), "error")
            return ""
    
    def _apply_safe_optimization(self, prompt_type: str, current_template: str, 
                               optimized_content: str) -> str:
        """Safely apply optimization while protecting core regions"""
        
        PrettyPrinter.status("SAFE OPTIMIZATION", f"Processing optimization for {prompt_type}", "info")
        PrettyPrinter.status("CURRENT TEMPLATE", current_template, "info")
        PrettyPrinter.status("OPTIMIZED CONTENT", optimized_content, "info")
        print("-" * 40)
        
        # Extract the improved prompt from the response
        improved_prompt = self._extract_improved_prompt(optimized_content)
        if not improved_prompt:
            PrettyPrinter.status("EXTRACTION FAILED", "No improved prompt found in response", "warn")
            return current_template
        
        PrettyPrinter.status("EXTRACTED PROMPT", improved_prompt, "success")
        print("-" * 40)
        
        # Auto-complete missing protected elements
        PrettyPrinter.status("AUTO-COMPLETION", "Adding missing protected elements", "info")
        improved_prompt = self._auto_complete_protected_elements(prompt_type, improved_prompt, current_template)
        PrettyPrinter.status("AFTER AUTO-COMPLETION", improved_prompt, "info")
        print("-" * 40)
        
        # Validate that protected regions are preserved
        if not self._validate_protected_regions(prompt_type, improved_prompt):
            PrettyPrinter.status("VALIDATION FAILED", "Protected regions validation failed", "error")
            return current_template
        
        PrettyPrinter.status("PROTECTED REGIONS", "Validation passed", "success")
        
        # Check for unauthorized template variables
        if not self._validate_template_variables(current_template, improved_prompt):
            PrettyPrinter.status("TEMPLATE VARS", "Template variable validation failed", "error")
            return current_template
        
        PrettyPrinter.status("TEMPLATE VARS", "Validation passed", "success")
        
        # Additional safety checks
        if not self._basic_safety_checks(improved_prompt):
            PrettyPrinter.status("SAFETY CHECKS", "Basic safety checks failed", "error")
            return current_template
        
        PrettyPrinter.status("SAFETY CHECKS", "All safety checks passed", "success")
        PrettyPrinter.status("FINAL OPTIMIZED TEMPLATE", improved_prompt, "success")
        return improved_prompt
    
    def _auto_complete_protected_elements(self, prompt_type: str, improved_prompt: str, 
                                        original_prompt: str) -> str:
        """Auto-complete missing protected elements from original prompt"""
        
        # Map prompt types for backward compatibility
        actual_prompt_type = prompt_type
        if prompt_type == 'judge' and prompt_type not in self.protected_regions:
            actual_prompt_type = 'judge_answer'  # Default fallback
        
        protected_regions = self.protected_regions.get(actual_prompt_type, [])
        completed_prompt = improved_prompt
        
        # Check and add missing protected elements
        for region in protected_regions:
            if not re.search(region.pattern, completed_prompt, re.DOTALL | re.IGNORECASE):
                # Find the element in the original prompt
                original_match = re.search(region.pattern, original_prompt, re.DOTALL | re.IGNORECASE)
                
                if original_match:
                    # Add the missing element with appropriate context
                    missing_element = original_match.group(0)
                    
                    # Smart placement based on prompt type and element type
                    if region.name == 'question_variable' and 'solver' in actual_prompt_type:
                        # For solver: ensure {question} is present
                        completed_prompt += f"\n\n**Input Format**: {missing_element}"
                    elif region.name == 'question_variable' and 'judge' in actual_prompt_type:
                        # For judge: ensure {question} variable
                        completed_prompt += f"\n\n**Question**: {missing_element}"
                    elif region.name == 'answer_variable' and 'judge' in actual_prompt_type:
                        # For judge: ensure {answer} variable
                        completed_prompt += f"\n\n**Answer**: {missing_element}"
                    elif 'answer_tags' in region.name:
                        # For solver: ensure <answer> tags
                        completed_prompt += f"\n\n**Output Format**: Please provide your response in {missing_element} tags."
                    elif 'think_tags' in region.name and 'close' not in region.name:
                        # Opening think tags
                        completed_prompt += f"\n\nFirst analyze in {missing_element} tags:"
                    elif 'think_close_tags' in region.name:
                        # Closing think tags
                        if '<think>' in completed_prompt:
                            # Insert before the end if <think> exists
                            completed_prompt += f"\n{missing_element}"
                        else:
                            completed_prompt += f"\n\nEnd analysis with {missing_element}"
                    elif 'score_tags' in region.name and 'close' not in region.name:
                        # Opening score tags
                        completed_prompt += f"\n\nProvide final rating in {missing_element} format."
                    elif 'score_close_tags' in region.name:
                        # Closing score tags
                        completed_prompt += f"\n{missing_element}"
                    elif 'question_tags' in region.name and 'close' not in region.name:
                        # Opening question tags for proposer
                        completed_prompt += f"\n\nGenerate your question in {missing_element} format."
                    elif 'question_close_tags' in region.name:
                        # Closing question tags
                        completed_prompt += f"\n{missing_element}"
                    elif 'score_range' in region.name:
                        # Score range specification
                        completed_prompt += f"\n\n**Rating**: Provide a {missing_element}."
                    else:
                        # Default: add with context
                        completed_prompt += f"\n\n**Required**: {missing_element}"
                    
                    print(f"[DEBUG] ActorPromptOptimizer: Auto-completed missing element: {region.name}")
                else:
                    # If not found in original, add generic format requirement
                    if region.name == 'question_variable':
                        completed_prompt += "\n\n**Input Format**: {question}"
                    elif region.name == 'answer_variable':
                        completed_prompt += "\n\n**Answer Format**: {answer}"
                    elif 'answer_tags' in region.name:
                        completed_prompt += "\n\n**Output Format**: Provide your response in <answer></answer> tags."
                    elif region.name == 'score_range':
                        completed_prompt += "\n\n**Rating**: Provide a score from 1 to 10."
                    elif 'think_tags' in region.name:
                        completed_prompt += "\n\n**Analysis**: Use <think></think> tags for your reasoning."
                    elif 'question_tags' in region.name:
                        completed_prompt += "\n\n**Output**: Generate questions in <question></question> tags."
                    
                    print(f"[DEBUG] ActorPromptOptimizer: Added generic format for missing element: {region.name}")
        
        return completed_prompt
    
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
        
        # Map prompt types for backward compatibility
        actual_prompt_type = prompt_type
        if prompt_type == 'judge' and prompt_type not in self.protected_regions:
            actual_prompt_type = 'judge_answer'  # Default fallback
        
        protected_regions = self.protected_regions.get(actual_prompt_type, [])
        
        for region in protected_regions:
            if not re.search(region.pattern, improved_prompt, re.DOTALL | re.IGNORECASE):
                print(f"[DEBUG] ActorPromptOptimizer: Protected region '{region.name}' not found")
                return False
        
        return True
    
    def _validate_template_variables(self, original_prompt: str, improved_prompt: str) -> bool:
        """Validate that no new unauthorized template variables were added"""
        
        # Extract all template variables from both prompts
        original_vars = set(re.findall(r'\{[^}]+\}', original_prompt))
        improved_vars = set(re.findall(r'\{[^}]+\}', improved_prompt))
        
        # Find new variables in the improved prompt
        new_vars = improved_vars - original_vars
        
        if new_vars:
            print(f"[DEBUG] ActorPromptOptimizer: New template variables detected: {new_vars}")
            
            # Allow only very specific new variables that are safe
            allowed_new_vars: set[str] = set()
            
            unauthorized_vars = new_vars - allowed_new_vars
            if unauthorized_vars:
                print(f"[DEBUG] ActorPromptOptimizer: Unauthorized template variables: {unauthorized_vars}")
                print(f"[DEBUG] ActorPromptOptimizer: Only allowed new variables: {allowed_new_vars}")
                return False
            else:
                print(f"[DEBUG] ActorPromptOptimizer: New variables are authorized: {new_vars}")
        
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
        
        # Check for new template variables that could affect chat templates
        # These are forbidden variables that might interfere with chat template processing
        forbidden_template_vars = [
            r'\{your_answer\}',
            r'\{user_input\}', 
            r'\{assistant_response\}',
            r'\{system_message\}',
            r'\{chat_template\}',
            r'\{user_prompt\}',
            r'\{ai_response\}',
            r'\{model_response\}',
            r'\{conversation\}',
            r'\{dialogue\}',
            r'\{user_query\}',
            r'\{bot_response\}',
            r'\{human_input\}',
            r'\{machine_output\}',
            r'\{input_text\}',
            r'\{output_text\}',
            r'\{prompt_template\}',
            r'\{instruction_template\}',
        ]
        
        for forbidden_var in forbidden_template_vars:
            if re.search(forbidden_var, improved_prompt, re.IGNORECASE):
                print(f"[DEBUG] ActorPromptOptimizer: Forbidden template variable detected: {forbidden_var}")
                print(f"[DEBUG] ActorPromptOptimizer: This could interfere with chat template processing")
                return False
        
        return True
    
    def _save_optimization_history(self, optimized_prompts: Dict[str, str], 
                                  benchmark_analysis: str, step: int):
        """Save optimization history to disk"""
        
        print(f"[DEBUG] ActorPromptOptimizer: Starting to save optimization history for step {step}")
        print(f"[DEBUG] ActorPromptOptimizer: Output directory: {self.output_dir}")
        print(f"[DEBUG] ActorPromptOptimizer: Directory exists: {self.output_dir.exists()}")
        print(f"[DEBUG] ActorPromptOptimizer: Number of optimized prompts: {len(optimized_prompts)}")
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] ActorPromptOptimizer: Ensured output directory exists")
            
            history_file = self.output_dir / f"optimization_history_step_{step}.json"
            print(f"[DEBUG] ActorPromptOptimizer: Target file path: {history_file}")
            
            # Check if we have write permissions
            try:
                test_file = self.output_dir / "test_write.tmp"
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # Delete test file
                print(f"[DEBUG] ActorPromptOptimizer: Write permissions verified")
            except Exception as perm_e:
                print(f"[DEBUG] ActorPromptOptimizer: Write permission test failed: {perm_e}")
                return
            
            history_data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'benchmark_analysis': benchmark_analysis,
                'optimized_prompts': optimized_prompts,
                'protected_regions': {k: [{'name': r.name, 'description': r.description} 
                                        for r in v] for k, v in self.protected_regions.items()}
            }
            
            print(f"[DEBUG] ActorPromptOptimizer: Created history data structure")
            print(f"[DEBUG] ActorPromptOptimizer: History data keys: {list(history_data.keys())}")
            
            # Try to serialize to JSON first to catch any serialization issues
            try:
                json_str = json.dumps(history_data, indent=2, ensure_ascii=False)
                print(f"[DEBUG] ActorPromptOptimizer: JSON serialization successful, length: {len(json_str)}")
            except Exception as json_e:
                print(f"[DEBUG] ActorPromptOptimizer: JSON serialization failed: {json_e}")
                return
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] ActorPromptOptimizer: Successfully saved optimization history to {history_file}")
            
            # Verify the file was created and has content
            if history_file.exists():
                file_size = history_file.stat().st_size
                print(f"[DEBUG] ActorPromptOptimizer: File created successfully, size: {file_size} bytes")
            else:
                print(f"[DEBUG] ActorPromptOptimizer: ERROR - File was not created!")
            
        except Exception as e:
            print(f"[DEBUG] ActorPromptOptimizer: Error saving optimization history: {e}")
            import traceback
            print(f"[DEBUG] ActorPromptOptimizer: Full traceback: {traceback.format_exc()}")
    
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
        print(f"[DEBUG] SafePromptUpdater: About to save prompt history for step {step}")
        try:
            self.prompt_manager._save_prompt_history(step)
            print(f"[DEBUG] SafePromptUpdater: Successfully saved prompt history")
        except Exception as e:
            print(f"[DEBUG] SafePromptUpdater: Error saving prompt history: {e}")
            import traceback
            print(f"[DEBUG] SafePromptUpdater: Full traceback: {traceback.format_exc()}")
