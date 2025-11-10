# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from collections import Counter
from typing import Tuple, List, Dict
from lxml import etree

from math_verify import parse, verify

from reason_rl.rewards.evalplus_wrapper import evaluate_sample
from reason_rl.rewards.math_utils import grade_answer_mathd, grade_answer_sympy


def choice_answer_clean(pred: str):
    """https://github.com/hkust-nlp/simpleRL-reason/blob/main/eval/grader.py"""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def get_gt_reward(solution_str: str, ground_truth: str, extraction_type: str, metric: str, math_metric: str = 'deepscaler') -> float:
    answer = extract_answer(solution_str, extraction_type)
    if metric == 'mc':
        mc_answer = choice_answer_clean(answer)
        if mc_answer == ground_truth:
            return 1.0
        if grade_answer_sympy(answer, ground_truth) or grade_answer_mathd(answer, ground_truth):
            return 1.0
        return 0.0
    elif metric == 'math':
        if math_metric == 'math_verify':
            gold = parse('\\boxed{' + ground_truth + '}')
            answer = parse('\\boxed{' + answer + '}')
            return 1.0 if verify(gold, answer) else 0.0
        elif math_metric == 'deepscaler':
            if grade_answer_sympy(answer, ground_truth) or grade_answer_mathd(answer, ground_truth):
                return 1.0
            return 0.0
        elif math_metric == 'union':
            math_verify_gold = parse('\\boxed{' + ground_truth + '}')
            math_verify_answer = parse('\\boxed{' + answer + '}')
            if grade_answer_sympy(answer, ground_truth) or grade_answer_mathd(answer, ground_truth) or verify(math_verify_gold, math_verify_answer):
                return 1.0
            return 0.0
        else:
            raise ValueError(f"Invalid math metric: {math_metric}")

    elif metric == 'code_eval':
        try:
            answer = eval(answer.strip())
        except Exception:
            return 0.0
        ground_truth = eval(ground_truth.strip())
        if answer == ground_truth:
            return 1.0
        return 0.0

    elif metric == 'evalplus':
        pattern = re.compile(rf"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(answer)
        extracted_answer = matches[-1] if len(matches) >= 1 else answer
        return evaluate_sample(**ground_truth, solution=extracted_answer)['base_passed'] * 1.0

    elif metric == 'openr1':
        return 0.0 # placeholder

    elif metric == 'em':
        if answer.lower().strip() == ground_truth.lower().strip():
            return 1.0
        return 0.0

    # we need all indices in the answer list to be in the ground truth
    elif metric == 'bon':
        try:
            answer_list = eval(answer.strip())
        except Exception: # not a valid python object
            return 0.0
        ground_truth = eval(ground_truth.strip())
        if isinstance(answer_list, list) or isinstance(answer_list, set) or isinstance(answer_list, tuple):
            # convert to list
            answer_list = list(answer_list)
            answer_list = [a for a in answer_list if a] # remove empty strings, empty lists, etc.
            for i, a in enumerate(answer_list):
                if isinstance(a, str):
                    try:
                        answer_list[i] = int(a)
                    except Exception:
                        return 0.0
                if isinstance(a, list):
                    if len(a) > 1:
                        return 0.0
                    elif len(a) == 1:
                        answer_list[i] = a[0]
                        if isinstance(answer_list[i], str):
                            try:
                                answer_list[i] = int(answer_list[i])
                            except Exception:
                                return 0.0
                    else:
                        return 0.0

            # Special cases
            if len(answer_list) == 0 and len(ground_truth) == 0:
                return 1.0
            if len(answer_list) == 0 and len(ground_truth) != 0:
                return 0.0

            try:
                # Convert to sets
                answer_set = set(answer_list)
                ground_truth_set = set(ground_truth)

                # Calculate intersection of correct answers
                correct_answers = answer_set.intersection(ground_truth_set)

                # If no correct answers found, return 0
                if len(correct_answers) == 0:
                    return 0.0

                # Base score for getting at least one right (e.g., 0.5)
                base_score = 0.5

                # Bonus score based on proportion of remaining correct answers
                if len(ground_truth_set) > 1:
                    bonus_proportion = (len(correct_answers) - 1) / (len(ground_truth_set) - 1)
                    bonus_score = (1.0 - base_score) * bonus_proportion
                else:
                    bonus_score = 1.0 - base_score

                # Combine base score and bonus
                return base_score + bonus_score
            except Exception:
                raise ValueError(f"Invalid answer: {answer}")

        else: # not a list, set, or tuple
            return 0.0

    # string matching
    elif metric == 'judge':
        assert isinstance(eval(ground_truth), bool)
        if bool(eval(ground_truth)):
            if answer.strip().lower() in ['correct', "'correct'", '"correct"', 'yes', 'true', '1', 'y', 't', 'right', 'valid', 'accurate', 'ok', 'okay', 'yep', 'yeah', 'indeed', 'affirmative', 'correct answer', 'solution is correct', 'is correct']:
                return 1.0
            return 0.0
        else: # false
            if answer.strip().lower() in ['incorrect', "'incorrect'", '"incorrect"', 'no', 'false', '0', 'n', 'f', 'wrong', 'invalid', 'inaccurate', 'nope', 'nah', 'negative', 'not correct', 'not right', 'solution is incorrect', 'solution is wrong', 'is incorrect', 'is not correct']:
                return 1.0
            return 0.0
    else:
        raise ValueError(f"Invalid metric: {metric}")


def extract_answer(solution_str: str, extraction_type: str) -> str:
    if extraction_type.startswith('answer') or extraction_type.startswith('kwai') or extraction_type.startswith('skillset'):
        if "<answer>" in solution_str:
            answer = solution_str.split("<answer>")[-1].split("</answer>")[0]
        else:
            return ''
        # Strip LaTeX math delimiters and whitespace
        answer = answer.strip()
        return answer
    elif extraction_type.startswith('boxed'):
        answer = last_boxed_only_string(solution_str)
        return answer.strip() if answer is not None else ''
    else:
        raise ValueError(f"Invalid extraction type: {extraction_type}")


def extract_thought(solution_str: str) -> str:
    if "<think>" in solution_str:
        return solution_str.split("<think>")[-1].split("</think>")[0]
    else:
        return ''


def validate_tags(text) -> tuple[bool, dict]:
    """
    Validates that the entire string is composed solely of properly nested HTML-like tags,
    allowing any whitespace (spaces, tabs, newlines) between tags or at the boundaries.
    Returns a count of the outer (top-level) tags only. Non-whitespace text outside of tags
    will cause validation to fail.

    Examples:
      - "<a>content</a >" returns (True, {"a": 1})
      - "<a><b>Nested</b></a >" returns (True, {"a": 1})
        (the inner 'b' tag is not counted as an outer tag)
      - "<div>Hello</div><p>World</p >" returns (True, {"div": 1, "p": 1})
      - "<a>Text</a > extra" returns (False, {})
      - "   <a>Text</a >\n\t<b>More</b>  " returns (True, {"a": 1, "b": 1})

    Args:
        text (str): The text containing the tags to validate.
    
    Returns:
        tuple: A tuple (is_valid, tag_counts) where is_valid is a boolean indicating
               if the text is valid and tag_counts is a dict with counts of outer tags.
    """
    # This pattern matches both opening and closing tags with a name consisting of letters and underscores only.
    tag_pattern = re.compile(r"</?([a-zA-Z_]+)>")
    
    pos = 0            # Current position in the string.
    stack = []         # Stack to keep track of nested open tags.
    outer_counts = Counter()  # Counter for outer (top-level) tags.

    # Iterate over every tag in the string.
    for match in tag_pattern.finditer(text):
        # At the outer level, allow whitespace between tags.
        if not stack:
            # If the text from the previous position to the current tag is non-empty after stripping,
            # then there is extra non-whitespace text.
            if text[pos:match.start()].strip() != "":
                return False, {}

        tag = match.group(0)
        tag_name = match.group(1)
        
        if tag.startswith("</"):  # This is a closing tag.
            if not stack or stack[-1] != tag_name:
                # Either no corresponding open tag, or the tag names do not match.
                return False, {}
            stack.pop()
        else:  # This is an opening tag.
            if not stack:
                # This opening tag is at the outer (top-level).
                outer_counts[tag_name] += 1
            stack.append(tag_name)

        # Update the current position to the end of the current tag.
        pos = match.end()

    # After processing all tags:
    # 1. Any trailing text (outside of tags) must be whitespace only.
    # 2. All tags should have been closed (stack is empty).
    if text[pos:].strip() != "" or stack:
        return False, {}

    try:
        xml_string = f'<root>{text}</root>'
        etree.fromstring(xml_string)
    except Exception:
        return False, {}

    return True, dict(outer_counts)


def count_nested_html(input_string):
    pattern = re.compile(r'<(/?)(\w+)\s*>')
    stack = []  # Stores tuples of (tag_name, was_nested_when_opened)
    nested_count = 0

    for match in pattern.finditer(input_string):
        is_closing, tag_name = match.groups()

        if not is_closing:
            # Record if this tag was opened within another tag
            was_nested = len(stack) > 0
            stack.append((tag_name, was_nested))
        else:
            # Search for matching opening tag
            found_index = -1
            for i in reversed(range(len(stack))):
                if stack[i][0] == tag_name:
                    found_index = i
                    break

            if found_index != -1:
                # Only count if the tag was originally nested when opened
                if stack[found_index][1]:
                    nested_count += 1
                # Remove all elements from found_index onward
                del stack[found_index:]

    return nested_count


def get_format_reward(
    solution_str: str,
    extraction_type: str,
    available_options: List[str] = None,
) -> float:
    if extraction_type.startswith('answer'):
        pattern = r"(?s)<think>.*?</think>\s*<answer>.*?</answer>"
        matched = re.match(pattern, solution_str)
        if matched:
            return 1.
        else:
            return 0.
    elif extraction_type.startswith('boxed'):
        if last_boxed_only_string(solution_str) is not None:
            return 1.
        else:
            return 0.
    elif extraction_type.startswith('kwai') or extraction_type.startswith('skillset'):
        valid, tag_counts = validate_tags(solution_str)
        if available_options is not None:
            for tag in tag_counts.keys():
                if tag not in available_options:
                    valid = False
        if extraction_type.startswith('kwai_vanilla') or extraction_type.startswith('skillset_vanilla'):
            return 1. if valid else 0.
        elif extraction_type.startswith('kwai_count') or extraction_type.startswith('skillset_count'):
            if not valid:
                return 0.
            return min(0.3 + sum([v for k, v in tag_counts.items() if k != 'answer']) * 0.02, 0.6)
        elif extraction_type.startswith('kwai_distinct_count') or extraction_type.startswith('skillset_distinct_count'):
            if not valid:
                return 0.
            return min(0.3 + sum([1 for k, v in tag_counts.items() if k != 'answer' and v > 0]) * 0.06, 0.6)
        else:
            raise ValueError(f"Invalid extraction type: {extraction_type}")
    else:
        raise ValueError(f"Invalid extraction type: {extraction_type}")


def extract_code_content(solution_str):
    # Check if the string starts with an XML code block
    xml_pattern = r'^```\s*xml\n(.*?)```'
    xml_match = re.match(xml_pattern, solution_str, re.DOTALL | re.IGNORECASE)

    if xml_match:
        # XML code block found at start
        return xml_match.group(1).strip()

    # Check if the string starts with any code block
    generic_pattern = r'^```\s*\w*\n(.*?)```'
    generic_match = re.match(generic_pattern, solution_str, re.DOTALL)

    if generic_match:
        # Some other code block found at start
        return generic_match.group(1).strip()

    # No code block found at start, return the original string
    return solution_str.strip()


def get_reward(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    extraction_type: str,
    splitter: str,
    math_metric: str = 'deepscaler',
    available_options: List[str] = None,
) -> Tuple[float, Dict[str, float]]:
    solution_str = solution_str.split(splitter)[1].strip()
    # sometimes model starts with code block for kwai and skillset
    if extraction_type.startswith('kwai') or extraction_type.startswith('skillset'):
        solution_str = extract_code_content(solution_str)
    solution_str = solution_str.strip('\"\'') 
    gt_reward = get_gt_reward(solution_str, ground_truth, extraction_type, extra_info['metric'], math_metric)
    format_reward = get_format_reward(solution_str, extraction_type, available_options)
    if extra_info['split'] == 'train':
        if extraction_type.startswith('kwai') or extraction_type.startswith('skillset'):
            if extraction_type.endswith('additon'):
                return gt_reward + format_reward, {'gt': gt_reward, 'format': format_reward}
            elif extraction_type.endswith('multiply'):
                return gt_reward * format_reward, {'gt': gt_reward, 'format': format_reward}
            elif extraction_type.endswith('conditional'):
                return gt_reward if format_reward else 0., {'gt': gt_reward, 'format': format_reward}
            elif extraction_type.endswith('conditional_v2'):
                # R(answer) =
                # 1 if correct formatting and correct answer
                # -0.5 if correct formatting and incorrect answer
                # -1 if incorrect formatting
                if not format_reward:
                    return -1., {'gt': gt_reward, 'format': format_reward}
                else:
                    return 1. if gt_reward else -0.5, {'gt': gt_reward, 'format': format_reward}
            else:
                raise ValueError(f"Invalid extraction type: {extraction_type}")
        elif extraction_type.startswith('answer') or extraction_type.startswith('boxed'):
            if extraction_type.endswith('conditional'):
                # R(answer) =
                # 1 if correct formatting and correct answer
                # -0.5 if correct formatting and incorrect answer
                # -1 if incorrect formatting
                if not format_reward:
                    return -1., {'gt': gt_reward, 'format': format_reward}
                # correct formatting
                else:
                    return 1. if gt_reward else -0.5, {'gt': gt_reward, 'format': format_reward}
            elif extraction_type.endswith('addition'):
                return (0.5 if format_reward else 0.) + gt_reward, {'gt': gt_reward, 'format': format_reward}
            elif extraction_type.endswith('multiply'):
                return format_reward * gt_reward, {'gt': gt_reward, 'format': format_reward}
            else:
                raise ValueError(f"Invalid extraction type: {extraction_type}")
    elif extra_info['split'] == 'test':
        return gt_reward, {'gt': gt_reward, 'format': format_reward}
    else:
        raise ValueError(f"Invalid split: {extra_info['split']}")


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    https://github.com/huggingface/open-r1
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(response: str, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        if response == "":
            return 0.0
        if len(response.split()) < ngram_size:
            return 0.0

        ngrams = set()
        total = 0
        for ng in zipngram(response, ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * max_penalty
        return reward

    return repetition_penalty_reward


if __name__ == "__main__":
    generation = """<think> To find the sum of the polynomials f(y) and g(y), we need to add the corresponding terms of each polynomial. The polynomials are:

f(y) = y^4 - 3y^3 + y - 3
g(y) = y^3 + 7y^2 - 2

Now, let's add the corresponding terms:

y^4 (from f(y)) + 0 (from g(y)) = y^4
-3y^3 (from f(y)) + y^3 (from g(y)) = -2y^3
0 (from f(y)) + 7y^2 (from g(y)) = 7y^2
y (from f(y)) + 0 (from g(y)) = y
-3 (from f(y)) - 2 (from g(y)) = -5

So, the sum of the polynomials f(y) and g(y) is:

f(y) + g(y) = y^4 - 2y^3 + 7y^2 + y - 5

</think> <answer> y^4 - 2y^3 + 7y^2 + y - 5 </answer><|endoftext|>"""
    print(get_gt_reward(generation, "y^4-2y^3+7y^2+y-5"))
