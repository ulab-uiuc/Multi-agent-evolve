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

from math_verify import parse, verify

from absolute_zero_reasoner.rewards.math_utils import grade_answer_mathd, grade_answer_sympy


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


def get_gt_reward(solution_str: str, ground_truth: str, extraction_type: str, metric: str, math_metric: str = 'deepscaler', boxed_retry: bool = False) -> float:
    answer = extract_answer(solution_str, extraction_type, boxed_retry=boxed_retry)
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
    else:
        raise ValueError(f"Invalid metric: {metric}")


def extract_answer(solution_str: str, extraction_type: str, boxed_retry: bool = False) -> str:
    if extraction_type.startswith('answer'):
        if "<answer>" in solution_str:
            answer = solution_str.split("<answer>")[-1].split("</answer>")[0]
        else:
            if boxed_retry:
                boxed_answer = last_boxed_only_string(solution_str)
                answer = boxed_answer if boxed_answer is not None else solution_str
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
        return solution_str


def get_format_reward(
    solution_str: str,
    extraction_type: str,
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
    boxed_retry: bool = False,
) -> Tuple[float, Dict[str, float]]:
    solution_str = solution_str.split(splitter)[1].strip()
    solution_str = solution_str.strip('\"\'') 
    gt_reward = get_gt_reward(solution_str, ground_truth, extraction_type, extra_info['metric'], math_metric, boxed_retry=boxed_retry)
    format_reward = get_format_reward(solution_str, extraction_type)
    if extra_info['split'] == 'train':
        if extraction_type.startswith('answer') or extraction_type.startswith('boxed'):
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
