#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://github.com/QwenLM/QwQ/blob/main/eval/eval/math_opensource_utils/python_executor.py

import copy
import datetime
import io
import logging
import pickle
import traceback
from concurrent.futures import TimeoutError
from contextlib import redirect_stdout
from functools import partial
from typing import Any, Dict, Optional, List, Tuple
import ast
import time

import numpy as np
import dateutil.relativedelta
import regex
from pebble import ProcessPool
from timeout_decorator import timeout
from tqdm import tqdm

from absolute_zero_reasoner.utils.code_utils.templates import (
    RUN_CODE_TEMPLATE,
    EVAL_INPUT_PREDICTION_TEMPLATE,
    EVAL_OUTPUT_PREDICTION_TEMPLATE,
    VALIDATE_CODE_TEMPLATE,
    CHECK_DETERMINISM_TEMPLATE,
    EVAL_K_INPUT_PREDICTION_TEMPLATE,
    EVAL_K_OUTPUT_PREDICTION_TEMPLATE,
)
from absolute_zero_reasoner.utils.code_utils.checks import contains_banned_imports
from absolute_zero_reasoner.utils.code_utils.parsers import parse_error


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r'(\s|^)?input\(', code_piece):
            # regex.search(r'(\s|^)?os.', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)

        # TODO: use: https://github.com/shroominic/codebox-api
        # @high safe exec in sandbox
        # byte_code = compile_restricted(
        #     code_piece,
        #     filename='<inline code>',
        #     mode='exec'
        # )
        # print("global vars:", self._global_vars)
        # _print_ = PrintCollector
        # exec(byte_code, {'__builtins__': utility_builtins}, None)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars['answer']


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime,
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 10,
        ast_check: bool = False,
        max_workers: int = 1,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length
        self.ast_check = ast_check
        self.max_workers = max_workers
        self._process_pool = None

    def __del__(self):
        try:
            self.cleanup()
            # self.pool.terminate()
        except Exception as e:
            print(f"Error terminating pool: {e}")
            pass

    def cleanup(self):
        """Explicitly clean up the process pool"""
        if self._process_pool is not None:
            self._process_pool.close()
            self._process_pool.join()
            self._process_pool = None

    def _get_process_pool(self, size_hint):
        """Get or create a ProcessPool with appropriate size"""
        if self._process_pool is None:
            self._process_pool = ProcessPool(max_workers=min(size_hint, self.max_workers))
        return self._process_pool

    def process_generation_to_code(self, gens: str):
        return [g.strip().split('\n') for g in gens]
    
    def run_code(self, code: str, inputs: str, imports: List[str] = []) -> Tuple[str, str]:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        code_snippet = RUN_CODE_TEMPLATE.format(code=code, inputs=inputs)
        # print(code_snippet)
        if self.ast_check:
            try:
                ast.parse(code_snippet)
            except:
                return '', 'error'
        return self.apply(code_snippet)

    def validate_code(self, code: str, inputs: str, imports: List[str] = []) -> bool:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        code_snippet = VALIDATE_CODE_TEMPLATE.format(code=code, inputs=inputs)
        if self.ast_check:
            try:
                ast.parse(code_snippet)
            except:
                return False
        _, status = self.apply(code_snippet)
        return not 'error' in status.lower()

    def eval_input_prediction(self, code: str, gold_output: str, agent_input: str, imports: List[str] = []) -> float:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        code_snippet = EVAL_INPUT_PREDICTION_TEMPLATE.format(code=code, gold_output=gold_output, agent_input=agent_input)
        if self.ast_check:
            try:
                ast.parse(code_snippet)
            except:
                return 0.0
        max_retries = 3
        for retry in range(max_retries):
            try:
                correct, status = self.apply(code_snippet)
                return 0.0 if 'error' in status.lower() or not eval(correct) else 1.0
            except Exception as e:
                if retry == max_retries - 1:
                    error_details = traceback.format_exc()
                    print(f"Error in eval_input_prediction: {e}\n{error_details}")
                    return
                time.sleep(0.1 * (retry + 1))  # Exponential backoff

    def eval_output_prediction(self, code: str, gold_output: str, agent_output: str, imports: List[str] = []) -> float:
        try: # fast check if we dont need to run the code
            if eval(gold_output) == eval(agent_output):
                return 1.0
        except:
            pass
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        code_snippet = EVAL_OUTPUT_PREDICTION_TEMPLATE.format(code=code, gold_output=gold_output, agent_output=agent_output)
        if self.ast_check:
            try:
                ast.parse(code_snippet)
            except:
                return 0.0
        max_retries = 3
        for retry in range(max_retries):
            try:
                correct, status = self.apply(code_snippet)
                return 0.0 if 'error' in status.lower() or not eval(correct) else 1.0
            except Exception as e:
                if retry == max_retries - 1:
                    error_details = traceback.format_exc()
                    print(f"Error in eval_output_prediction: {e}\n{error_details}")
                    return
                time.sleep(0.1 * (retry + 1))  # Exponential backoff

    def eval_k_input_prediction(self, code: str, gold_output: str, k_agent_inputs: List[str], imports: List[str] = []) -> List[float]:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        invalid_lists = []
        valid_k_agent_inputs = []
        for k_agent_input in k_agent_inputs:
            try:
                ast.parse(f'f({k_agent_input})')
                valid_k_agent_inputs.append(k_agent_input)
            except:
                invalid_lists.append(0.0)
        acc_list, status = self.apply(EVAL_K_INPUT_PREDICTION_TEMPLATE(code=code, gold_output=gold_output, k_agent_inputs=valid_k_agent_inputs))
        assert 'error' not in status.lower()
        output_acc = eval(acc_list) + invalid_lists
        assert len(output_acc) == len(k_agent_inputs)
        return output_acc

    def eval_k_output_prediction(self, code: str, gold_output: str, k_agent_outputs: List[str], imports: List[str] = []) -> List[float]:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        invalid_lists = []
        valid_k_agent_outputs = []
        for k_agent_output in k_agent_outputs:
            try:
                if k_agent_output != '':
                    ast.parse(f'f({k_agent_output})')
                    valid_k_agent_outputs.append(k_agent_output)
                else:
                    invalid_lists.append(0.0)
            except:
                invalid_lists.append(0.0)
        acc_list, status = self.apply(EVAL_K_OUTPUT_PREDICTION_TEMPLATE(code=code, gold_output=gold_output, k_agent_outputs=valid_k_agent_outputs))
        assert 'error' not in status.lower()
        output_acc = eval(acc_list) + invalid_lists
        assert len(output_acc) == len(k_agent_outputs)
        return output_acc

    def check_all(
        self,
        code: str,
        inputs: str,
        banned_keywords: List[str] = [],
        check_determinism: bool = True,
        imports: List[str] = [],
        check_error: bool = False,
        banned_keywords_for_errors_and_exceptions: List[str] = [],
    ) -> Tuple[bool, str]:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        if contains_banned_imports(code=code, banned_keywords=banned_keywords, banned_keywords_for_errors_and_exceptions=banned_keywords_for_errors_and_exceptions if check_error else []):
            return False, None
        if check_error:
            code_snippet = RUN_CODE_TEMPLATE.format(code=code, inputs=inputs)
            try:
                ast.parse(code_snippet)
            except:
                return False, 'error'
            output, status = self.apply(code_snippet)
            if check_determinism: # run the code again, see if outputs are same
                output_2, status_2 = self.apply(code_snippet)
                if status_2.lower() != status.lower() and output != output_2:
                    return False, 'error'
            # True if the code is valid code but might have error, output no error if the code returns something
            return True, 'NoError' if status.lower() == 'done' else parse_error(status)
        else:
            if check_determinism:
                code_snippet = CHECK_DETERMINISM_TEMPLATE.format(code=code, inputs=inputs)
            else:
                code_snippet = RUN_CODE_TEMPLATE.format(code=code, inputs=inputs)
            if self.ast_check:
                try:
                    ast.parse(code_snippet)
                except:
                    return False, 'error'
            output, status = self.apply(code_snippet)
            return not 'error' in status.lower(), output

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout=None,
        runtime=None,
        answer_symbol=None,
        answer_expr=None,
        timeout_length=10,
        auto_mode=False
    ):
        try:
            if auto_mode:
                if "print(" in code[-1]:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    program_io.seek(0)
                    result = program_io.read()
                else:
                    # print(code)
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            else:
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    program_io.seek(0)
                    result = program_io.read()
                elif answer_symbol:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    result = runtime._global_vars[answer_symbol]
                elif answer_expr:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code))
                    result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
                else:
                    timeout(timeout_length)(runtime.exec_code)('\n'.join(code[:-1]))
                    result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)           # codec check
            pickle.dumps(result)  # serialization check
        except:
            result = ''
            report = traceback.format_exc().split('\n')[-2]
        return result, report

    def apply(self, code):
        return self.batch_apply([code])[0]

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        
        pool = self._get_process_pool(len(all_code_snippets))
        executor = partial(
            self.execute,
            get_answer_from_stdout=self.get_answer_from_stdout,
            runtime=self.runtime,
            answer_symbol=self.answer_symbol,
            answer_expr=self.answer_expr,
            timeout_length=self.timeout_length,
            auto_mode=True
        )
        
        try:
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    logging.warning(f"Timeout error in code execution: {error}")
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as error:
                    logging.warning(f"Error in code execution: {error}")
                    all_exec_results.append(("", f"Error: {str(error)}"))
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()
        except Exception as e:
            logging.error(f"Critical error in batch execution: {e}")
            # Make sure we have results for all snippets
            while len(all_exec_results) < len(all_code_snippets):
                all_exec_results.append(("", f"Critical Error: {str(e)}"))
            
            # Cleanup the pool on critical errors
            self.cleanup()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results


def _test():
    batch_code = [
"""
def f(a):
    return a
print(f(1,2))
"""
    ]

    executor = PythonExecutor(get_answer_from_stdout=True)
    predictions = executor.apply(batch_code[0])
    print(predictions)


if __name__ == '__main__':
    _test()
