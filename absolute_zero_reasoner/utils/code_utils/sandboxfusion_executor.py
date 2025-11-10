import traceback
from typing import List, Tuple
import ast
import time
import requests
import docker
from docker.errors import DockerException
import socket

import numpy as np
from pebble import ProcessPool
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint, RunStatus

from absolute_zero_reasoner.utils.code_utils.templates import (
    RUN_CODE_TEMPLATE_REPR,
    EVAL_INPUT_PREDICTION_TEMPLATE_REPR,
    EVAL_OUTPUT_PREDICTION_TEMPLATE_REPR,
    VALIDATE_CODE_TEMPLATE_REPR,
    CHECK_DETERMINISM_TEMPLATE_REPR,
    EVAL_K_INPUT_PREDICTION_TEMPLATE,
    EVAL_K_OUTPUT_PREDICTION_TEMPLATE,
)
from absolute_zero_reasoner.utils.code_utils.checks import contains_banned_imports
from absolute_zero_reasoner.utils.code_utils.parsers import parse_error


# Docker images
IMAGES = {
    'global': 'volcengine/sandbox-fusion:server-20250609',
    'china': 'vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609'
}
class DockerAPIRunner:
    def __init__(self, use_china_mirror=True, silent=False):
        self.image = IMAGES['china'] if use_china_mirror else IMAGES['global']
        self.container = None
        self.silent = silent
        self.client = docker.from_env()
        self.port = self._find_free_port()
    
    def _find_free_port(self):
        """Find an available port dynamically"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start(self):
        """Start the Docker container using Docker API"""
        try:
            # Pull image if not exists
            if not self.silent:
                print(f"Pulling image: {self.image}")
            self.client.images.pull(self.image)
            
            # Run container
            self.container = self.client.containers.run(
                self.image,
                ports={'8080/tcp': self.port},
                detach=True,
                remove=True  # Auto-remove when stopped
            )
            
            if not self.silent:
                print(f"Container started: {self.container.short_id}")
            return True
            
        except DockerException as e:
            if not self.silent:
                print(f"Error starting container: {e}")
            return False
    
    def stop(self):
        """Stop the Docker container"""
        if self.container:
            try:
                self.container.stop()
                if not self.silent:
                    print("Container stopped")
                return True
            except DockerException as e:
                if not self.silent:
                    print(f"Error stopping container: {e}")
                return False
        return False
    
    def _wait_for_container_ready(self, max_wait_time: int = 60, check_interval: float = 1.0):
        """Wait for the Docker container to be ready"""
        if not self.container:
            raise Exception("Container not started")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            # Reload container status
            self.container.reload()
            
            if not self.silent:
                print(f"Container status: {self.container.status}")
            
            if self.container.status == 'running':
                # Container is running, now check if service is ready
                # First try a simple port connection test
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', self.port))
                    sock.close()
                    
                    if result == 0:  # Port is open
                        # Try to make a simple request to test the service
                        try:
                            response = requests.get(f'http://localhost:{self.port}/', timeout=2)
                            if not self.silent:
                                print(f"Service responded with status: {response.status_code}")
                            return True  # Service is responding
                        except requests.exceptions.RequestException:
                            # Try alternative endpoints or just accept that port is open
                            if not self.silent:
                                print(f"Port {self.port} is open, assuming service is ready")
                            return True
                except:
                    pass
            elif self.container.status in ['exited', 'dead']:
                # Get container logs for debugging
                logs = self.container.logs().decode('utf-8')
                raise Exception(f"Container failed to start. Status: {self.container.status}. Logs: {logs[:500]}")
            
            time.sleep(check_interval)
        
        # Get final container logs for debugging
        logs = self.container.logs().decode('utf-8') if self.container else "No container"
        raise Exception(f"Container not ready after {max_wait_time} seconds. Final status: {self.container.status if self.container else 'None'}. Logs: {logs[:500]}")


class SandboxfusionExecutor:
    def __init__(
        self,
        timeout_length: int = 10,
        ast_check: bool = False,
        max_workers: int = 1,
        use_china_mirror: bool = True,
    ) -> None:
        self.runner = DockerAPIRunner(use_china_mirror=use_china_mirror)
        running = self.runner.start()
        if not running:
            raise Exception("Failed to start Sandboxfusion Docker container")
        
        # Wait for the container to be ready
        self._wait_for_container_ready()
        set_endpoint(f'http://localhost:{self.runner.port}')
        
        self.timeout_length = timeout_length
        self.ast_check = ast_check
        self.max_workers = max_workers

    def _wait_for_container_ready(self, max_wait_time: int = 60, check_interval: float = 1.0):
        """Wait for the Docker container to be ready"""
        self.runner._wait_for_container_ready(max_wait_time, check_interval)

    def __del__(self):
        try:
            self.cleanup()
            self.runner.stop()
        except Exception as e:
            print(f"Error terminating pool: {e}")
            pass

    def cleanup(self):
        self.runner.stop()

    def process_generation_to_code(self, gens: str):
        return [g.strip().split('\n') for g in gens]
    
    def run_code(self, code: str, inputs: str, imports: List[str] = []) -> Tuple[str, str]:
        if isinstance(imports, np.ndarray):
            imports = imports.tolist()
        if imports:
            code = '\n'.join(imports) + '\n' + code
        code_snippet = RUN_CODE_TEMPLATE_REPR.format(code=code, inputs=inputs)
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
        code_snippet = VALIDATE_CODE_TEMPLATE_REPR.format(code=code, inputs=inputs)
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
        code_snippet = EVAL_INPUT_PREDICTION_TEMPLATE_REPR.format(code=code, gold_output=gold_output, agent_input=agent_input)
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
        code_snippet = EVAL_OUTPUT_PREDICTION_TEMPLATE_REPR.format(code=code, gold_output=gold_output, agent_output=agent_output)
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
        acc_list, status = self.apply(EVAL_K_INPUT_PREDICTION_TEMPLATE(code=code, gold_output=gold_output, k_agent_inputs=valid_k_agent_inputs, repr_output=True))
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
        acc_list, status = self.apply(EVAL_K_OUTPUT_PREDICTION_TEMPLATE(code=code, gold_output=gold_output, k_agent_outputs=valid_k_agent_outputs, repr_output=True))
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
            code_snippet = RUN_CODE_TEMPLATE_REPR.format(code=code, inputs=inputs)
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
                code_snippet = CHECK_DETERMINISM_TEMPLATE_REPR.format(code=code, inputs=inputs)
            else:
                code_snippet = RUN_CODE_TEMPLATE_REPR.format(code=code, inputs=inputs)
            if self.ast_check:
                try:
                    ast.parse(code_snippet)
                except:
                    return False, 'error'
            output, status = self.apply(code_snippet)
            return not 'error' in status.lower(), output

    def apply(self, code) -> Tuple[str, str]:
        try:
            response = run_code(
                RunCodeRequest(
                    code=code,
                    language='python',
                    compile_timeout=self.timeout_length,
                    run_timeout=self.timeout_length,
                )
            )
            if response.status == RunStatus.Success:
                # taking [1:-1] to exclude prefix space and suffix newline
                return response.run_result.stdout.split('<FINAL_REPR_SYMBOL>')[-1][1:-1], 'done'
            else:
                return '', 'error'

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            return error_msg, 'error'


def _test():
    batch_code = [
"""
def f(a):
    return a
print('<FINAL_REPR_SYMBOL>', repr(f(12eee)))
"""
    ]

    executor = SandboxfusionExecutor()
    predictions = executor.apply(batch_code[0])
    print(predictions)


if __name__ == '__main__':
    _test()
