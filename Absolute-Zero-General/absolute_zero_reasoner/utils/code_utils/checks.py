import hashlib
import ast
import re
from typing import List


def check_determinism(code: str, inputs: str, executor, prev_output: str = None, n_runs: int = 1):
    """expects an executor that outputs string output and status"""
    all_outputs = set()
    if prev_output is not None:
        hash = hashlib.md5(str(prev_output).encode()).hexdigest()
        all_outputs.add(hash)
    for _ in range(n_runs):
        result = executor.run_code(code, inputs)[0]
        hash = hashlib.md5(str(result).encode()).hexdigest()
        all_outputs.add(hash)
    return len(all_outputs) == 1


def contains_banned_imports(code: str, banned_keywords: List[str], banned_keywords_for_errors_and_exceptions: List[str] = []) -> bool:
    """Check if code imports any banned modules using AST parsing."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(banned in alias.name.split('.') for banned in banned_keywords):
                        return True
            elif isinstance(node, ast.ImportFrom):
                module = node.module.split('.') if node.module else []
                if any(banned in module for banned in banned_keywords):
                    return True
                for alias in node.names:
                    if any(banned in alias.name.split('.') for banned in banned_keywords):
                        return True

            if banned_keywords_for_errors_and_exceptions:
                # Check for assert statements
                if isinstance(node, ast.Assert) and 'assert' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for raise statements
                elif isinstance(node, ast.Raise) and 'raise' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for try-except blocks
                elif isinstance(node, ast.Try) and 'try' in banned_keywords_for_errors_and_exceptions:
                    return True

                # Check for except handlers
                elif isinstance(node, ast.ExceptHandler) and 'except' in banned_keywords_for_errors_and_exceptions:
                    return True

        return False
    except SyntaxError:
        # Fallback to simple check if AST parsing fails
        return any(re.search(rf'\b{re.escape(banned)}\b', code) for banned in banned_keywords)


def check_no_definitions(code: str, composite_functions: List[str]) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in composite_functions:
            return False
    return True


def check_composite_function(code: str, composite_functions: List[str]) -> bool:
    composite_function_names = [f"g_{i}" for i in range(len(composite_functions))]

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    
    f_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'f':
            f_def = node
            break
    if f_def is None:
        return False
    
    parameters = {arg.arg for arg in f_def.args.args}
    
    assigned_vars_visitor = AssignedVarsVisitor()
    for stmt in f_def.body:
        assigned_vars_visitor.visit(stmt)
    scope_vars = parameters | assigned_vars_visitor.assigned
    
    call_checker = CallChecker(composite_function_names, scope_vars)
    for stmt in f_def.body:
        call_checker.visit(stmt)
    
    result = call_checker.called == set(composite_function_names) and call_checker.valid
    return result


class AssignedVarsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.assigned = set()
    
    def visit_Assign(self, node):
        for target in node.targets:
            self.collect_names(target)
        self.generic_visit(node)
    
    def collect_names(self, node):
        if isinstance(node, ast.Name):
            self.assigned.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self.collect_names(elt)


class CallChecker(ast.NodeVisitor):
    def __init__(self, composite_functions, scope_vars):
        self.composite_functions = composite_functions
        self.scope_vars = scope_vars
        self.called = set()
        self.valid = True
        self.local_scopes = [{}]
    
    def visit_FunctionDef(self, node):
        self.local_scopes.append({arg.arg: None for arg in node.args.args})
        self.generic_visit(node)
        self.local_scopes.pop()
    
    def visit_ListComp(self, node):
        comp_scope = {}
        for gen in node.generators:
            if isinstance(gen.iter, ast.Name) and gen.iter.id in self.scope_vars:
                self.collect_names(gen.target, comp_scope)
        self.local_scopes.append(comp_scope)
        self.visit(node.elt)
        for gen in node.generators:
            for comp_if in gen.ifs:
                self.visit(comp_if)
        self.local_scopes.pop()
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.composite_functions:
                func_name = node.func.id
                self.called.add(func_name)
                current_scope = self.build_current_scope()
                for arg in node.args:
                    names = self.get_names(arg)
                    if not all(name in current_scope for name in names):
                        self.valid = False
            elif node.func.id in {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}:
                for parent in ast.walk(node):
                    if isinstance(parent, ast.FunctionDef) and parent.name == node.func.id:
                        for param, arg in zip(parent.args.args, node.args):
                            if isinstance(arg, ast.Name):
                                self.local_scopes[-1][param.arg] = arg.id
        self.generic_visit(node)
    
    def build_current_scope(self):
        scope = set(self.scope_vars)
        for local_scope in self.local_scopes:
            scope.update(local_scope.keys())
            for mapped_var in local_scope.values():
                if mapped_var:
                    scope.add(mapped_var)
        return scope
    
    def collect_names(self, node, scope_dict):
        if isinstance(node, ast.Name):
            scope_dict[node.id] = None
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self.collect_names(elt, scope_dict)
    
    def get_names(self, node):
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) 
                and isinstance(n.ctx, ast.Load) 
                and n.id not in self.composite_functions]
