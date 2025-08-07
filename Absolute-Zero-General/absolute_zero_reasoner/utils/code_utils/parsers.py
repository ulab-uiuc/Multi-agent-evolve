import ast
import re
from typing import List


def parse_imports(code_snippet: str) -> List[str]:
    imports = []
    try:
        tree = ast.parse(code_snippet)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Reconstruct import line from AST node
                if isinstance(node, ast.Import):
                    import_line = "import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                else:
                    module = node.module or ""
                    import_line = f"from {module} import " + ", ".join(
                        [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                            for alias in node.names]
                    )
                    if node.level > 0:
                        import_line = f"from {'.' * node.level}{module} import " + ", ".join(
                            [alias.name + (f" as {alias.asname}" if alias.asname else "") 
                                for alias in node.names]
                        )
                imports.append(import_line)
    except Exception as e:
        import_pattern = r"^\s*(?:from|import)\s+.*$"
        imports = [i.strip() for i in re.findall(import_pattern, code_snippet, re.MULTILINE)]
    return imports


def parse_error(error_message: str) -> str:
    # split by colon
    error_message = error_message.split(':')[0]
    return error_message.strip()


def replace_main_function_name(code: str, old_name: str, new_name: str) -> str:
    """
    Replace all occurrences of `old_name` with `new_name` in the code.
    Replace the definition and all recursive calls of `old_name` with `new_name`.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == old_name:
            node.name = new_name
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == old_name:
            node.func.id = new_name
    return ast.unparse(tree)


def remove_comments_and_docstrings(code: str) -> str:
    """
    Remove all comments and docstrings from the code.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef, ast.Module)):
                # Remove all leading docstrings
                while node.body and isinstance(node.body[0], ast.Expr):
                    expr = node.body[0].value
                    if isinstance(expr, (ast.Str, ast.Constant)) and (
                        isinstance(expr.value, str) if isinstance(expr, ast.Constant) else True
                    ):
                        node.body.pop(0)
                    else:
                        break
        
        # Convert back to code - AST unparse already removes comments
        code_without_docstrings = ast.unparse(tree)
        
        # Only remove empty lines and trim whitespace
        lines = [
            line.rstrip()
            for line in code_without_docstrings.split('\n')
            if line.strip()
        ]
        
        return '\n'.join(lines)
    except Exception as e:
        return code  # Return original code if parsing fails


def remove_any_not_definition_imports(code: str) -> str:
    """
    Remove anything that is not a definition or import.
    Preserves: 
    - Import/From imports
    - Class definitions
    - Function/AsyncFunction definitions
    Removes:
    - Top-level assignments
    - Standalone expressions
    - Constant declarations
    """
    class DefinitionFilter(ast.NodeTransformer):
        def visit_Module(self, node):
            # Keep only definitions and imports (explicitly exclude assignments)
            node.body = [
                n for n in node.body
                if isinstance(n, (
                    ast.Import,
                    ast.ImportFrom,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef
                ))
            ]
            return node

    try:
        tree = ast.parse(code)
        tree = DefinitionFilter().visit(tree)
        ast.fix_missing_locations(tree)

        # Remove empty lines and format
        cleaned = ast.unparse(tree)
        return '\n'.join([line for line in cleaned.split('\n') if line.strip()])

    except Exception as e:
        return code


class PrintRemover(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Handle top-level print statements
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None
        return node

    def visit_Call(self, node):
        # Handle print calls in other contexts (like assignments)
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            return ast.Constant(value=None)
        return node

    def _handle_block(self, node):
        self.generic_visit(node)
        if not node.body:
            node.body.append(ast.Pass())
        return node

    def visit_For(self, node):
        return self._handle_block(node)

    def visit_While(self, node):
        return self._handle_block(node)

    def visit_FunctionDef(self, node):
        return self._handle_block(node)

    def visit_AsyncFunctionDef(self, node):
        return self._handle_block(node)

    def visit_If(self, node):
        return self._handle_block(node)

    def visit_With(self, node):
        return self._handle_block(node)

    def visit_Try(self, node):
        self.generic_visit(node)
        
        # Handle main try body
        if not node.body:
            node.body.append(ast.Pass())
            
        # Handle except handlers
        for handler in node.handlers:
            if not handler.body:
                handler.body.append(ast.Pass())
                
        # Handle else clause
        if node.orelse and not node.orelse:
            node.orelse.append(ast.Pass())
            
        # Handle finally clause
        if node.finalbody and not node.finalbody:
            node.finalbody.append(ast.Pass())
            
        return node


def remove_print_statements(code: str) -> str:
    """
    Remove all print statements from the code.
    """
    tree = ast.parse(code)
    tree = PrintRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


if __name__ == "__main__":
    print(parse_error("NameError: name 'x' is not defined"))
    print(parse_error("TypeError: unsupported operand type(s) for -: 'str' and 'str'"))
    print(parse_error("ValueError: invalid literal for int() with base 10: 'x'"))
