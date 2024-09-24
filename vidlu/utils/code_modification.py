import ast
import textwrap
import inspect
import functools
import copy
import types
import typing as T
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def extract_yield_separated_ast_statement_groups(func):
    """Extracts lists of AST nodes of statements separated by `yield` statements from a function."""
    statements = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0].body
    groups = []
    group = []
    for v in statements:
        if isinstance(v, ast.Expr) and isinstance(v.value, ast.Yield):
            groups.append(group)
            group = []
        else:
            group.append(v)
    groups.append(group)
    return groups


class TemplateNodeTransformer(ast.NodeTransformer):
    """Inserts code lines from a template function.

    The template functions be a tuple of 3 lists of AST nodes.
    - The first list is inserted into the beginning of the target function.
    - The middle list is modified and inserted after every line. Every function call in the list
    that has `_` as its first argument is modifed by replacing `_` with a tuple`(lineno, code)`,
    where `lineno: int` is the line number, and `code: str` is the content of the code line.
    - THe last list is inserted before every `return` statement and at the end of the function.
    """

    def __init__(self, template, func_nesting_depth=0, is_first_level=True):
        super().__init__()
        self.template = template
        self.func_nesting_depth = func_nesting_depth
        self.is_first_level = is_first_level

    def _process_stmt_list(self, stmt_list, template):
        start, inter, end = template
        result = copy.deepcopy(start)
        for v in stmt_list:
            if isinstance(v, ast.Return):
                result.extend(end)
                result.append(v)
            elif isinstance(v, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
                                ast.Delete)):
                result.append(v)
                new_nodes = copy.deepcopy(inter)
                arg = ast.Constant(value=(v.lineno, ast.unparse(v)))
                for n in new_nodes:
                    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                        n.value.args = [arg if isinstance(a, ast.Name) and a.id == '_' else a
                                        for a in n.value.args]
                result.extend(new_nodes)
            elif hasattr(v, 'body') and not isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # isinstance(v, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Match, ast.Try, ast.TryStar)):
                result.append(self._visit_bodied_node(v, ([], inter, [])))
            else:
                result.append(self.visit(v))
        return result

    def _visit_bodied_node(self, node, template):
        node.body = self._process_stmt_list(node.body, template)
        for name in ['orelse', 'finalbody']:
            if val := getattr(node, name, None):
                setattr(node, name, self._process_stmt_list(val, template))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.func_nesting_depth >= 0:
            tnt = TemplateNodeTransformer(self.template,
                                          func_nesting_depth=self.func_nesting_depth - 1,
                                          is_first_level=False)
            node = tnt._visit_bodied_node(node, self.template)
            if self.is_first_level:
                node.body.extend(copy.deepcopy(self.template[-1]))
        return node


def transform_func_code(func: types.FunctionType, template: T.Callable,
                        preprocess: T.Optional[T.Callable[[str], str]] = None,
                        additional_namespace: T.Mapping = None):
    if not isinstance(func, types.FunctionType):
        raise TypeError("The type of func is not FunctionType.")

    source: str = (preprocess or (lambda x: x))(inspect.getsource(func))

    ast_template = extract_yield_separated_ast_statement_groups(template)
    tnt = TemplateNodeTransformer(ast_template)

    old_ast = ast.parse(textwrap.dedent(source))
    new_ast = tnt.visit(copy.deepcopy(old_ast))
    new_ast.body[0].decorator_list.pop()
    new_ast = ast.fix_missing_locations(new_ast)
    ast.increment_lineno(new_ast, func.__code__.co_firstlineno - new_ast.body[0].lineno + 1)

    logger.info(ast.unparse(new_ast))

    code = compile(new_ast, filename=func.__code__.co_filename, mode='exec')
    namespace = {**func.__globals__, **(additional_namespace or dict())}
    new_func_code = next(c for c in code.co_consts
                         if isinstance(c, type(code)) and c.co_name == func.__name__)
    new_func = types.FunctionType(new_func_code, namespace, func.__name__,
                                  func.__defaults__, func.__closure__)

    return functools.wraps(func)(new_func)
