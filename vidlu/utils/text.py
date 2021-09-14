from argparse import Namespace
import re

import arpeggio
from arpeggio import (Optional, ZeroOrMore, OneOrMore, EOF, PTNodeVisitor,
                      visit_parse_tree)
from arpeggio import RegExMatch as R
import vidlu.utils.func as func


# @formatter:off
# [...] is disjunction, (...) is conjunction
def const():
    return R(r"[\w\.]*")
def ident():
    return R(r"\w+")

def re_pattern():
    return R(r"\([^\)]*\)")
def pattern():
    return [const, re_pattern]
def or_pattern():
    return pattern, ZeroOrMore('|', pattern)
def var_pattern():
    return ':', or_pattern
def input_var():
    return ident, Optional(var_pattern)
def scanner_expr():
    return OneOrMore([pattern, ('{', [input_var, var_pattern], '}')]), EOF()

def dict_translation():
    return const, '->', const
def dict_multi_translation():
    return dict_translation, ZeroOrMore('|', dict_translation)
def dict_translator():
    return ident, Optional(':', dict_multi_translation)
def py_translator():
    return R('`[^`]+`')
def translator():
    return [dict_translator, py_translator]
def writer_expr():
    return OneOrMore([const, ('{', translator, '}')]), EOF()
# @formatter:on


class _ScannerWriterVisitorBase(PTNodeVisitor):
    def visit_const(self, node, children):
        if self.debug:
            print(f"const: {node.value}")
        return node.value

    def visit_ident(self, node, children):
        if self.debug:
            print(f"ident: {node.value}")
        return node.value


class _ScannerExprVisitor(_ScannerWriterVisitorBase):
    def visit_const(self, node, children):
        if self.debug:
            print(f"const: {node.value}")
        return re.escape(node.value)

    def visit_ident(self, node, children):
        if self.debug:
            print(f"ident: {node.value}")
        return re.escape(node.value)

    def visit_re_pattern(self, node, children):
        if self.debug:
            print(f"re_pattern: {node.value}")
        return '(?:' + node.value[1:-1] + ')'

    def visit_pattern(self, node, children):
        if self.debug:
            print(f"re_pattern: {node.value}")
        return ''.join(children) if len(children) > 1 else children[0]

    def visit_or_pattern(self, node, children):
        if self.debug:
            print(f"or_pattern: {children}")
        return '(?:' + '|'.join(children) + ')'

    def visit_var_pattern(self, node, children):
        if self.debug:
            print(f"var_pattern: {children}")
        return children[0]

    def visit_input_var(self, node, children):
        if self.debug:
            print(f"input_var: {children}")
        return Namespace(var_name=children[0], pattern=".*?" if len(children) == 1 else children[1])

    def visit_scanner_expr(self, node, children):
        if self.debug:
            print("input_expr {}".format(children))
        pattern = []
        vars_ = []
        for c in children:
            if isinstance(c, Namespace):
                vars_.append(c.var_name)
                c = f"({c.pattern})"
            pattern.append(c)
        return ''.join(pattern), vars_


class _WriterExprVisitor(_ScannerWriterVisitorBase):
    def __init__(self, vars_, context=None, **kwargs):
        super().__init__(**kwargs)
        self.vars = vars_
        self.vars_str = ', '.join(f"{k}='{v}'" for k, v in vars_.items())
        self.context = context or dict()

    def visit_dict_translation(self, node, children):
        if self.debug:
            print(f"dict_translation: {children}")
        return {children[0]: children[1]}

    def visit_dict_multi_translation(self, node, children):
        if self.debug:
            print(f"dict_multi_translation: {children}")
        return {k: v for d in children for k, v in reversed(list(d.items()))}

    def visit_dict_translator(self, node, children):
        if self.debug:
            print(f"dict_translator: {children}")
        var_name = children[0]
        translation_dict = children[1] if len(children) > 1 else dict()
        key = self.vars[var_name]
        return translation_dict.get(key, key)

    def visit_py_translator(self, node, children):
        if self.debug:
            print(f"py_translator: {node}")
        try:
            return eval(f"str((lambda {self.vars_str}: {node.value[1:-1]})())", None, self.context)
        except NameError as e:
            raise NameError(f'{e}. Cannot evaluate "{node.value[1:-1]}".'
                            + f' Available variables: {self.vars_str}.')

    def translator(self, node, children):
        if self.debug:
            print(f"translator: {node}")
        return children[0]

    def visit_writer_expr(self, node, children):
        if self.debug:
            print(f"writer_expr {children}")
        return list(children)


class NoMatchError(Exception):
    def __init__(self, message, inner_exception=None):
        super().__init__(message)
        self.inner_exception = inner_exception

    def __str__(self):
        if self.inner_exception is not None:
            return super().__str__() + ":\n" + str(self.inner_exception)
        else:
            return super().__str__()


class FormatScanner:
    def __init__(self, fmt, full_match, debug=False):
        self.format = fmt
        format_parser = arpeggio.ParserPython(scanner_expr)
        try:
            format_parse_tree = format_parser.parse(fmt)
        except arpeggio.NoMatch as e:
            raise NoMatchError("Invalid format.") from e
        input_pattern, self.var_names = visit_parse_tree(format_parse_tree,
                                                         _ScannerExprVisitor(debug=debug))
        self.full_match = full_match

        input_pattern = input_pattern + "$" if full_match else ".*?" + input_pattern
        self.regex = re.compile(input_pattern)

    def __call__(self, x):
        parsed = (self.regex.fullmatch if self.full_match else self.regex.match)(x)
        if parsed is None:
            raise NoMatchError(f'The input string\n  "{x}"\n does not match the format\n'
                               + f'  "{self.format}"\n  (regex: {self.regex.pattern})')
        var_values = parsed.groups()
        return dict(zip(self.var_names, var_values))

    def try_scan(self, x):
        try:
            return self(x)
        except arpeggio.NoMatch:
            return None


def scan(format, x, *, full_match):
    return FormatScanner(format, full_match=full_match)(x)


class FormatWriter:
    def __init__(self, format, debug=False, context=None):
        self.debug = debug
        try:
            format_parser = arpeggio.ParserPython(writer_expr)
        except arpeggio.NoMatch as e:
            raise NoMatchError("Invalid format.") from e
        self.format_parse_tree = format_parser.parse(format)
        self.context = context

    def __call__(self, **vars_):
        parts = visit_parse_tree(self.format_parse_tree,
                                 _WriterExprVisitor(vars_, debug=self.debug, context=self.context))
        return ''.join(parts)


class FormatTranslator:
    def __init__(self, input_format, output_format, full_match=True, error_on_no_match=False,
                 context=None):
        self.scanner = FormatScanner(input_format, full_match=full_match)
        self.writer = FormatWriter(output_format, context=context)
        self.error_on_no_match = error_on_no_match

    def __call__(self, x, **additional_vars):
        try:
            vars_ = self.scanner(x)
            return self.writer(**vars_, **additional_vars)
        except NoMatchError as e:
            if self.error_on_no_match:
                raise
            return None


class FormatTranslatorCascade:
    def __init__(self, input_output_format_pairs, error_on_no_match=True, context=None):
        self.input_output_format_pairs = tuple(input_output_format_pairs)
        self.translators = [
            FormatTranslator(inp, out, full_match=True, error_on_no_match=error_on_no_match,
                             context=context)
            for inp, out in input_output_format_pairs]
        self.error_on_no_match = error_on_no_match

    def __call__(self, x):
        for t in self.translators:
            output = func.tryable(t, None, NoMatchError)(x)
            if output is not None:
                return output
        if not self.error_on_no_match:
            return x
        messages = []
        for (inp, _), t in zip(self.input_output_format_pairs, self.translators):
            try:
                t(x)
            except NoMatchError as e:
                messages.append((inp, str(e)))
        if self.error_on_no_match:
            messages = '\n'.join(
                f"  {i}. {inp} -> {err}\n" for i, (inp, err) in enumerate(messages))
            raise NoMatchError(f'Input "{x}" matches no input format.\n{messages}')


def to_snake_case(identifier):
    identifier = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', identifier)
    identifier = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', identifier)
    return identifier.lower()


def to_pascal_case(identifier):
    return ''.join(x.title() for x in identifier.split('_'))


def common_prefix(strings):
    for i, (c, *others) in enumerate(zip(*strings)):
        if any(d != c for d in others):
            return strings[0][:i]
    return strings[0]
