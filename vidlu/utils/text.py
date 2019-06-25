from argparse import Namespace

import arpeggio
from arpeggio import (Optional, ZeroOrMore, OneOrMore, EOF, ParserPython, PTNodeVisitor,
                      visit_parse_tree)
from arpeggio import RegExMatch as R


# @formatter:off
# [...] is disjunction, (...) is conjunction
def const(): return R(r"[\w\.]*")
def re_pattern(): return R(r"\([^\)]*\)")
def pattern(): return [const, re_pattern]
def or_pattern(): return pattern, ZeroOrMore('|', pattern)
def ident(): return R(r"\w+")

def var_pattern(): return ':', or_pattern
def input_var(): return ident, Optional(var_pattern)
def scanner_expr(): return OneOrMore([pattern, ('{', [input_var, var_pattern], '}')]), EOF()

def dict_translation(): return const, '->', const
def dict_multi_translation(): return dict_translation, ZeroOrMore('|', dict_translation)
def dict_translator(): return ident, Optional(':', dict_multi_translation)
def py_translator(): return R('`[^`]+`')
def translator(): return [dict_translator, py_translator]
def writer_expr(): return OneOrMore([const, ('{', translator, '}')]), EOF()
# @formatter:on


class _ScannerWriterVisitorBase(PTNodeVisitor):
    def visit_const(self, node, children):
        if self.debug: print(f"const: {node.value}")
        return node.value

    def visit_re_pattern(self, node, children):
        if self.debug: print(f"re_pattern: {node.value}")
        return R(node.value[1:-1])

    def visit_pattern(self, node, children):
        if self.debug: print(f"re_pattern: {node.value}")
        return tuple(children) if len(children) > 1 else children[0]

    def visit_or_pattern(self, node, children):
        if self.debug: print(f"or_pattern: {children}")
        return list(children)

    def visit_ident(self, node, children):
        if self.debug: print(f"ident: {node.value}")
        return node.value


class _ScannerExprVisitor(_ScannerWriterVisitorBase):
    def visit_var_pattern(self, node, children):
        if self.debug: print(f"var_pattern: {children}")
        return children[0]

    def visit_input_var(self, node, children):
        if self.debug: print(f"input_var: {children}")
        var_name = children[0]
        pattern = R(".") if len(children) == 1 else children[1]
        return Namespace(var_name=var_name, pattern=pattern)

    def visit_scanner_expr(self, node, children):
        if self.debug: print("input_expr {}".format(children))
        pattern = tuple()
        var_to_index = dict()
        for c in children:
            if isinstance(c, Namespace):
                n = len(pattern)
                var_to_index[c.var_name] = n
                c = c.pattern
            pattern += (c,)
        return pattern, var_to_index


class _WriterExprVisitor(_ScannerWriterVisitorBase):
    def __init__(self, vars_, **kwargs):
        super().__init__(**kwargs)
        self.vars = vars_
        self.vars_str = ','.join(f"{k}='{v}'" for k, v in vars_.items())

    def visit_dict_translation(self, node, children):
        if self.debug: print(f"dict_translation: {children}")
        return {children[0]: children[1]}

    def visit_dict_multi_translation(self, node, children):
        if self.debug: print(f"dict_multi_translation: {children}")
        return {k: v for d in children for k, v in reversed(list(d.items()))}

    def visit_dict_translator(self, node, children):
        if self.debug: print(f"dict_translator: {children}")
        var_name = children[0]
        translation_dict = children[1] if len(children) > 1 else dict()
        key = self.vars[var_name]
        return translation_dict.get(key, key)

    def visit_py_translator(self, node, children):
        if self.debug: print(f"py_translator: {node}")
        return eval(f"str((lambda {self.vars_str}: {node.value[1:-1]})())")

    def translator(self, node, children):
        if self.debug: print(f"translator: {node}")
        return children[0]

    def visit_writer_expr(self, node, children):
        if self.debug: print(f"writer_expr {children}")
        return list(children)


class FormatScanner:
    def __init__(self, format, debug=False):
        format_parser = arpeggio.ParserPython(scanner_expr)
        format_parse_tree = format_parser.parse(format)
        input_pattern, self.var_to_index = visit_parse_tree(format_parse_tree,
                                                            _ScannerExprVisitor(debug=debug))
        input_pattern += (EOF(),)
        self.input_parser = arpeggio.ParserPython(lambda: input_pattern)

    def __call__(self, input):
        parsed = self.input_parser.parse(input)
        return {k: parsed[ind].value for k, ind in self.var_to_index.items()}

    def try_scan(self, input):
        try:
            return self(input)
        except arpeggio.NoMatch as ex:
            return None


class FormatWriter:
    def __init__(self, format, debug=False):
        self.debug = debug
        format_parser = arpeggio.ParserPython(writer_expr)
        self.format_parse_tree = format_parser.parse(format)

    def __call__(self, **vars_):
        parts = visit_parse_tree(self.format_parse_tree,
                                 _WriterExprVisitor(vars_, debug=self.debug))
        return ''.join(parts)


class FormatTranslator:
    def __init__(self, input_format, output_format, debug=False):
        self.scanner = FormatScanner(input_format)
        self.writer = FormatWriter(output_format)

    def __call__(self, input, **additional_vars):
        vars_ = self.scanner(input)
        return self.writer(**vars_, **additional_vars)

    def try_translate(self, input, **additional_vars):
        try:
            return self(input, **additional_vars)
        except arpeggio.NoMatch as ex:
            return None
