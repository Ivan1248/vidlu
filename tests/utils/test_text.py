import pytest

import arpeggio

from vidlu.utils import text


def test_format_scanner_full_match():
    scanner = text.FormatScanner(r"duck(\d+).{a:conv|bn}1.float{:va|(\d+)}.zup{bee:0|1|(x*)}",
                                 full_match=True, debug=True)
    result = scanner("duck1.bn1.float22.zup0")
    assert result == dict(a='bn', bee='0')
    result = scanner("duck98.conv1.float22.zupxx")
    assert result == dict(a='conv', bee='xx')
    result = scanner("duck98.conv1.float22.zup")
    assert result == dict(a='conv', bee='')
    for invalid in ["duck1", "duck1.c1.float22.zup0", "duck1.conv1.float22.zupxxy"]:
        with pytest.raises(Exception):
            scanner(invalid)


def test_format_scanner_non_full_match():
    scanner = text.FormatScanner("onu{a:(.)}.{bee:(.+?)}.carrying(air|)", full_match=False,
                                 debug=True)
    result = scanner("coconut.laden.carrying.airspeed")
    assert result == dict(a='t', bee='laden')


def test_format_writer():
    writer = text.FormatWriter("{a:1->0}{a:1->0|2->1}.african.{`int(b)*2`}{`a+b`}{`a`}{b}.{see}ow",
                               debug=True)
    output = writer(a='2', b='3', see='swall', d='whatever')
    assert output == f"21.african.62323.swallow"
    for invalid in ["{int(b)*2}", r"{a:(\d+)->d|bla->s"]:
        with pytest.raises(arpeggio.NoMatch):
            text.FormatWriter(invalid, debug=True)


def test_format_translator():
    input_format = r"backbone.layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}{e}"
    output_format = "backbone.unit{`int(a)-1`}_{b}.{c:bn->norm}{`int(d)-1`}.orig{e}"
    translator = text.FormatTranslator(input_format, output_format)
    assert translator("backbone.layer4.0.bn1.bias") == "backbone.unit3_0.norm0.orig.bias"
    translator = text.FormatTranslator(input_format[:-1] + ":(.*)}", output_format)
    assert translator("backbone.layer4.0.bn1.bias") == "backbone.unit3_0.norm0.orig.bias"


def test_snake_case_pascal_case():
    # examples from https://stackoverflow.com/questions/1993721/how-to-convert-pascalcase-to-pascal-case
    examples = [
        ('simpleTest', 'simple_test', 'SimpleTest'),
        ('easy', 'easy', 'Easy'),
        ('HTML', 'html', 'Html'),
        ('simpleXML', 'simple_xml', 'SimpleXml'),
        ('PDFLoad', 'pdf_load', 'PdfLoad'),
        ('startMIDDLELast', 'start_middle_last', 'StartMiddleLast'),
        ('AString', 'a_string', 'AString'),
        ('Some4Numbers234', 'some4_numbers234', 'Some4Numbers234'),
        ('TEST123String', 'test123_string', 'Test123String'),
        ('ArraySpatial2D', 'array_spatial2d', 'ArraySpatial2d'),
        ('ArraySpatial2d', 'array_spatial2d', 'ArraySpatial2d'),
    ]
    for input, snake, pascal in examples:
        assert text.to_snake_case(input) == snake
        assert text.to_pascal_case(snake) == pascal
        assert text.to_pascal_case(input) == input[0].upper() + input[1:]
