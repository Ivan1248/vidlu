import arpeggio

import pytest

from vidlu.utils import text


def test_format_scanner():
    scanner = text.FormatScanner("duck(\d+).{a:conv|bn}1.blarp{:va|(\d+)}.zup{bee:0|1|(x*)}",
                                 debug=True)
    result = scanner("duck1.bn1.blarp22.zup0")
    assert result == dict(a='bn', bee='0')
    result = scanner("duck98.conv1.blarp22.zupxx")
    assert result == dict(a='conv', bee='xx')
    for invalid in ["duck1", "duck1.c1.blarp22.zup0", "duck1.bn1.blarp22.zup",
                    "duck1.conv1.blarp22.zupxxy"]:
        with pytest.raises(arpeggio.NoMatch):
            scanner(invalid)


def test_format_writer():
    writer = text.FormatWriter("{a:1->0}{a:1->0|2->1}.african.{`int(b)*2`}{`a+b`}{`a`}{b}.{see}ow",
                               debug=True)
    output = writer(a='2', b='3', see='swall', d='whatever')
    assert output == f"21.african.62323.swallow"
    for invalid in ["{int(b)*2}", "{a:(\d+)->d|bla->s"]:
        with pytest.raises(arpeggio.NoMatch):
            text.FormatWriter(invalid, debug=True)


def test_format_translator():
    translator = text.FormatTranslator(
        input_format="backbone.layer{a:(\d+)}.{b:(\d+)}.{c:conv|bn}{d:(\d+)}{e:(.*)}",
        output_format="backbone.unit{`int(a)-1`}_{b}.{c:bn->norm}{`int(d)-1`}.orig{e}")
    assert translator("backbone.layer4.0.bn1.bias") == "backbone.unit3_0.norm0.orig.bias"
