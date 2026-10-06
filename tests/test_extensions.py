import sys

import pytest

from vidlu import extensions as vext


@pytest.fixture
def test_extension(tmp_path, monkeypatch):
    """A `vidlu_testext` package with a `sub` module, registered as the extension `testext`."""
    package_dir = tmp_path / "vidlu_testext"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "sub.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(vext.extensions, "testext", vext.LazyModule("vidlu_testext"))
    yield
    for name in ("vidlu_testext", "vidlu_testext.sub"):
        sys.modules.pop(name, None)


def test_import_extension_module_resolves_extension_names(test_extension):
    sub = vext.import_extension_module("testext.sub")
    assert sub is sys.modules["vidlu_testext.sub"]
    assert sub.value == 1
    assert vext.import_extension_module("testext") is sys.modules["vidlu_testext"]


def test_import_extension_module_keeps_other_names(test_extension):
    assert vext.import_extension_module("vidlu_testext.sub") is sys.modules["vidlu_testext.sub"]
    assert vext.import_extension_module("json") is sys.modules["json"]
