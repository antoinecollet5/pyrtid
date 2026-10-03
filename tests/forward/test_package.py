"""Tests of the top level package (``pyrtid/__init__.py``)."""

import importlib
import sys

import pyrtid
import pytest


def test_package_metadata_and_exports() -> None:
    assert isinstance(pyrtid.__version__, str)
    for name in pyrtid.__all__:
        assert hasattr(pyrtid, name), name
    # a star import works (every name of __all__ exists)
    namespace: dict = {}
    exec("from pyrtid import *", namespace)  # noqa: S102
    assert namespace["Report"] is pyrtid.Report


def test_report_with_scooby() -> None:
    pytest.importorskip("scooby")
    report = pyrtid.Report(additional=None, ncol=2, text_width=70, sort=True)
    text = str(report)
    assert "pyrtid" in text
    assert "numpy" in text


def test_report_without_scooby_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """``scooby`` is a soft dependency: a warning is raised if it is missing."""
    monkeypatch.setitem(sys.modules, "scooby", None)  # makes the import fail
    try:
        importlib.reload(pyrtid)
        with pytest.warns(UserWarning, match="requires `scooby`"):
            pyrtid.Report()
    finally:
        monkeypatch.undo()
        importlib.reload(pyrtid)
    # the original class is restored
    assert pyrtid.Report.__mro__[1].__module__.startswith("scooby")
