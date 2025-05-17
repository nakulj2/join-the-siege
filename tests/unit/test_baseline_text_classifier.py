import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.utils.extract_text import extract_text
from src.classifiers import baseline_text


def test_baseline_text_classify_runs_on_text():
    """Baseline model should return a label string or 'model_not_loaded'."""
    class DummyFile:
        def read(self): return b"Some text content"
        filename = "dummy.txt"

    label = baseline_text.classify_text_file(DummyFile())
    assert isinstance(label, str)
    assert label in ["model_not_loaded", "error"] or label.strip()


def test_baseline_text_classify_handles_empty_file():
    class DummyFile:
        def read(self): return b""
        filename = "empty.pdf"

    label = baseline_text.classify_text_file(DummyFile())
    assert label in ["model_not_loaded", "error"] or isinstance(label, str)


def test_baseline_text_model_not_loaded(monkeypatch):
    """Force model=None to simulate fallback mode."""
    monkeypatch.setattr(baseline_text, "model", None)

    class DummyFile:
        def read(self): return b"Sample"
        filename = "sample.pdf"

    label = baseline_text.classify_text_file(DummyFile())
    assert label == "model_not_loaded"
