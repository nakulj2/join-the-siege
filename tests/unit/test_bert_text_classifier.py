import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
from src.utils.extract_text import extract_text
from src.classifiers.bert_text import classify, LABELS


def test_bert_text_classify_runs_on_sample_input():
    """Basic smoke test: classify returns a result for minimal input."""
    text = "Sample text"
    label, probs = classify(text)
    assert isinstance(label, str)
    assert isinstance(probs, list)


def test_bert_text_classify_handles_empty_text():
    label, probs = classify("")
    assert label in LABELS + ["error", "model_not_loaded"]
    assert isinstance(probs, list)


def test_bert_text_classify_structure():
    """Ensure output is a known label and valid probability distribution."""
    label, probs = classify("Your total invoice is $134.12")
    assert label in LABELS, f"Unexpected label: {label}"
    assert isinstance(probs, list), "Probs not list"
    assert len(probs) == len(LABELS), f"Expected {len(LABELS)} probs, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 0.1, f"Probabilities should sum to ~1, got {sum(probs)}"


@pytest.mark.parametrize("filename,expected_label", [
    ("test_data/bank_statements/bank_statement_2.pdf", "bank_statement"),
    ("test_data/invoices/invoice_2.pdf", "invoice"),
    ("test_data/drivers_license/drivers_license_2.png", "drivers_license"),
])
def test_bert_text_classify_real_file(filename, expected_label):
    with open(filename, "rb") as f:
        f = BytesIO(f.read())
        f.filename = os.path.basename(filename)

        text = extract_text(f)
        assert text.strip(), f"No text extracted from {filename}"

        label, probs = classify(text)

        print(f"\n📄 File: {filename}\n🧠 Extracted Text: {text[:200]}...\n🔍 Label: {label}")
        assert label in LABELS
        assert isinstance(probs, list)
        assert len(probs) == len(LABELS)
        assert abs(sum(probs) - 1.0) < 0.1

        # Optional accuracy check:
        assert label == expected_label, f"Expected {expected_label}, got {label}"
