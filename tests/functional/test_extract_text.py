# Tests extract_text() for PDF and JPEG
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.app import app
from src.utils.extract_text import extract_text
from io import BytesIO

def test_pdf_text_extraction():
    with open("train_data/invoices/invoice_1.pdf", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "invoice_1.pdf"  # spoof filename attribute
        text = extract_text(file)
        assert len(text.strip()) > 0


def test_jpeg_text_extraction():
    with open("train_data/drivers_license/drivers_license_1.jpg", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "drivers_license_1.jpg"  # spoof filename attribute
        text = extract_text(file)
        assert len(text.strip()) > 0


def test_png_text_extraction():
    with open("train_data/bank_statements/bank_statement_5.png", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "bank_statement_5.png"  # spoof filename attribute
        text = extract_text(file)
        assert len(text.strip()) > 0
