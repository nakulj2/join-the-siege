import sys
import os
from io import BytesIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.extract_text import extract_text

def test_extract_text_from_pdf():
    with open("test_data/invoices/invoice_2.pdf", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "invoice_1.pdf"
        text = extract_text(file)
        assert isinstance(text, str)
        assert len(text.strip()) > 0


def test_extract_text_from_png():
    with open("test_data/drivers_license/drivers_license_2.png", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "drivers_license_2.png"
        text = extract_text(file)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
