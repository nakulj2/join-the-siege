from src.utils.extract_text import extract_text
from io import BytesIO

def test_pdf_text_extraction():
    with open("data/invoices/invoice_1.pdf", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "invoice_1.pdf"  # spoof filename attribute
        text = extract_text(file)
        assert len(text.strip()) > 0


def test_jpeg_text_extraction():
    with open("data/drivers_license/drivers_license_1.jpg", "rb") as f:
        file = BytesIO(f.read())
        file.filename = "drivers_license_1.jpg"  # spoof filename attribute
        text = extract_text(file)
        assert len(text.strip()) > 0

