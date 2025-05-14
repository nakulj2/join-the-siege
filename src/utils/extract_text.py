from pdfplumber import open as pdf_open
from PIL import Image
import pytesseract
import tempfile

def extract_text(file):
    filename = getattr(file, "filename", "").lower()

    if filename.endswith(".pdf"):
        with pdf_open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif filename.endswith((".jpg", ".jpeg", ".png")):
        # Handle both Flask file and BytesIO
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            if hasattr(file, "save"):
                file.save(temp.name)
            else:
                with open(temp.name, "wb") as f:
                    f.write(file.read())
                    file.seek(0)
            image = Image.open(temp.name)
            return pytesseract.image_to_string(image)

    return ""
