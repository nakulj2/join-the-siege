from pdfplumber import open as pdf_open
from PIL import Image
import pytesseract
import tempfile
import sys

def extract_text(file):
    filename = getattr(file, "filename", "").lower()

    if filename.endswith(".pdf"):
        with pdf_open(file) as pdf:
            text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    # Fallback to OCR
                    im = page.to_image(resolution=300).original
                    text.append(pytesseract.image_to_string(im))
            return "\n".join(text)

    elif filename.endswith((".jpg", ".jpeg", ".png")):
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

if __name__ == "__main__":
    file_path = sys.argv[1]
    with open(file_path, "rb") as f:
        f.filename = file_path
        print(extract_text(f))
