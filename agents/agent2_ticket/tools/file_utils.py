import shutil
from pathlib import Path
import docx
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

def save_uploaded_file(uploaded_file, dest_folder: Path) -> Path:
    """
    Saves an uploaded file to the destination folder.
    """
    file_path = dest_folder / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def move_to_processed(processed_dir, file_path: Path) -> Path:
    """
    Copies a file to the processed directory without removing it from its source.
    """
    dest = processed_dir / file_path.name
    shutil.copy2(file_path, dest)
    return dest

def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from the files
    """
    suffix = file_path.suffix.lower()
    text = ""
    
    if suffix in [".txt", ".md"]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".docx":
        doc = docx.Document(str(file_path))
        text = "\n".join([p.text for p in doc.paragraphs])
    elif suffix == ".pdf":
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    images = convert_from_path(str(file_path), first_page=i+1, last_page=i+1, dpi=300)
                    for img in images:
                        text += pytesseract.image_to_string(img, lang="eng+pol")
    return text.strip()

def save_processed_text(processed_dir: Path, original_file: Path, text: str) -> Path:
    """
    Save processed file as .txt into proccesed folder
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_file = processed_dir / f"{original_file.stem}.txt" 
    with open(processed_file, "w", encoding="utf-8") as f:
        f.write(text)
    return processed_file