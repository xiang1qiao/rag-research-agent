import os
from pypdf import PdfReader
from docx import Document


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    text_list = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_list.append(paragraph.text)

    return "\n".join(text_list)


def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)

    if ext == ".pdf":
        return load_pdf(file_path)

    if ext == ".docx":
        return load_docx(file_path)

    raise ValueError("暂不支持该文件格式，只支持 txt、pdf、docx")