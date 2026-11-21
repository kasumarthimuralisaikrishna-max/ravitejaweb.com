"""Secure PDF validation and text extraction for RESUMATE AI."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import ALLOWED_EXTENSIONS, MAX_PDF_SIZE_BYTES, MAX_PDF_SIZE_MB, MIN_RESUME_CHARS


class PDFExtractionError(Exception):
    """User-friendly error raised when PDF validation or extraction fails."""


@dataclass(frozen=True)
class PDFExtractionResult:
    """Validated metadata and extracted text from an uploaded resume PDF."""

    filename: str
    file_size_bytes: int
    page_count: int
    word_count: int
    char_count: int
    text: str


def _clean_extracted_text(raw_text: str) -> str:
    """Normalize whitespace and strip non-printable control characters."""
    # Remove null bytes and non-printable control chars except newlines/tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", raw_text)
    # Collapse excessive horizontal spaces while preserving paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _page_has_images(page) -> bool:
    """Safely check whether a pypdf page contains embedded images."""
    try:
        if getattr(page, "images", None) and len(page.images) > 0:
            return True
    except Exception:
        pass
    try:
        resources = page.get("/Resources")
        if resources and "/XObject" in resources:
            xobjects = resources["/XObject"].get_object()
            for obj_name in xobjects:
                obj = xobjects[obj_name].get_object()
                if obj.get("/Subtype") == "/Image":
                    return True
    except Exception:
        pass
    return False


def extract_resume_from_pdf(filename: str, file_bytes: bytes) -> PDFExtractionResult:
    """
    Validate an uploaded PDF file and extract its text safely using pypdf.

    Never executes PDF actions/JavaScript and never writes uploaded PDFs to disk.
    """
    safe_name = Path(filename or "resume.pdf").name
    ext = Path(safe_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise PDFExtractionError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a valid PDF (.pdf) file."
        )

    if not file_bytes or len(file_bytes) == 0:
        raise PDFExtractionError(
            "The uploaded file is empty (0 bytes). Please upload a valid PDF resume."
        )

    file_size = len(file_bytes)
    if file_size > MAX_PDF_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise PDFExtractionError(
            f"File size ({size_mb:.2f} MB) exceeds the {MAX_PDF_SIZE_MB} MB limit. "
            "Please compress or upload a smaller PDF."
        )

    # Check PDF magic signature within first 1024 bytes (standard PDF allowance)
    if b"%PDF-" not in file_bytes[:1024]:
        raise PDFExtractionError(
            "Invalid PDF file content. The file does not appear to be a genuine PDF document."
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes), strict=False)
    except PdfReadError:
        raise PDFExtractionError(
            "Unable to read this PDF because it appears corrupted or malformed. "
            "Please re-export your resume as a fresh PDF and try again."
        )
    except Exception:
        raise PDFExtractionError(
            "Failed to parse the uploaded PDF file. Please verify the file is not damaged."
        )

    if reader.is_encrypted:
        raise PDFExtractionError(
            "This PDF is password-protected or encrypted. Please remove the password "
            "protection and upload an unlocked PDF."
        )

    page_count = len(reader.pages)
    if page_count == 0:
        raise PDFExtractionError(
            "The uploaded PDF contains 0 pages. Please upload a valid resume PDF."
        )

    extracted_pages: list[str] = []
    has_embedded_images = False

    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)
        except Exception:
            # Continue extracting remaining pages if a single page has encoding issues
            pass

        if not has_embedded_images and _page_has_images(page):
            has_embedded_images = True

    combined_text = _clean_extracted_text("\n\n".join(extracted_pages))
    alnum_count = len(re.findall(r"[A-Za-z0-9]", combined_text))

    if not combined_text or alnum_count < 20 or len(combined_text) < MIN_RESUME_CHARS:
        if has_embedded_images or (page_count > 0 and alnum_count < 20):
            raise PDFExtractionError(
                "No selectable text could be extracted from this PDF. It appears to be a "
                "scanned document or image-only PDF. Optical Character Recognition (OCR) is "
                "not supported—please export a text-based PDF from Word, Google Docs, or LaTeX."
            )
        raise PDFExtractionError(
            "The PDF appears to be empty or lacks readable resume text. Please check the file."
        )

    word_count = len(combined_text.split())
    return PDFExtractionResult(
        filename=safe_name,
        file_size_bytes=file_size,
        page_count=page_count,
        word_count=word_count,
        char_count=len(combined_text),
        text=combined_text,
    )
