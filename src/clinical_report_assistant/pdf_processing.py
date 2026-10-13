"""Safe, in-memory validation and text extraction for uploaded PDF files."""

from __future__ import annotations

from io import BytesIO

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

DEFAULT_MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
_PDF_HEADER_SEARCH_BYTES = 1024


class PdfIngestionError(ValueError):
    """Base class for errors that are safe to display to an end user."""


class EmptyPdfError(PdfIngestionError):
    """Raised when an upload contains no bytes."""


class PdfTooLargeError(PdfIngestionError):
    """Raised when an upload exceeds the configured size limit."""


class NonPdfError(PdfIngestionError):
    """Raised when uploaded bytes do not contain a PDF header."""


class CorruptPdfError(PdfIngestionError):
    """Raised when pypdf cannot parse the uploaded document."""


class EncryptedPdfError(PdfIngestionError):
    """Raised when the uploaded PDF is encrypted."""


class PdfExtractionError(PdfIngestionError):
    """Raised when a readable page cannot be processed safely."""


class PdfPage(BaseModel):
    """Extracted text and warnings for one one-based PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str
    warnings: list[str] = Field(default_factory=list)


class PdfDocument(BaseModel):
    """Validated PDF metadata and page-by-page extraction results."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1)
    page_count: int = Field(ge=1)
    pages: list[PdfPage]
    warnings: list[str] = Field(default_factory=list)


def process_pdf(
    uploaded_bytes: bytes,
    original_filename: str,
    *,
    max_size_bytes: int = DEFAULT_MAX_PDF_SIZE_BYTES,
) -> PdfDocument:
    '''Validate and extract an uploaded PDF entirely in memory.'''

    if max_size_bytes <= 0:
        raise ValueError('max_size_bytes must be greater than zero.')
    if not isinstance(uploaded_bytes, bytes):
        raise TypeError('uploaded_bytes must be bytes.')
    if not original_filename or not original_filename.strip():
        raise PdfIngestionError('The uploaded file must have a filename.')
    if not uploaded_bytes:
        raise EmptyPdfError('The uploaded file is empty.')
    if len(uploaded_bytes) > max_size_bytes:
        limit_mib = max_size_bytes / (1024 * 1024)
        raise PdfTooLargeError(
            f'The uploaded file exceeds the {limit_mib:g} MiB size limit.'
        )
    if b'%PDF-' not in uploaded_bytes[:_PDF_HEADER_SEARCH_BYTES]:
        raise NonPdfError('The uploaded content is not a PDF file.')

    try:
        reader = PdfReader(BytesIO(uploaded_bytes), strict=False)
        if reader.is_encrypted:
            raise EncryptedPdfError(
                'Encrypted PDFs are not supported. Upload an unencrypted copy.'
            )
        page_count = len(reader.pages)
    except EncryptedPdfError:
        raise
    except (PdfReadError, EOFError, ValueError, TypeError) as exc:
        raise CorruptPdfError(
            'The PDF is malformed or corrupt and could not be read.'
        ) from exc

    if page_count == 0:
        raise CorruptPdfError('The PDF contains no pages.')

    pages: list[PdfPage] = []
    document_warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ''
        except Exception as exc:
            raise PdfExtractionError(
                f'Text extraction failed on page {page_number}; the PDF may be corrupt.'
            ) from exc

        page_warnings: list[str] = []
        if not text.strip():
            warning = (
                f'Page {page_number} contains no extractable text and may require OCR.'
            )
            page_warnings.append(warning)
            document_warnings.append(warning)
        pages.append(
            PdfPage(page_number=page_number, text=text, warnings=page_warnings)
        )

    if all(not page.text.strip() for page in pages):
        document_warnings.insert(
            0,
            'This PDF contains no extractable text. It may be scanned or image-only '
            'and require OCR; OCR is not included in this demo.',
        )

    return PdfDocument(
        filename=original_filename,
        page_count=page_count,
        pages=pages,
        warnings=document_warnings,
    )
