'''Tests for safe, offline PDF validation and extraction.'''

from __future__ import annotations

import pytest
from conftest import make_text_pdf

from clinical_report_assistant.pdf_processing import (
    CorruptPdfError,
    EmptyPdfError,
    EncryptedPdfError,
    NonPdfError,
    PdfTooLargeError,
    process_pdf,
)


def test_extracts_text_and_one_based_page_references(two_page_pdf: bytes) -> None:
    result = process_pdf(two_page_pdf, 'synthetic-report.pdf')

    assert result.filename == 'synthetic-report.pdf'
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert 'Synthetic patient summary' in result.pages[0].text
    assert 'Synthetic follow-up plan' in result.pages[1].text
    assert result.warnings == []


def test_validates_pdf_content_instead_of_filename(two_page_pdf: bytes) -> None:
    result = process_pdf(two_page_pdf, 'synthetic-report.bin')

    assert result.page_count == 2


def test_rejects_empty_upload() -> None:
    with pytest.raises(EmptyPdfError, match='empty'):
        process_pdf(b'', 'empty.pdf')


def test_rejects_non_pdf_content() -> None:
    with pytest.raises(NonPdfError, match='not a PDF'):
        process_pdf(b'synthetic plain text', 'pretend.pdf')


def test_rejects_malformed_pdf() -> None:
    with pytest.raises(CorruptPdfError, match='malformed or corrupt'):
        process_pdf(b'%PDF-1.7\nsynthetic but invalid', 'malformed.pdf')


def test_rejects_encrypted_pdf(encrypted_pdf: bytes) -> None:
    with pytest.raises(EncryptedPdfError, match='Encrypted PDFs'):
        process_pdf(encrypted_pdf, 'encrypted.pdf')


def test_rejects_upload_over_configured_size() -> None:
    upload = b'%PDF-1.7\n' + (b'x' * 32)

    with pytest.raises(PdfTooLargeError, match='size limit'):
        process_pdf(upload, 'large.pdf', max_size_bytes=16)


def test_warns_about_blank_page_in_readable_pdf() -> None:
    result = process_pdf(
        make_text_pdf(['Synthetic readable page', None]),
        'partly-blank.pdf',
    )

    assert result.pages[0].warnings == []
    assert 'Page 2' in result.pages[1].warnings[0]
    assert 'OCR' in result.pages[1].warnings[0]


def test_explains_when_pdf_has_no_extractable_text() -> None:
    result = process_pdf(make_text_pdf([None]), 'image-only-simulation.pdf')

    assert result.page_count == 1
    assert result.pages[0].text == ''
    assert any('no extractable text' in warning for warning in result.warnings)
    assert any('OCR is not included' in warning for warning in result.warnings)
