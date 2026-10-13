'''Synthetic PDF factories shared by the ingestion tests.'''

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_text_pdf(page_texts: list[str | None]) -> bytes:
    '''Build an in-memory PDF containing only synthetic test text.'''

    output = BytesIO()
    document = canvas.Canvas(output, pagesize=letter)
    for text in page_texts:
        if text is not None:
            document.drawString(72, 720, text)
        document.showPage()
    document.save()
    return output.getvalue()


@pytest.fixture
def two_page_pdf() -> bytes:
    return make_text_pdf(
        ['Synthetic patient summary', 'Synthetic follow-up plan']
    )


@pytest.fixture
def encrypted_pdf(two_page_pdf: bytes) -> bytes:
    source = PdfReader(BytesIO(two_page_pdf))
    writer = PdfWriter()
    writer.append_pages_from_reader(source)
    writer.encrypt('synthetic-password')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()
