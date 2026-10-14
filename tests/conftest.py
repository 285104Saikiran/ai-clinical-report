"""Synthetic PDF factories shared by the ingestion tests."""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def make_text_pdf(page_texts: list[str | None]) -> bytes:
    """Build an in-memory PDF containing only synthetic test text."""

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
    return make_text_pdf(["Synthetic patient summary", "Synthetic follow-up plan"])


@pytest.fixture
def synthetic_guideline_pdf() -> bytes:
    label = "SYNTHETIC TRAINING GUIDELINE - NOT MEDICAL ADVICE."
    return make_text_pdf(
        [
            (
                f"{label} Fictional River-7 hydration log for capstone testing. "
                "Record the beverage label, simulated volume, source, and time. "
                "Use BLUE CUP CHECK to mark a completed training review. "
                "Missing values stay NOT RECORDED and are never inferred."
            ),
            (
                f"{label} Fictional follow-up reconciliation exercise. Compare "
                "River-7 entries with the training worksheet, retain both values, "
                "and label differences TRAINING REVIEW NEEDED. The completion "
                "marker is RIVER ARCHIVE COMPLETE."
            ),
        ]
    )


@pytest.fixture
def encrypted_pdf(two_page_pdf: bytes) -> bytes:
    source = PdfReader(BytesIO(two_page_pdf))
    writer = PdfWriter()
    writer.append_pages_from_reader(source)
    writer.encrypt("synthetic-password")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()
