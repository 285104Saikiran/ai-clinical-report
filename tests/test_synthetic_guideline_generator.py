"""Regression tests for the PDFs used in the manual retrieval walkthrough."""

import importlib.util
from pathlib import Path

from langchain_core.documents import Document

from clinical_report_assistant.guideline_kb import GuidelineKnowledgeBase
from clinical_report_assistant.pdf_processing import process_pdf

_GENERATOR_PATH = (
    Path(__file__).parents[1] / "scripts" / "generate_synthetic_guidelines.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "synthetic_guideline_generator_under_test",
        _GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load the synthetic guideline generator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingVectorStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add_documents(
        self,
        *,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        self.documents.extend(documents)
        return ids


def test_generated_review_guidance_is_selectable_and_focused(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generator = _load_generator()
    monkeypatch.setattr(generator, "OUTPUT_DIRECTORY", tmp_path)
    generator.main()

    review_path = tmp_path / "synthetic_capstone_review_guidance.pdf"
    review = process_pdf(review_path.read_bytes(), review_path.name)

    assert review.page_count == 6
    review_text = "\n".join(page.text for page in review.pages)
    assert all(
        page.text.startswith("SYNTHETIC CAPSTONE REVIEW GUIDANCE - NOT MEDICAL ADVICE.")
        for page in review.pages
    )
    for expected in (
        "Glucose with unknown fasting status",
        "Missing reference range",
        "Mismatched units",
        "Medication uncertainty",
        "Incomplete allergy verification",
        "Missing follow-up",
    ):
        assert expected in review_text
    for excluded in (
        "diagnosis",
        "treatment",
        "urgent",
        "clinical threshold",
        "change medication",
    ):
        assert excluded not in review_text.lower()


def test_generated_guidelines_have_substantive_distinct_page_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generate_synthetic_guidelines = _load_generator()
    monkeypatch.setattr(
        generate_synthetic_guidelines,
        "OUTPUT_DIRECTORY",
        tmp_path,
    )

    generate_synthetic_guidelines.main()

    hydration_path = tmp_path / "synthetic_hydration_guideline.pdf"
    sleep_path = tmp_path / "synthetic_sleep_guideline.pdf"
    hydration = process_pdf(hydration_path.read_bytes(), hydration_path.name)
    sleep = process_pdf(sleep_path.read_bytes(), sleep_path.name)

    assert hydration.page_count == sleep.page_count == 2
    assert all(len(page.text) > 500 for page in hydration.pages)
    assert all(len(page.text) > 500 for page in sleep.pages)
    assert "BLUE CUP CHECK" in hydration.pages[0].text
    assert "RIVER ARCHIVE COMPLETE" in hydration.pages[1].text
    assert "SILVER CLOCK CHECK" in sleep.pages[0].text
    assert "MOON ARCHIVE COMPLETE" in sleep.pages[1].text
    assert "Moon-4" not in "\n".join(page.text for page in hydration.pages)
    assert "River-7" not in "\n".join(page.text for page in sleep.pages)

    store = RecordingVectorStore()
    index_result = GuidelineKnowledgeBase(
        storage_path=tmp_path / "test-index",
        vector_store=store,
    ).index_pdf(hydration_path.read_bytes(), hydration_path.name)

    assert index_result.chunks_added > hydration.page_count
    assert {document.metadata["page_number"] for document in store.documents} == {
        1,
        2,
    }
