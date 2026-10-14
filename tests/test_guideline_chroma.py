"""Offline persistent-Chroma test with a deterministic fake embedder."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from clinical_report_assistant.guideline_kb import GuidelineKnowledgeBase


class DeterministicFakeEmbeddings(Embeddings):
    """Small local test vectors that never download a model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float("hydration" in lowered or "river-7" in lowered),
            float("follow-up" in lowered),
            float("sleep" in lowered or "moon-4" in lowered),
            1.0,
        ]


def test_persistent_chroma_with_mocked_embeddings(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    first = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        embeddings=DeterministicFakeEmbeddings(),
    )
    indexed = first.index_pdf(
        synthetic_guideline_pdf,
        "synthetic-guideline.pdf",
    )

    reopened = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        embeddings=DeterministicFakeEmbeddings(),
    )
    passages = reopened.retrieve("River-7 hydration", top_k=1)
    duplicate = reopened.index_pdf(
        synthetic_guideline_pdf,
        "renamed-guideline.pdf",
    )

    assert indexed.chunks_added > 0
    assert len(passages) == 1
    assert passages[0].document_name == "synthetic-guideline.pdf"
    assert passages[0].page_number == 1
    assert "River-7" in passages[0].text
    assert duplicate.duplicate is True
