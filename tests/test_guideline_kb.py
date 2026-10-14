"""Offline tests for page-aware guideline indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import make_text_pdf
from langchain_core.documents import Document

from clinical_report_assistant.guideline_kb import (
    EMBEDDING_MODEL_NAME,
    EmptyGuidelineIndexError,
    EmptyGuidelineQueryError,
    GuidelineKnowledgeBase,
    IncompatibleGuidelineIndexError,
    NoExtractableGuidelineTextError,
)
from clinical_report_assistant.pdf_processing import process_pdf


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_batches: list[tuple[list[Document], list[str]]] = []
        self.search_results: list[Document] = []
        self.search_queries: list[tuple[str, int]] = []

    def add_documents(
        self,
        *,
        documents: list[Document],
        ids: list[str],
    ) -> list[str]:
        self.added_batches.append((documents, ids))
        return ids

    def similarity_search(self, query: str, *, k: int) -> list[Document]:
        self.search_queries.append((query, k))
        return self.search_results[:k]


def test_index_preserves_metadata_and_uses_deterministic_ids(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    first_store = FakeVectorStore()
    first = GuidelineKnowledgeBase(
        storage_path=tmp_path / "first",
        chunk_size=80,
        chunk_overlap=10,
        vector_store=first_store,
    )

    result = first.index_pdf(synthetic_guideline_pdf, "synthetic-guideline.pdf")

    documents, ids = first_store.added_batches[0]
    expected_hash = hashlib.sha256(synthetic_guideline_pdf).hexdigest()
    assert result.document_hash == expected_hash
    assert result.chunks_added == len(documents) == len(ids)
    assert {document.metadata["page_number"] for document in documents} == {1, 2}
    assert all(
        document.metadata["document_name"] == "synthetic-guideline.pdf"
        for document in documents
    )
    assert all(
        document.metadata["document_hash"] == expected_hash for document in documents
    )
    assert [document.metadata["chunk_id"] for document in documents] == ids
    assert len(documents) > 2
    for page_number in (1, 2):
        page_chunks = [
            document
            for document in documents
            if document.metadata["page_number"] == page_number
        ]
        assert len(page_chunks) > 1
        assert [document.metadata["chunk_index"] for document in page_chunks] == list(
            range(len(page_chunks))
        )

    second_store = FakeVectorStore()
    second = GuidelineKnowledgeBase(
        storage_path=tmp_path / "second",
        chunk_size=80,
        chunk_overlap=10,
        vector_store=second_store,
    )
    second.index_pdf(synthetic_guideline_pdf, "synthetic-guideline.pdf")
    assert second_store.added_batches[0][1] == ids


def test_duplicate_document_is_not_indexed_twice(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    store = FakeVectorStore()
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=store,
    )

    first = knowledge_base.index_pdf(
        synthetic_guideline_pdf,
        "synthetic-guideline.pdf",
    )
    duplicate = knowledge_base.index_pdf(
        synthetic_guideline_pdf,
        "renamed-copy.pdf",
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.chunks_added == 0
    assert len(store.added_batches) == 1
    assert knowledge_base.indexed_document_count == 1

    reloaded_store = FakeVectorStore()
    reloaded = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=reloaded_store,
    )
    persisted_duplicate = reloaded.index_pdf(
        synthetic_guideline_pdf,
        "another-copy.pdf",
    )
    assert persisted_duplicate.duplicate is True
    assert reloaded_store.added_batches == []


def test_manifest_records_embedding_and_chunk_configuration(
    tmp_path: Path,
) -> None:
    GuidelineKnowledgeBase(
        storage_path=tmp_path,
        chunk_size=600,
        chunk_overlap=60,
        vector_store=FakeVectorStore(),
    )

    manifest = json.loads(
        (tmp_path / "index_manifest.json").read_text(encoding="utf-8")
    )
    configuration = manifest["configuration"]
    assert configuration["embedding_model"] == EMBEDDING_MODEL_NAME
    assert configuration["chunk_size"] == 600
    assert configuration["chunk_overlap"] == 60
    assert configuration["splitter"] == "RecursiveCharacterTextSplitter"


def test_rejects_incompatible_existing_index(tmp_path: Path) -> None:
    GuidelineKnowledgeBase(
        storage_path=tmp_path,
        chunk_size=600,
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(IncompatibleGuidelineIndexError, match="chunk_size"):
        GuidelineKnowledgeBase(
            storage_path=tmp_path,
            chunk_size=601,
            vector_store=FakeVectorStore(),
        )


def test_rejects_existing_storage_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "chroma.sqlite3").write_bytes(b"synthetic-not-a-database")

    with pytest.raises(IncompatibleGuidelineIndexError, match="without"):
        GuidelineKnowledgeBase(
            storage_path=tmp_path,
            vector_store=FakeVectorStore(),
        )


def test_retrieval_returns_passages_with_sources(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    store = FakeVectorStore()
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=store,
    )
    knowledge_base.index_pdf(
        synthetic_guideline_pdf,
        "synthetic-guideline.pdf",
    )
    indexed_documents, _ = store.added_batches[0]
    store.search_results = indexed_documents

    passages = knowledge_base.retrieve("fictional River-7 hydration", top_k=2)

    assert len(passages) == 2
    assert passages[0].document_name == "synthetic-guideline.pdf"
    assert passages[0].page_number >= 1
    assert passages[0].chunk_id
    assert (
        passages[0].document_hash == hashlib.sha256(synthetic_guideline_pdf).hexdigest()
    )
    assert "SYNTHETIC" in passages[0].text
    assert store.search_queries == [("fictional River-7 hydration", 2)]


def test_index_fingerprint_changes_when_guideline_set_changes(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=FakeVectorStore(),
    )
    empty = knowledge_base.index_fingerprint()
    result = knowledge_base.index_pdf(
        synthetic_guideline_pdf,
        "synthetic-guideline.pdf",
    )
    populated = knowledge_base.index_fingerprint()

    assert empty.fingerprint != populated.fingerprint
    assert populated.document_hashes == [result.document_hash]


def test_rejects_empty_query_and_empty_index(tmp_path: Path) -> None:
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(EmptyGuidelineQueryError):
        knowledge_base.retrieve("   ")
    with pytest.raises(EmptyGuidelineIndexError):
        knowledge_base.retrieve("synthetic query")


def test_rejects_guideline_without_selectable_text(tmp_path: Path) -> None:
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(NoExtractableGuidelineTextError, match="OCR"):
        knowledge_base.index_pdf(
            make_text_pdf([None]),
            "synthetic-image-only-guideline.pdf",
        )


def test_report_processing_does_not_touch_guideline_storage(
    tmp_path: Path,
    two_page_pdf: bytes,
) -> None:
    store = FakeVectorStore()
    knowledge_base = GuidelineKnowledgeBase(
        storage_path=tmp_path,
        vector_store=store,
    )

    report = process_pdf(two_page_pdf, "synthetic-clinical-report.pdf")

    assert report.page_count == 2
    assert store.added_batches == []
    assert knowledge_base.indexed_document_count == 0
