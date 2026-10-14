"""Persistent, page-aware guideline indexing and local retrieval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from clinical_report_assistant.pdf_processing import PdfDocument, process_pdf

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_COLLECTION_NAME = "clinical-guidelines"
DEFAULT_STORAGE_PATH = Path("data/chroma_guidelines")
INDEX_SCHEMA_VERSION = 1
_MANIFEST_FILENAME = "index_manifest.json"


class GuidelineKnowledgeBaseError(RuntimeError):
    """Base class for safe guideline knowledge-base errors."""


class IncompatibleGuidelineIndexError(GuidelineKnowledgeBaseError):
    """Raised when persisted index settings differ from current settings."""


class NoExtractableGuidelineTextError(GuidelineKnowledgeBaseError):
    """Raised when a guideline has no selectable text to index."""


class EmptyGuidelineIndexError(GuidelineKnowledgeBaseError):
    """Raised when retrieval is attempted before any guideline is indexed."""


class EmptyGuidelineQueryError(GuidelineKnowledgeBaseError):
    """Raised when a retrieval query contains no text."""


class GuidelineIndexingError(GuidelineKnowledgeBaseError):
    """Raised when local embedding or Chroma persistence fails."""


class GuidelineRetrievalError(GuidelineKnowledgeBaseError):
    """Raised when local query embedding or retrieval fails."""


class GuidelineIndexResult(BaseModel):
    """Outcome of indexing one guideline document."""

    model_config = ConfigDict(frozen=True)

    document_name: str
    document_hash: str
    page_count: int = Field(ge=1)
    chunks_added: int = Field(ge=0)
    duplicate: bool


class RetrievedPassage(BaseModel):
    """One retrieved guideline passage with its source reference."""

    model_config = ConfigDict(frozen=True)

    text: str
    document_name: str
    page_number: int = Field(ge=1)
    chunk_id: str
    document_hash: str = Field(min_length=64, max_length=64)


class GuidelineIndexFingerprint(BaseModel):
    """Stable identity of the complete guideline index and its configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fingerprint: str = Field(min_length=64, max_length=64)
    document_hashes: list[str]
    configuration: dict[str, str | int | bool]


class GuidelineKnowledgeBase:
    """Index selectable-text guideline PDFs and retrieve local passages."""

    def __init__(
        self,
        *,
        storage_path: str | Path = DEFAULT_STORAGE_PATH,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embeddings: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size.")

        self._storage_path = Path(storage_path)
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._lock = Lock()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self._manifest = self._load_or_create_manifest()

    @property
    def indexed_document_count(self) -> int:
        return len(self._manifest["documents"])

    def index_fingerprint(self) -> GuidelineIndexFingerprint:
        """Return a deterministic fingerprint without opening the vector store."""

        document_hashes = sorted(self._manifest["documents"])
        payload = {
            "configuration": self._manifest["configuration"],
            "documents": {
                key: self._manifest["documents"][key] for key in document_hashes
            },
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return GuidelineIndexFingerprint(
            fingerprint=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            document_hashes=document_hashes,
            configuration=dict(self._manifest["configuration"]),
        )

    def index_pdf(
        self,
        uploaded_bytes: bytes,
        original_filename: str,
    ) -> GuidelineIndexResult:
        """Extract, page-chunk, embed, and persist one guideline PDF."""

        document_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        pdf = process_pdf(uploaded_bytes, original_filename)

        with self._lock:
            existing = self._manifest["documents"].get(document_hash)
            if existing is not None:
                return GuidelineIndexResult(
                    document_name=original_filename,
                    document_hash=document_hash,
                    page_count=pdf.page_count,
                    chunks_added=0,
                    duplicate=True,
                )

            documents, chunk_ids = self._build_chunks(pdf, document_hash)
            if not documents:
                raise NoExtractableGuidelineTextError(
                    "The guideline PDF contains no selectable text to index. "
                    "It may require OCR, which is not included."
                )

            try:
                self._get_vector_store().add_documents(
                    documents=documents,
                    ids=chunk_ids,
                )
            except Exception as exc:
                raise GuidelineIndexingError(
                    "Guideline indexing failed while creating local embeddings "
                    "or writing the Chroma index."
                ) from exc

            self._manifest["documents"][document_hash] = {
                "document_name": original_filename,
                "page_count": pdf.page_count,
                "chunk_count": len(documents),
            }
            self._write_manifest(self._manifest)

        return GuidelineIndexResult(
            document_name=original_filename,
            document_hash=document_hash,
            page_count=pdf.page_count,
            chunks_added=len(documents),
            duplicate=False,
        )

    def retrieve(self, query: str, *, top_k: int = 4) -> list[RetrievedPassage]:
        """Return top-k passages without interpreting distance as confidence."""

        if not query.strip():
            raise EmptyGuidelineQueryError("Enter a non-empty retrieval query.")
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50.")
        if self.indexed_document_count == 0:
            raise EmptyGuidelineIndexError(
                "The guideline index is empty. Index a guideline PDF first."
            )

        try:
            matches = self._get_vector_store().similarity_search(
                query.strip(),
                k=top_k,
            )
        except Exception as exc:
            raise GuidelineRetrievalError(
                "Local guideline retrieval failed. Verify the embedding model "
                "and Chroma index are available."
            ) from exc

        passages: list[RetrievedPassage] = []
        for match in matches:
            metadata = match.metadata
            try:
                passages.append(
                    RetrievedPassage(
                        text=match.page_content,
                        document_name=str(metadata["document_name"]),
                        page_number=int(metadata["page_number"]),
                        chunk_id=str(metadata["chunk_id"]),
                        document_hash=str(metadata["document_hash"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise IncompatibleGuidelineIndexError(
                    "A retrieved chunk is missing required source metadata. "
                    "Use a fresh guideline index."
                ) from exc
        return passages

    def _build_chunks(
        self,
        pdf: PdfDocument,
        document_hash: str,
    ) -> tuple[list[Document], list[str]]:
        documents: list[Document] = []
        chunk_ids: list[str] = []

        for page in pdf.pages:
            if not page.text.strip():
                continue
            for chunk_index, text in enumerate(self._splitter.split_text(page.text)):
                if not text.strip():
                    continue
                chunk_id = _make_chunk_id(
                    document_hash,
                    page.page_number,
                    chunk_index,
                    text,
                )
                metadata = {
                    "document_name": pdf.filename,
                    "document_hash": document_hash,
                    "page_number": page.page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                    "embedding_model": self._embedding_model,
                    "chunk_size": self._chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                }
                documents.append(Document(page_content=text, metadata=metadata))
                chunk_ids.append(chunk_id)
        return documents, chunk_ids

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            try:
                if self._embeddings is None:
                    self._embeddings = HuggingFaceEmbeddings(
                        model=self._embedding_model,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                        show_progress=False,
                    )
                self._vector_store = Chroma(
                    collection_name=self._collection_name,
                    embedding_function=self._embeddings,
                    persist_directory=str(self._storage_path),
                    collection_metadata=self._expected_configuration(),
                    client_settings=Settings(anonymized_telemetry=False),
                )
            except Exception as exc:
                raise GuidelineIndexingError(
                    "The local embedding model or Chroma index could not be opened."
                ) from exc
        return self._vector_store

    def _expected_configuration(self) -> dict[str, str | int | bool]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "embedding_model": self._embedding_model,
            "normalize_embeddings": True,
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }

    def _load_or_create_manifest(self) -> dict[str, Any]:
        self._storage_path.mkdir(parents=True, exist_ok=True)
        manifest_path = self._storage_path / _MANIFEST_FILENAME
        expected = self._expected_configuration()

        if not manifest_path.exists():
            existing_entries = list(self._storage_path.iterdir())
            if existing_entries:
                raise IncompatibleGuidelineIndexError(
                    "The guideline storage directory contains an index without "
                    "a compatibility manifest. Use a fresh storage directory."
                )
            manifest: dict[str, Any] = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "configuration": expected,
                "documents": {},
            }
            self._write_manifest(manifest)
            return manifest

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IncompatibleGuidelineIndexError(
                "The guideline index manifest is unreadable. Use a fresh index."
            ) from exc

        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("documents"), dict
        ):
            raise IncompatibleGuidelineIndexError(
                "The guideline index manifest has an unsupported format."
            )

        current = manifest.get("configuration")
        if current != expected:
            mismatches = sorted(
                key
                for key in expected
                if not isinstance(current, dict) or current.get(key) != expected[key]
            )
            fields = ", ".join(mismatches) or "configuration"
            raise IncompatibleGuidelineIndexError(
                "The existing guideline index is incompatible in: "
                f"{fields}. Use a fresh storage directory or matching settings."
            )
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest_path = self._storage_path / _MANIFEST_FILENAME
        temporary_path = self._storage_path / f"{_MANIFEST_FILENAME}.tmp"
        temporary_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)


def _make_chunk_id(
    document_hash: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    identity = f"{document_hash}:{page_number}:{chunk_index}:{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
