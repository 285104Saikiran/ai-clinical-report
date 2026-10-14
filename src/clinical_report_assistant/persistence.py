"""Validated local SQLite persistence for report analyses and summaries."""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from clinical_report_assistant.config import GeminiConfig
from clinical_report_assistant.recommendation_agent import (
    DEFAULT_MAX_RETRIEVAL_QUERIES,
    DEFAULT_RECOMMENDATION_TIMEOUT_SECONDS,
    DEFAULT_RETRIEVAL_TOP_K,
    RECOMMENDATION_PROMPT_VERSION,
    RECOMMENDATION_SCHEMA_VERSION,
    RecommendationDraftResult,
    RecommendationFindingEvidence,
)
from clinical_report_assistant.report_analysis_agent import (
    DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    REPORT_ANALYSIS_PROMPT_VERSION,
    REPORT_ANALYSIS_SCHEMA_VERSION,
    NarrativeFinding,
    ReportAnalysisResult,
    ReportFinding,
)
from clinical_report_assistant.summary_agent import (
    DEFAULT_SUMMARY_TIMEOUT_SECONDS,
    SUMMARY_PROMPT_VERSION,
    SUMMARY_SCHEMA_VERSION,
    ReportSummaryResult,
    SummaryApplicabilityEvidence,
    SummarySourceEvidence,
)

DEFAULT_RESULTS_DATABASE_PATH = Path("data/local_results/results.sqlite3")
DATABASE_SCHEMA_VERSION = 2
_HASH_LENGTH = 64
_PROVIDER_NAME = "crewai-native-gemini"


class PersistenceError(RuntimeError):
    """Safe base error for local saved-result operations."""


class SavedResultValidationError(PersistenceError):
    """Raised when saved JSON or its dependency relationship is invalid."""


class IncompatibleDatabaseError(PersistenceError):
    """Raised when a database schema cannot be used safely."""


class RuntimeVersions(BaseModel):
    """Auditable runtime dependency versions stored with every result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    application: str
    python: str
    pydantic: str
    crewai: str
    google_genai: str


class AnalysisGenerationSettings(BaseModel):
    """Non-secret settings that define analysis compatibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = _PROVIDER_NAME
    model: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    retry_attempts: int = Field(default=1, ge=1)
    temperature: float | None = None
    prompt_version: str = REPORT_ANALYSIS_PROMPT_VERSION
    schema_version: str = REPORT_ANALYSIS_SCHEMA_VERSION
    schema_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)


class SummaryGenerationSettings(BaseModel):
    """Non-secret settings that define summary compatibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = _PROVIDER_NAME
    model: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    retry_attempts: int = Field(default=1, ge=1)
    temperature: float | None = None
    prompt_version: str = SUMMARY_PROMPT_VERSION
    schema_version: str = SUMMARY_SCHEMA_VERSION
    schema_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)


class RecommendationGenerationSettings(BaseModel):
    """Non-secret settings and guideline identity defining draft compatibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = _PROVIDER_NAME
    model: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    retry_attempts: int = Field(default=1, ge=1)
    temperature: float | None = None
    prompt_version: str = RECOMMENDATION_PROMPT_VERSION
    schema_version: str = RECOMMENDATION_SCHEMA_VERSION
    schema_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    retrieval_top_k: int = Field(ge=1, le=10)
    max_retrieval_queries: int = Field(ge=1, le=5)
    guideline_index_fingerprint: str = Field(
        min_length=_HASH_LENGTH, max_length=_HASH_LENGTH
    )
    guideline_document_hashes: list[str]

    @field_validator("guideline_index_fingerprint")
    @classmethod
    def fingerprint_is_lowercase_hex(cls, value: str) -> str:
        return _validated_hash(value)

    @field_validator("guideline_document_hashes")
    @classmethod
    def document_hashes_are_canonical(cls, value: list[str]) -> list[str]:
        validated = [_validated_hash(item) for item in value]
        if validated != sorted(set(validated)):
            raise ValueError("guideline_document_hashes must be sorted and unique.")
        return validated


class SavedAnalysisRecord(BaseModel):
    """One immutable validated analysis stored locally."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    report_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    compatibility_key: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    created_at: datetime
    settings: AnalysisGenerationSettings
    runtime_versions: RuntimeVersions
    result: ReportAnalysisResult

    @field_validator("report_hash", "compatibility_key")
    @classmethod
    def hashes_are_lowercase_hex(cls, value: str) -> str:
        return _validated_hash(value)


class SavedSummaryRecord(BaseModel):
    """One immutable validated summary tied to an exact saved analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    report_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    analysis_record_id: str = Field(min_length=1)
    analysis_result_hash: str = Field(
        min_length=_HASH_LENGTH,
        max_length=_HASH_LENGTH,
    )
    compatibility_key: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    created_at: datetime
    settings: SummaryGenerationSettings
    runtime_versions: RuntimeVersions
    result: ReportSummaryResult

    @field_validator("report_hash", "analysis_result_hash", "compatibility_key")
    @classmethod
    def hashes_are_lowercase_hex(cls, value: str) -> str:
        return _validated_hash(value)


class SavedRecommendationDraft(BaseModel):
    """One immutable validated draft tied to exact analysis and summary rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    report_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    analysis_record_id: str = Field(min_length=1)
    analysis_result_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    summary_record_id: str = Field(min_length=1)
    summary_result_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    compatibility_key: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    created_at: datetime
    version_number: int = Field(ge=1)
    origin: Literal["generated", "edited"]
    parent_draft_id: str | None = None
    settings: RecommendationGenerationSettings
    runtime_versions: RuntimeVersions
    result: RecommendationDraftResult

    @field_validator(
        "report_hash",
        "analysis_result_hash",
        "summary_result_hash",
        "compatibility_key",
    )
    @classmethod
    def hashes_are_lowercase_hex(cls, value: str) -> str:
        return _validated_hash(value)


class SavedReviewDecision(BaseModel):
    """One immutable human decision bound to an exact immutable draft."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(min_length=1)
    draft_record_id: str = Field(min_length=1)
    draft_result_hash: str = Field(min_length=_HASH_LENGTH, max_length=_HASH_LENGTH)
    decision: Literal["approved", "rejected"]
    notes: str = Field(default="", max_length=2000)
    created_at: datetime

    @field_validator("draft_result_hash")
    @classmethod
    def draft_hash_is_lowercase_hex(cls, value: str) -> str:
        return _validated_hash(value)


class DeletionResult(BaseModel):
    """Counts returned after confirmed report-associated deletion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyses_deleted: int = Field(ge=0)
    summaries_deleted: int = Field(ge=0)
    recommendation_drafts_deleted: int = Field(default=0, ge=0)
    review_decisions_deleted: int = Field(default=0, ge=0)


class AnalysisResolution(BaseModel):
    """Saved or newly generated analysis plus its reuse status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: SavedAnalysisRecord
    reused: bool


class SummaryResolution(BaseModel):
    """Saved or newly generated summary plus its reuse status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: SavedSummaryRecord
    reused: bool


class RecommendationResolution(BaseModel):
    """Saved or newly generated recommendation draft plus reuse status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: SavedRecommendationDraft
    reused: bool


def report_sha256(uploaded_bytes: bytes) -> str:
    """Identify a report strictly by the SHA-256 of its uploaded bytes."""

    return hashlib.sha256(uploaded_bytes).hexdigest()


def current_runtime_versions() -> RuntimeVersions:
    """Return non-secret versions needed to audit compatibility decisions."""

    return RuntimeVersions(
        application=_package_version("clinical-report-assistant"),
        python=platform.python_version(),
        pydantic=_package_version("pydantic"),
        crewai=_package_version("crewai"),
        google_genai=_package_version("google-genai"),
    )


def analysis_generation_settings(config: GeminiConfig) -> AnalysisGenerationSettings:
    """Build the exact analysis compatibility settings for the current app."""

    return AnalysisGenerationSettings(
        model=config.model,
        max_output_tokens=config.report_analysis_max_output_tokens,
        timeout_seconds=DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
        schema_hash=_model_schema_hash(ReportAnalysisResult),
    )


def summary_generation_settings(config: GeminiConfig) -> SummaryGenerationSettings:
    """Build the exact summary compatibility settings for the current app."""

    return SummaryGenerationSettings(
        model=config.model,
        max_output_tokens=config.summary_max_output_tokens,
        timeout_seconds=DEFAULT_SUMMARY_TIMEOUT_SECONDS,
        schema_hash=_model_schema_hash(ReportSummaryResult),
    )


def recommendation_generation_settings(
    config: GeminiConfig,
    *,
    guideline_index_fingerprint: str,
    guideline_document_hashes: list[str],
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    max_retrieval_queries: int = DEFAULT_MAX_RETRIEVAL_QUERIES,
) -> RecommendationGenerationSettings:
    """Build exact recommendation, retrieval, and guideline compatibility settings."""

    return RecommendationGenerationSettings(
        model=config.model,
        max_output_tokens=config.recommendation_max_output_tokens,
        timeout_seconds=DEFAULT_RECOMMENDATION_TIMEOUT_SECONDS,
        schema_hash=_model_schema_hash(RecommendationDraftResult),
        retrieval_top_k=retrieval_top_k,
        max_retrieval_queries=max_retrieval_queries,
        guideline_index_fingerprint=guideline_index_fingerprint,
        guideline_document_hashes=sorted(guideline_document_hashes),
    )


class SQLiteResultStore:
    """Store immutable validated results without storing PDF bytes or secrets."""

    def __init__(self, path: str | Path = DEFAULT_RESULTS_DATABASE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_analysis(
        self,
        report_hash: str,
        settings: AnalysisGenerationSettings,
        result: ReportAnalysisResult,
        *,
        runtime_versions: RuntimeVersions | None = None,
        created_at: datetime | None = None,
    ) -> SavedAnalysisRecord:
        """Validate and append one analysis record."""

        validated_hash = _validated_hash(report_hash)
        validated_result = ReportAnalysisResult.model_validate(result.model_dump())
        validated_settings = AnalysisGenerationSettings.model_validate(
            settings.model_dump()
        )
        versions = RuntimeVersions.model_validate(
            (runtime_versions or current_runtime_versions()).model_dump()
        )
        compatibility_key = analysis_compatibility_key(
            validated_hash,
            validated_settings,
            versions,
        )
        record = SavedAnalysisRecord(
            record_id=uuid.uuid4().hex,
            report_hash=validated_hash,
            compatibility_key=compatibility_key,
            created_at=created_at or datetime.now(UTC),
            settings=validated_settings,
            runtime_versions=versions,
            result=validated_result,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO analysis_records
                        (record_id, report_hash, compatibility_key, created_at, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.report_hash,
                        record.compatibility_key,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                "The validated analysis could not be saved locally."
            ) from exc
        return record

    def load_analysis(
        self,
        report_hash: str,
        settings: AnalysisGenerationSettings,
        *,
        runtime_versions: RuntimeVersions | None = None,
    ) -> SavedAnalysisRecord | None:
        """Load only the newest exactly compatible analysis."""

        validated_hash = _validated_hash(report_hash)
        versions = runtime_versions or current_runtime_versions()
        key = analysis_compatibility_key(validated_hash, settings, versions)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, compatibility_key, payload_json
                FROM analysis_records
                WHERE report_hash = ? AND compatibility_key = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (validated_hash, key),
            ).fetchone()
        if row is None:
            return None
        return self._validated_analysis_row(row, expected_key=key)

    def load_analysis_by_id(self, record_id: str) -> SavedAnalysisRecord:
        """Load one exact analysis dependency through Pydantic validation."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, compatibility_key, payload_json
                FROM analysis_records WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()
        if row is None:
            raise SavedResultValidationError(
                "The saved analysis dependency is unavailable."
            )
        return self._validated_analysis_row(row, expected_key=None)

    def save_summary(
        self,
        analysis: SavedAnalysisRecord,
        settings: SummaryGenerationSettings,
        result: ReportSummaryResult,
        *,
        runtime_versions: RuntimeVersions | None = None,
        created_at: datetime | None = None,
    ) -> SavedSummaryRecord:
        """Validate and append a summary tied to the exact saved analysis."""

        stored_analysis = self.load_analysis_by_id(analysis.record_id)
        if stored_analysis != analysis:
            raise SavedResultValidationError(
                "The summary analysis dependency does not match its saved record."
            )
        validated_result = ReportSummaryResult.model_validate(result.model_dump())
        validated_settings = SummaryGenerationSettings.model_validate(
            settings.model_dump()
        )
        _validate_summary_evidence(validated_result, stored_analysis.result)
        versions = RuntimeVersions.model_validate(
            (runtime_versions or current_runtime_versions()).model_dump()
        )
        analysis_result_hash = _model_hash(stored_analysis.result)
        compatibility_key = summary_compatibility_key(
            stored_analysis,
            validated_settings,
            versions,
        )
        record = SavedSummaryRecord(
            record_id=uuid.uuid4().hex,
            report_hash=stored_analysis.report_hash,
            analysis_record_id=stored_analysis.record_id,
            analysis_result_hash=analysis_result_hash,
            compatibility_key=compatibility_key,
            created_at=created_at or datetime.now(UTC),
            settings=validated_settings,
            runtime_versions=versions,
            result=validated_result,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO summary_records
                        (record_id, report_hash, analysis_record_id,
                         compatibility_key, created_at, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.report_hash,
                        record.analysis_record_id,
                        record.compatibility_key,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                "The validated summary could not be saved locally."
            ) from exc
        return record

    def load_summary(
        self,
        analysis: SavedAnalysisRecord,
        settings: SummaryGenerationSettings,
        *,
        runtime_versions: RuntimeVersions | None = None,
    ) -> SavedSummaryRecord | None:
        """Load only a summary for the exact analysis and settings."""

        stored_analysis = self.load_analysis_by_id(analysis.record_id)
        if stored_analysis != analysis:
            raise SavedResultValidationError(
                "The summary analysis dependency does not match its saved record."
            )
        versions = runtime_versions or current_runtime_versions()
        key = summary_compatibility_key(stored_analysis, settings, versions)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, analysis_record_id,
                       compatibility_key, payload_json
                FROM summary_records
                WHERE analysis_record_id = ? AND compatibility_key = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (analysis.record_id, key),
            ).fetchone()
        if row is None:
            return None
        try:
            record = SavedSummaryRecord.model_validate_json(row["payload_json"])
            if (
                record.record_id != row["record_id"]
                or record.report_hash != row["report_hash"]
                or record.analysis_record_id != row["analysis_record_id"]
                or record.compatibility_key != row["compatibility_key"]
                or record.compatibility_key != key
                or record.analysis_result_hash != _model_hash(stored_analysis.result)
                or record.compatibility_key
                != summary_compatibility_key(
                    stored_analysis,
                    record.settings,
                    record.runtime_versions,
                )
            ):
                raise ValueError("saved summary metadata mismatch")
            _validate_summary_evidence(record.result, stored_analysis.result)
        except (ValidationError, ValueError, TypeError) as exc:
            raise SavedResultValidationError(
                "A saved summary failed validation and was not reused."
            ) from exc
        return record

    def load_summary_by_id(self, record_id: str) -> SavedSummaryRecord:
        """Load one exact summary dependency through Pydantic validation."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, analysis_record_id,
                       compatibility_key, payload_json
                FROM summary_records WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()
        if row is None:
            raise SavedResultValidationError(
                "The saved summary dependency is unavailable."
            )
        analysis = self.load_analysis_by_id(row["analysis_record_id"])
        try:
            record = SavedSummaryRecord.model_validate_json(row["payload_json"])
            if (
                record.record_id != row["record_id"]
                or record.report_hash != row["report_hash"]
                or record.analysis_record_id != row["analysis_record_id"]
                or record.compatibility_key != row["compatibility_key"]
                or record.analysis_result_hash != _model_hash(analysis.result)
                or record.compatibility_key
                != summary_compatibility_key(
                    analysis, record.settings, record.runtime_versions
                )
            ):
                raise ValueError("saved summary metadata mismatch")
            _validate_summary_evidence(record.result, analysis.result)
        except (ValidationError, ValueError, TypeError) as exc:
            raise SavedResultValidationError(
                "A saved summary failed validation and was not reused."
            ) from exc
        return record

    def save_recommendation(
        self,
        analysis: SavedAnalysisRecord,
        summary: SavedSummaryRecord,
        settings: RecommendationGenerationSettings,
        result: RecommendationDraftResult,
        *,
        runtime_versions: RuntimeVersions | None = None,
        created_at: datetime | None = None,
        origin: Literal["generated", "edited"] = "generated",
        parent_draft_id: str | None = None,
    ) -> SavedRecommendationDraft:
        """Validate and append one draft tied to exact saved upstream rows."""

        stored_summary = self._validated_summary_dependency(analysis, summary)
        validated_result = RecommendationDraftResult.model_validate(result.model_dump())
        validated_settings = RecommendationGenerationSettings.model_validate(
            settings.model_dump()
        )
        _validate_recommendation_evidence(
            validated_result,
            analysis.result,
            validated_settings,
        )
        versions = RuntimeVersions.model_validate(
            (runtime_versions or current_runtime_versions()).model_dump()
        )
        key = recommendation_compatibility_key(
            analysis, stored_summary, validated_settings, versions
        )
        if origin == "edited" and parent_draft_id is None:
            raise SavedResultValidationError(
                "An edited draft requires an exact parent draft."
            )
        if parent_draft_id is not None:
            parent = self.load_recommendation_by_id(parent_draft_id)
            if parent.compatibility_key != key:
                raise SavedResultValidationError(
                    "The edited draft parent is not compatible."
                )
        with self._connect() as connection:
            prior_version = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0)
                FROM recommendation_drafts
                WHERE summary_record_id = ? AND compatibility_key = ?
                """,
                (stored_summary.record_id, key),
            ).fetchone()[0]
        record = SavedRecommendationDraft(
            record_id=uuid.uuid4().hex,
            report_hash=analysis.report_hash,
            analysis_record_id=analysis.record_id,
            analysis_result_hash=_model_hash(analysis.result),
            summary_record_id=stored_summary.record_id,
            summary_result_hash=_model_hash(stored_summary.result),
            compatibility_key=key,
            created_at=created_at or datetime.now(UTC),
            version_number=prior_version + 1,
            origin=origin,
            parent_draft_id=parent_draft_id,
            settings=validated_settings,
            runtime_versions=versions,
            result=validated_result,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO recommendation_drafts
                        (record_id, report_hash, analysis_record_id,
                         summary_record_id, compatibility_key, created_at,
                         version_number, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.report_hash,
                        record.analysis_record_id,
                        record.summary_record_id,
                        record.compatibility_key,
                        record.created_at.isoformat(),
                        record.version_number,
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                "The validated recommendation draft could not be saved locally."
            ) from exc
        return record

    def load_recommendation(
        self,
        analysis: SavedAnalysisRecord,
        summary: SavedSummaryRecord,
        settings: RecommendationGenerationSettings,
        *,
        runtime_versions: RuntimeVersions | None = None,
    ) -> SavedRecommendationDraft | None:
        """Load only the newest draft for exact upstream and guideline settings."""

        stored_summary = self._validated_summary_dependency(analysis, summary)
        versions = runtime_versions or current_runtime_versions()
        key = recommendation_compatibility_key(
            analysis, stored_summary, settings, versions
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, analysis_record_id,
                       summary_record_id, compatibility_key, payload_json
                FROM recommendation_drafts
                WHERE summary_record_id = ? AND compatibility_key = ?
                ORDER BY version_number DESC, rowid DESC
                LIMIT 1
                """,
                (stored_summary.record_id, key),
            ).fetchone()
        if row is None:
            return None
        return self._validated_recommendation_row(
            row, analysis, stored_summary, expected_key=key
        )

    def load_recommendation_by_id(
        self,
        record_id: str,
    ) -> SavedRecommendationDraft:
        """Load an exact immutable draft through Pydantic validation."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT record_id, report_hash, analysis_record_id,
                       summary_record_id, compatibility_key, payload_json
                FROM recommendation_drafts WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()
        if row is None:
            raise SavedResultValidationError(
                "The saved recommendation draft is unavailable."
            )
        analysis = self.load_analysis_by_id(row["analysis_record_id"])
        with self._connect() as connection:
            summary_row = connection.execute(
                "SELECT payload_json FROM summary_records WHERE record_id = ?",
                (row["summary_record_id"],),
            ).fetchone()
        if summary_row is None:
            raise SavedResultValidationError(
                "The saved recommendation summary dependency is unavailable."
            )
        try:
            summary = SavedSummaryRecord.model_validate_json(
                summary_row["payload_json"]
            )
        except ValidationError as exc:
            raise SavedResultValidationError(
                "The saved recommendation summary dependency is invalid."
            ) from exc
        self._validated_summary_dependency(analysis, summary)
        return self._validated_recommendation_row(
            row, analysis, summary, expected_key=None
        )

    def save_edited_recommendation(
        self,
        parent: SavedRecommendationDraft,
        result: RecommendationDraftResult,
        *,
        created_at: datetime | None = None,
    ) -> SavedRecommendationDraft:
        """Create a new unreviewed draft version from an exact parent."""

        stored_parent = self.load_recommendation_by_id(parent.record_id)
        if stored_parent != parent:
            raise SavedResultValidationError("The edited draft parent is invalid.")
        analysis = self.load_analysis_by_id(parent.analysis_record_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM summary_records WHERE record_id = ?",
                (parent.summary_record_id,),
            ).fetchone()
        if row is None:
            raise SavedResultValidationError(
                "The edited draft summary dependency is unavailable."
            )
        summary = SavedSummaryRecord.model_validate_json(row["payload_json"])
        return self.save_recommendation(
            analysis,
            summary,
            parent.settings,
            result,
            runtime_versions=parent.runtime_versions,
            created_at=created_at,
            origin="edited",
            parent_draft_id=parent.record_id,
        )

    def save_review_decision(
        self,
        draft: SavedRecommendationDraft,
        decision: Literal["approved", "rejected"],
        notes: str = "",
        *,
        created_at: datetime | None = None,
    ) -> SavedReviewDecision:
        """Append a human review decision bound to exact immutable draft content."""

        stored = self.load_recommendation_by_id(draft.record_id)
        if stored != draft:
            raise SavedResultValidationError("The reviewed draft is invalid.")
        record = SavedReviewDecision(
            decision_id=uuid.uuid4().hex,
            draft_record_id=draft.record_id,
            draft_result_hash=_model_hash(draft.result),
            decision=decision,
            notes=notes.strip(),
            created_at=created_at or datetime.now(UTC),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO recommendation_reviews
                        (decision_id, draft_record_id, created_at, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        record.decision_id,
                        record.draft_record_id,
                        record.created_at.isoformat(),
                        record.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                "The recommendation review decision could not be saved locally."
            ) from exc
        return record

    def load_latest_review(
        self,
        draft: SavedRecommendationDraft,
    ) -> SavedReviewDecision | None:
        """Load the newest decision only for this exact draft version."""

        stored = self.load_recommendation_by_id(draft.record_id)
        if stored != draft:
            raise SavedResultValidationError("The reviewed draft is invalid.")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT decision_id, draft_record_id, payload_json
                FROM recommendation_reviews
                WHERE draft_record_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (draft.record_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            decision = SavedReviewDecision.model_validate_json(row["payload_json"])
            if (
                decision.decision_id != row["decision_id"]
                or decision.draft_record_id != row["draft_record_id"]
                or decision.draft_result_hash != _model_hash(draft.result)
            ):
                raise ValueError("saved review metadata mismatch")
        except (ValidationError, ValueError) as exc:
            raise SavedResultValidationError(
                "A saved review decision failed validation."
            ) from exc
        return decision

    def _validated_summary_dependency(
        self,
        analysis: SavedAnalysisRecord,
        summary: SavedSummaryRecord,
    ) -> SavedSummaryRecord:
        stored = self.load_summary_by_id(summary.record_id)
        if stored != summary:
            raise SavedResultValidationError(
                "The recommendation summary dependency does not match its saved row."
            )
        return stored

    def _validated_recommendation_row(
        self,
        row: sqlite3.Row,
        analysis: SavedAnalysisRecord,
        summary: SavedSummaryRecord,
        *,
        expected_key: str | None,
    ) -> SavedRecommendationDraft:
        try:
            record = SavedRecommendationDraft.model_validate_json(row["payload_json"])
            calculated = recommendation_compatibility_key(
                analysis, summary, record.settings, record.runtime_versions
            )
            if (
                record.record_id != row["record_id"]
                or record.report_hash != row["report_hash"]
                or record.analysis_record_id != row["analysis_record_id"]
                or record.summary_record_id != row["summary_record_id"]
                or record.compatibility_key != row["compatibility_key"]
                or (
                    expected_key is not None
                    and record.compatibility_key != expected_key
                )
                or record.compatibility_key != calculated
                or record.analysis_result_hash != _model_hash(analysis.result)
                or record.summary_result_hash != _model_hash(summary.result)
            ):
                raise ValueError("saved recommendation metadata mismatch")
            _validate_recommendation_evidence(
                record.result, analysis.result, record.settings
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise SavedResultValidationError(
                "A saved recommendation draft failed validation and was not reused."
            ) from exc
        return record

    def delete_report(self, report_hash: str) -> DeletionResult:
        """Delete all saved results associated with one report hash."""

        validated_hash = _validated_hash(report_hash)
        try:
            with self._connect() as connection:
                reviews = connection.execute(
                    """
                    SELECT COUNT(*) FROM recommendation_reviews
                    WHERE draft_record_id IN (
                        SELECT record_id FROM recommendation_drafts
                        WHERE report_hash = ?
                    )
                    """,
                    (validated_hash,),
                ).fetchone()[0]
                drafts = connection.execute(
                    "SELECT COUNT(*) FROM recommendation_drafts WHERE report_hash = ?",
                    (validated_hash,),
                ).fetchone()[0]
                summaries = connection.execute(
                    "SELECT COUNT(*) FROM summary_records WHERE report_hash = ?",
                    (validated_hash,),
                ).fetchone()[0]
                analyses = connection.execute(
                    "SELECT COUNT(*) FROM analysis_records WHERE report_hash = ?",
                    (validated_hash,),
                ).fetchone()[0]
                connection.execute(
                    "DELETE FROM analysis_records WHERE report_hash = ?",
                    (validated_hash,),
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                "Saved report results could not be deleted."
            ) from exc
        return DeletionResult(
            analyses_deleted=analyses,
            summaries_deleted=summaries,
            recommendation_drafts_deleted=drafts,
            review_decisions_deleted=reviews,
        )

    def _validated_analysis_row(
        self,
        row: sqlite3.Row,
        *,
        expected_key: str | None,
    ) -> SavedAnalysisRecord:
        try:
            record = SavedAnalysisRecord.model_validate_json(row["payload_json"])
            if (
                record.record_id != row["record_id"]
                or record.report_hash != row["report_hash"]
                or record.compatibility_key != row["compatibility_key"]
                or (
                    expected_key is not None
                    and record.compatibility_key != expected_key
                )
                or record.compatibility_key
                != analysis_compatibility_key(
                    record.report_hash,
                    record.settings,
                    record.runtime_versions,
                )
            ):
                raise ValueError("saved analysis metadata mismatch")
        except (ValidationError, ValueError, TypeError) as exc:
            raise SavedResultValidationError(
                "A saved analysis failed validation and was not reused."
            ) from exc
        return record

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1, DATABASE_SCHEMA_VERSION):
                    raise IncompatibleDatabaseError(
                        "The local results database schema is incompatible."
                    )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_records (
                        record_id TEXT PRIMARY KEY,
                        report_hash TEXT NOT NULL,
                        compatibility_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS analysis_compatibility_idx
                    ON analysis_records(report_hash, compatibility_key, created_at);

                    CREATE TABLE IF NOT EXISTS summary_records (
                        record_id TEXT PRIMARY KEY,
                        report_hash TEXT NOT NULL,
                        analysis_record_id TEXT NOT NULL,
                        compatibility_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (analysis_record_id)
                            REFERENCES analysis_records(record_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS summary_compatibility_idx
                    ON summary_records(analysis_record_id, compatibility_key, created_at);

                    CREATE TABLE IF NOT EXISTS recommendation_drafts (
                        record_id TEXT PRIMARY KEY,
                        report_hash TEXT NOT NULL,
                        analysis_record_id TEXT NOT NULL,
                        summary_record_id TEXT NOT NULL,
                        compatibility_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        version_number INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (analysis_record_id)
                            REFERENCES analysis_records(record_id) ON DELETE CASCADE,
                        FOREIGN KEY (summary_record_id)
                            REFERENCES summary_records(record_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS recommendation_compatibility_idx
                    ON recommendation_drafts(
                        summary_record_id, compatibility_key, version_number
                    );

                    CREATE TABLE IF NOT EXISTS recommendation_reviews (
                        decision_id TEXT PRIMARY KEY,
                        draft_record_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (draft_record_id)
                            REFERENCES recommendation_drafts(record_id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS recommendation_review_idx
                    ON recommendation_reviews(draft_record_id, created_at);
                    """
                )
                connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
        except IncompatibleDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(
                "The local results database could not be initialized."
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


class PersistedResultService:
    """Coordinate exact saved-result reuse without invoking upstream agents."""

    def __init__(
        self,
        store: SQLiteResultStore,
        *,
        runtime_versions: RuntimeVersions | None = None,
    ) -> None:
        self.store = store
        self.runtime_versions = runtime_versions or current_runtime_versions()

    def resolve_analysis(
        self,
        report_hash: str,
        settings: AnalysisGenerationSettings,
        generate: Callable[[], ReportAnalysisResult],
        *,
        force: bool = False,
    ) -> AnalysisResolution:
        """Reuse a compatible analysis, or generate and save a validated one."""

        if not force:
            existing = self.store.load_analysis(
                report_hash,
                settings,
                runtime_versions=self.runtime_versions,
            )
            if existing is not None:
                return AnalysisResolution(record=existing, reused=True)
        result = generate()
        record = self.store.save_analysis(
            report_hash,
            settings,
            result,
            runtime_versions=self.runtime_versions,
        )
        return AnalysisResolution(record=record, reused=False)

    def resolve_summary(
        self,
        analysis: SavedAnalysisRecord,
        settings: SummaryGenerationSettings,
        generate: Callable[[], ReportSummaryResult],
        *,
        force: bool = False,
    ) -> SummaryResolution:
        """Reuse or generate a summary without any path that reruns analysis."""

        if not force:
            existing = self.store.load_summary(
                analysis,
                settings,
                runtime_versions=self.runtime_versions,
            )
            if existing is not None:
                return SummaryResolution(record=existing, reused=True)
        result = generate()
        record = self.store.save_summary(
            analysis,
            settings,
            result,
            runtime_versions=self.runtime_versions,
        )
        return SummaryResolution(record=record, reused=False)

    def resolve_recommendation(
        self,
        analysis: SavedAnalysisRecord,
        summary: SavedSummaryRecord,
        settings: RecommendationGenerationSettings,
        generate: Callable[[], RecommendationDraftResult],
        *,
        force: bool = False,
    ) -> RecommendationResolution:
        """Reuse or generate a draft without rerunning analysis or summary."""

        if not force:
            existing = self.store.load_recommendation(
                analysis,
                summary,
                settings,
                runtime_versions=self.runtime_versions,
            )
            if existing is not None:
                return RecommendationResolution(record=existing, reused=True)
        result = generate()
        record = self.store.save_recommendation(
            analysis,
            summary,
            settings,
            result,
            runtime_versions=self.runtime_versions,
        )
        return RecommendationResolution(record=record, reused=False)


def analysis_compatibility_key(
    report_hash: str,
    settings: AnalysisGenerationSettings,
    runtime_versions: RuntimeVersions,
) -> str:
    """Hash all declared inputs needed for safe analysis reuse."""

    return _canonical_hash(
        {
            "report_hash": _validated_hash(report_hash),
            "settings": settings.model_dump(mode="json"),
            "runtime_versions": runtime_versions.model_dump(mode="json"),
        }
    )


def summary_compatibility_key(
    analysis: SavedAnalysisRecord,
    settings: SummaryGenerationSettings,
    runtime_versions: RuntimeVersions,
) -> str:
    """Hash the exact analysis dependency and summary-generation context."""

    return _canonical_hash(
        {
            "analysis_record_id": analysis.record_id,
            "analysis_result_hash": _model_hash(analysis.result),
            "settings": settings.model_dump(mode="json"),
            "runtime_versions": runtime_versions.model_dump(mode="json"),
        }
    )


def recommendation_compatibility_key(
    analysis: SavedAnalysisRecord,
    summary: SavedSummaryRecord,
    settings: RecommendationGenerationSettings,
    runtime_versions: RuntimeVersions,
) -> str:
    """Hash exact upstream records, generation settings, and guideline identity."""

    if summary.analysis_record_id != analysis.record_id:
        raise SavedResultValidationError(
            "The recommendation dependencies do not reference the same analysis."
        )
    return _canonical_hash(
        {
            "analysis_record_id": analysis.record_id,
            "analysis_result_hash": _model_hash(analysis.result),
            "summary_record_id": summary.record_id,
            "summary_result_hash": _model_hash(summary.result),
            "settings": settings.model_dump(mode="json"),
            "runtime_versions": runtime_versions.model_dump(mode="json"),
        }
    )


def _validate_summary_evidence(
    summary: ReportSummaryResult,
    analysis: ReportAnalysisResult,
) -> None:
    source_by_id: dict[str, ReportFinding | NarrativeFinding] = {
        item.finding_id: item
        for item in [*analysis.findings, *analysis.narrative_findings]
    }
    items = [
        summary.overview,
        *summary.important_findings,
        *summary.repeat_result_changes,
        *summary.narrative_context,
        *summary.uncertainties,
    ]
    for item in items:
        evidence_ids = [entry.finding_id for entry in item.source_evidence]
        if item.finding_ids != evidence_ids:
            raise SavedResultValidationError(
                "A summary evidence relationship failed validation."
            )
        if item.source_pages != list(
            dict.fromkeys(entry.source_page for entry in item.source_evidence)
        ):
            raise SavedResultValidationError(
                "A summary evidence relationship failed validation."
            )
        for evidence in item.source_evidence:
            source = source_by_id.get(evidence.finding_id)
            if source is None or evidence != _expected_summary_evidence(source):
                raise SavedResultValidationError(
                    "A summary source failed validation and was not saved."
                )


def _validate_recommendation_evidence(
    draft: RecommendationDraftResult,
    analysis: ReportAnalysisResult,
    settings: RecommendationGenerationSettings,
) -> None:
    source_by_id: dict[str, ReportFinding | NarrativeFinding] = {
        item.finding_id: item
        for item in [*analysis.findings, *analysis.narrative_findings]
    }
    passages = {item.passage_id: item for item in draft.retrieved_passages}
    if len(passages) != len(draft.retrieved_passages):
        raise SavedResultValidationError(
            "Recommendation passage provenance contains duplicate IDs."
        )
    allowed_hashes = set(settings.guideline_document_hashes)
    if any(item.document_hash not in allowed_hashes for item in passages.values()):
        raise SavedResultValidationError(
            "Recommendation passage provenance does not match the guideline index."
        )
    for suggestion in draft.suggestions:
        finding_ids = [item.finding_id for item in suggestion.patient_evidence]
        passage_ids = [item.passage_id for item in suggestion.guideline_evidence]
        if (
            suggestion.finding_ids != finding_ids
            or suggestion.passage_ids != passage_ids
        ):
            raise SavedResultValidationError(
                "Recommendation evidence relationships failed validation."
            )
        for evidence in suggestion.patient_evidence:
            source = source_by_id.get(evidence.finding_id)
            if source is None or evidence != _expected_recommendation_evidence(source):
                raise SavedResultValidationError(
                    "Recommendation patient evidence failed validation."
                )
        for evidence in suggestion.guideline_evidence:
            if passages.get(evidence.passage_id) != evidence:
                raise SavedResultValidationError(
                    "Recommendation guideline provenance failed validation."
                )


def _expected_recommendation_evidence(
    source: ReportFinding | NarrativeFinding,
) -> RecommendationFindingEvidence:
    if isinstance(source, ReportFinding):
        return RecommendationFindingEvidence(
            finding_id=source.finding_id,
            display_name=source.name,
            source_page=source.source_page,
            supporting_excerpt=source.supporting_excerpt,
            reported_date=source.reported_date,
            specimen_event=source.specimen_event,
            reported_value=source.reported_value,
            reported_unit=source.reported_unit,
            assessment=source.assessment.value,
            uncertainty_notes=source.review_reasons,
        )
    notes = []
    if source.status.value != "reported":
        name = source.category.value.replace("_", " ").capitalize()
        status = source.status.value.replace("_", " ")
        notes.append(f"{name} is {status}.")
    return RecommendationFindingEvidence(
        finding_id=source.finding_id,
        display_name=source.category.value.replace("_", " ").capitalize(),
        source_page=source.source_page,
        supporting_excerpt=source.supporting_excerpt,
        uncertainty_notes=notes,
    )


def _expected_summary_evidence(
    source: ReportFinding | NarrativeFinding,
) -> SummarySourceEvidence:
    if isinstance(source, ReportFinding):
        return SummarySourceEvidence(
            finding_id=source.finding_id,
            source_page=source.source_page,
            supporting_excerpt=source.supporting_excerpt,
            display_name=source.name,
            reported_date=source.reported_date,
            specimen_event=source.specimen_event,
            reported_value=source.reported_value,
            reported_unit=source.reported_unit,
            reference_text=source.reference_text,
            reference_range=source.reference_range,
            reference_unit=source.reference_unit,
            reference_supporting_excerpt=source.reference_supporting_excerpt,
            reference_absence_statement=source.reference_absence_statement,
            assessment=source.assessment.value,
            applicability_status=source.applicability_status.value,
            applicability_evidence=[
                SummaryApplicabilityEvidence(
                    text=context.text,
                    source_page=context.source_page,
                    supporting_excerpt=context.supporting_excerpt,
                    specimen_event=context.specimen_event,
                )
                for context in source.applicability_contexts
            ],
            rejected_applicability_contexts=(source.rejected_applicability_contexts),
            review_reasons=source.review_reasons,
        )
    return SummarySourceEvidence(
        finding_id=source.finding_id,
        source_page=source.source_page,
        supporting_excerpt=source.supporting_excerpt,
        display_name=source.category.value.replace("_", " ").capitalize(),
        narrative_statement=source.statement,
        narrative_status=source.status.value,
        attribution=source.attribution,
    )


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _model_schema_hash(model: type[BaseModel]) -> str:
    return _canonical_hash(model.model_json_schema())


def _model_hash(model: BaseModel) -> str:
    return _canonical_hash(model.model_dump(mode="json"))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_hash(value: str) -> str:
    if len(value) != _HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("Expected a lowercase SHA-256 hash.")
    return value
