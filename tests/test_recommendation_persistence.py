"""Offline persistence and human-review tests for recommendation drafts."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from clinical_report_assistant.guideline_kb import RetrievedPassage
from clinical_report_assistant.persistence import (
    AnalysisGenerationSettings,
    PersistedResultService,
    RecommendationGenerationSettings,
    RuntimeVersions,
    SQLiteResultStore,
    SummaryGenerationSettings,
    report_sha256,
)
from clinical_report_assistant.recommendation_agent import (
    ExtractedFindingCitation,
    ExtractedPassageCitation,
    ExtractedRecommendationDraft,
    ExtractedSuggestion,
    RecommendationAgent,
    RecommendationDraftResult,
    create_edited_draft,
)
from clinical_report_assistant.report_analysis_agent import (
    ApplicabilityStatus,
    FindingAssessment,
    FindingKind,
    ReportAnalysisResult,
    ReportFinding,
)
from clinical_report_assistant.summary_agent import (
    ExtractedReportSummary,
    ReportSummaryResult,
    SummaryAgent,
)

_DOCUMENT_HASH = "a" * 64


class Counter:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.factory()


class SummaryRunner:
    def summarize(self, analysis: ReportAnalysisResult) -> ExtractedReportSummary:
        finding_id = analysis.findings[0].finding_id
        return ExtractedReportSummary(
            overview_finding_ids=[finding_id],
            important_finding_ids=[finding_id],
        )


class RecommendationRunner:
    def __init__(self, passage: RetrievedPassage) -> None:
        self.passage = passage

    def recommend(
        self,
        analysis: ReportAnalysisResult,
        summary: ReportSummaryResult,
        passages: list[RetrievedPassage],
    ) -> ExtractedRecommendationDraft:
        finding = analysis.findings[0]
        return ExtractedRecommendationDraft(
            suggestions=[
                ExtractedSuggestion(
                    text="Review the synthetic glucose result with fasting status.",
                    patient_evidence=[
                        ExtractedFindingCitation(
                            finding_id=finding.finding_id,
                            quoted_excerpt="Glucose result: 118 mg/dL",
                        )
                    ],
                    guideline_evidence=[
                        ExtractedPassageCitation(
                            passage_id=self.passage.chunk_id,
                            quoted_excerpt="review glucose with fasting status",
                        )
                    ],
                )
            ]
        )


@pytest.fixture
def versions() -> RuntimeVersions:
    return RuntimeVersions(
        application="test-app",
        python="3.12",
        pydantic="test",
        crewai="test",
        google_genai="test",
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteResultStore:
    return SQLiteResultStore(tmp_path / "results.sqlite3")


@pytest.fixture
def settings() -> tuple[
    AnalysisGenerationSettings,
    SummaryGenerationSettings,
    RecommendationGenerationSettings,
]:
    return (
        AnalysisGenerationSettings(
            model="synthetic-model",
            max_output_tokens=8192,
            timeout_seconds=60,
            schema_hash="1" * 64,
        ),
        SummaryGenerationSettings(
            model="synthetic-model",
            max_output_tokens=4096,
            timeout_seconds=60,
            schema_hash="2" * 64,
        ),
        RecommendationGenerationSettings(
            model="synthetic-model",
            max_output_tokens=4096,
            timeout_seconds=60,
            schema_hash="3" * 64,
            retrieval_top_k=4,
            max_retrieval_queries=3,
            guideline_index_fingerprint="4" * 64,
            guideline_document_hashes=[_DOCUMENT_HASH],
        ),
    )


def analysis() -> ReportAnalysisResult:
    finding = ReportFinding(
        name="Glucose",
        reported_value="118",
        reported_date="2026-01-01",
        reported_unit="mg/dL",
        reference_text="70-99 mg/dL",
        reference_range="70-99",
        reference_unit="mg/dL",
        specimen_event="initial",
        applicability_status=ApplicabilityStatus.REVIEW_REQUIRED,
        finding_kind=FindingKind.PATIENT_MEASUREMENT,
        finding_id="glucose",
        source_page=1,
        supporting_excerpt="Glucose result: 118 mg/dL; fasting status unknown.",
        assessment=FindingAssessment.ABOVE_RANGE,
        review_required=True,
        review_reasons=["Fasting status is unknown."],
    )
    return ReportAnalysisResult(filename="synthetic.pdf", findings=[finding])


def passage() -> RetrievedPassage:
    return RetrievedPassage(
        text=(
            "SYNTHETIC TRAINING GUIDELINE - NOT MEDICAL ADVICE. "
            "For this exercise, review glucose with fasting status."
        ),
        document_name="synthetic-glucose.pdf",
        document_hash=_DOCUMENT_HASH,
        page_number=1,
        chunk_id="chunk-glucose",
    )


def draft(
    analysis_result: ReportAnalysisResult, summary_result: ReportSummaryResult
) -> RecommendationDraftResult:
    source = passage()
    return RecommendationAgent(runner=RecommendationRunner(source)).recommend(
        analysis_result, summary_result, [source]
    )


def saved_dependencies(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
):
    analysis_settings, summary_settings, _ = settings
    saved_analysis = store.save_analysis(
        report_sha256(b"synthetic report"),
        analysis_settings,
        analysis(),
        runtime_versions=versions,
    )
    summary_result = SummaryAgent(runner=SummaryRunner()).summarize(
        saved_analysis.result
    )
    saved_summary = store.save_summary(
        saved_analysis,
        summary_settings,
        summary_result,
        runtime_versions=versions,
    )
    return saved_analysis, saved_summary


def test_restart_reuses_exact_draft_with_zero_agent_calls(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
) -> None:
    saved_analysis, saved_summary = saved_dependencies(store, versions, settings)
    recommendation_settings = settings[2]
    counter = Counter(lambda: draft(saved_analysis.result, saved_summary.result))
    first_service = PersistedResultService(store, runtime_versions=versions)
    first = first_service.resolve_recommendation(
        saved_analysis, saved_summary, recommendation_settings, counter
    )
    restarted = PersistedResultService(
        SQLiteResultStore(store.path), runtime_versions=versions
    )
    second = restarted.resolve_recommendation(
        saved_analysis, saved_summary, recommendation_settings, counter
    )

    assert counter.calls == 1
    assert not first.reused
    assert second.reused
    assert second.record.record_id == first.record.record_id


def test_guideline_fingerprint_change_invalidates_reuse(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
) -> None:
    saved_analysis, saved_summary = saved_dependencies(store, versions, settings)
    service = PersistedResultService(store, runtime_versions=versions)
    first_counter = Counter(lambda: draft(saved_analysis.result, saved_summary.result))
    service.resolve_recommendation(
        saved_analysis, saved_summary, settings[2], first_counter
    )
    changed = settings[2].model_copy(update={"guideline_index_fingerprint": "5" * 64})
    second_counter = Counter(lambda: draft(saved_analysis.result, saved_summary.result))
    result = service.resolve_recommendation(
        saved_analysis, saved_summary, changed, second_counter
    )

    assert second_counter.calls == 1
    assert not result.reused


def test_review_is_exact_and_edit_resets_to_pending(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
) -> None:
    saved_analysis, saved_summary = saved_dependencies(store, versions, settings)
    original = store.save_recommendation(
        saved_analysis,
        saved_summary,
        settings[2],
        draft(saved_analysis.result, saved_summary.result),
        runtime_versions=versions,
    )
    approval = store.save_review_decision(original, "approved", "Reviewed.")
    assert store.load_latest_review(original) == approval

    edited_result = create_edited_draft(
        original.result, ["Review the result and fasting context together."]
    )
    edited = store.save_edited_recommendation(original, edited_result)

    assert edited.version_number == original.version_number + 1
    assert edited.parent_draft_id == original.record_id
    assert store.load_latest_review(edited) is None
    rejection = store.save_review_decision(edited, "rejected", "Needs revision.")
    assert rejection.decision == "rejected"
    assert store.load_latest_review(original) == approval


def test_failed_regeneration_preserves_prior_valid_draft(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
) -> None:
    saved_analysis, saved_summary = saved_dependencies(store, versions, settings)
    service = PersistedResultService(store, runtime_versions=versions)
    current = service.resolve_recommendation(
        saved_analysis,
        saved_summary,
        settings[2],
        lambda: draft(saved_analysis.result, saved_summary.result),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        service.resolve_recommendation(
            saved_analysis,
            saved_summary,
            settings[2],
            lambda: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
            force=True,
        )

    loaded = store.load_recommendation(
        saved_analysis, saved_summary, settings[2], runtime_versions=versions
    )
    assert loaded == current.record


def test_report_deletion_cascades_drafts_and_reviews(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    settings: tuple[
        AnalysisGenerationSettings,
        SummaryGenerationSettings,
        RecommendationGenerationSettings,
    ],
) -> None:
    saved_analysis, saved_summary = saved_dependencies(store, versions, settings)
    saved_draft = store.save_recommendation(
        saved_analysis,
        saved_summary,
        settings[2],
        draft(saved_analysis.result, saved_summary.result),
        runtime_versions=versions,
    )
    store.save_review_decision(saved_draft, "approved")

    deleted = store.delete_report(saved_analysis.report_hash)

    assert deleted.recommendation_drafts_deleted == 1
    assert deleted.review_decisions_deleted == 1


def test_schema_one_database_is_migrated_without_dropping_tables(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE analysis_records (
                record_id TEXT PRIMARY KEY,
                report_hash TEXT NOT NULL,
                compatibility_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE summary_records (
                record_id TEXT PRIMARY KEY,
                report_hash TEXT NOT NULL,
                analysis_record_id TEXT NOT NULL,
                compatibility_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (analysis_record_id)
                    REFERENCES analysis_records(record_id) ON DELETE CASCADE
            );
            PRAGMA user_version = 1;
            """
        )

    SQLiteResultStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {row[1] for row in connection.execute("PRAGMA table_list")}
    assert "analysis_records" in tables
    assert "summary_records" in tables
    assert "recommendation_drafts" in tables
    assert "recommendation_reviews" in tables
