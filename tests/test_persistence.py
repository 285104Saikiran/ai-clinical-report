"""Offline tests for validated local SQLite result persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from clinical_report_assistant.persistence import (
    AnalysisGenerationSettings,
    PersistedResultService,
    RuntimeVersions,
    SavedResultValidationError,
    SQLiteResultStore,
    SummaryGenerationSettings,
    report_sha256,
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


class _SummaryRunner:
    def __init__(self, output: ExtractedReportSummary) -> None:
        self.output = output

    def summarize(self, analysis: ReportAnalysisResult) -> Any:
        return self.output


class _Counter:
    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        return self.factory()


@pytest.fixture
def versions() -> RuntimeVersions:
    return RuntimeVersions(
        application="test-app-1",
        python="3.12.test",
        pydantic="test-pydantic-1",
        crewai="test-crewai-1",
        google_genai="test-google-genai-1",
    )


@pytest.fixture
def analysis_settings() -> AnalysisGenerationSettings:
    return AnalysisGenerationSettings(
        model="synthetic-gemini-model",
        max_output_tokens=8192,
        timeout_seconds=60,
        schema_hash="a" * 64,
    )


@pytest.fixture
def summary_settings() -> SummaryGenerationSettings:
    return SummaryGenerationSettings(
        model="synthetic-gemini-model",
        max_output_tokens=4096,
        timeout_seconds=60,
        schema_hash="b" * 64,
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteResultStore:
    return SQLiteResultStore(tmp_path / "local-results.sqlite3")


def _analysis(
    *, filename: str = "same-name.pdf", value: str = "90"
) -> ReportAnalysisResult:
    finding = ReportFinding(
        name="Synthetic glucose",
        reported_value=value,
        reported_date="2026-01-01",
        reported_unit="mg/dL",
        reference_text="70-99 mg/dL",
        reference_range="70-99",
        reference_unit="mg/dL",
        specimen_event="synthetic specimen one",
        applicability_status=ApplicabilityStatus.DOCUMENTED,
        finding_kind=FindingKind.PATIENT_MEASUREMENT,
        finding_id=f"finding-{value}",
        source_page=1,
        supporting_excerpt=(
            f"Synthetic glucose {value} mg/dL; reference range 70-99 mg/dL."
        ),
        assessment=FindingAssessment.WITHIN_RANGE,
        review_required=False,
    )
    return ReportAnalysisResult(filename=filename, findings=[finding])


def _summary(analysis: ReportAnalysisResult) -> ReportSummaryResult:
    finding_id = analysis.findings[0].finding_id
    selection = ExtractedReportSummary(
        overview_finding_ids=[finding_id],
        important_finding_ids=[],
        narrative_finding_ids=[],
        uncertainty_finding_ids=[],
    )
    return SummaryAgent(runner=_SummaryRunner(selection)).summarize(analysis)


def test_compatible_analysis_and_summary_reuse_without_generation_calls(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    service = PersistedResultService(store, runtime_versions=versions)
    report_hash = report_sha256(b"synthetic report bytes")
    analysis_counter = _Counter(_analysis)

    first = service.resolve_analysis(report_hash, analysis_settings, analysis_counter)
    second = service.resolve_analysis(report_hash, analysis_settings, analysis_counter)

    assert analysis_counter.calls == 1
    assert not first.reused
    assert second.reused
    assert second.record.record_id == first.record.record_id

    summary_counter = _Counter(lambda: _summary(first.record.result))
    first_summary = service.resolve_summary(
        first.record,
        summary_settings,
        summary_counter,
    )
    second_summary = service.resolve_summary(
        first.record,
        summary_settings,
        summary_counter,
    )

    assert summary_counter.calls == 1
    assert not first_summary.reused
    assert second_summary.reused
    assert second_summary.record.record_id == first_summary.record.record_id


def test_report_hash_ignores_filename_but_distinguishes_different_bytes(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
) -> None:
    service = PersistedResultService(store, runtime_versions=versions)
    first_bytes = b"synthetic report version one"
    second_bytes = b"synthetic report version two"
    counter = _Counter(lambda: _analysis(filename="same-name.pdf"))

    first = service.resolve_analysis(
        report_sha256(first_bytes),
        analysis_settings,
        counter,
    )
    second = service.resolve_analysis(
        report_sha256(second_bytes),
        analysis_settings,
        counter,
    )

    assert first.record.report_hash != second.record.report_hash
    assert counter.calls == 2

    identical_bytes_counter = _Counter(lambda: _analysis(filename="renamed-copy.pdf"))
    renamed = service.resolve_analysis(
        report_sha256(first_bytes),
        analysis_settings,
        identical_bytes_counter,
    )
    assert renamed.reused
    assert identical_bytes_counter.calls == 0
    assert renamed.record.result.filename == "same-name.pdf"


def test_setting_and_version_changes_are_never_silently_reused(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    report_hash = report_sha256(b"synthetic report")
    record = store.save_analysis(
        report_hash,
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    summary = store.save_summary(
        record,
        summary_settings,
        _summary(record.result),
        runtime_versions=versions,
    )

    assert (
        store.load_analysis(
            report_hash,
            analysis_settings.model_copy(update={"model": "different-model"}),
            runtime_versions=versions,
        )
        is None
    )
    assert (
        store.load_analysis(
            report_hash,
            analysis_settings.model_copy(update={"prompt_version": "prompt-v-next"}),
            runtime_versions=versions,
        )
        is None
    )
    changed_versions = versions.model_copy(update={"crewai": "test-crewai-2"})
    assert (
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=changed_versions,
        )
        is None
    )
    assert (
        store.load_summary(
            record,
            summary_settings.model_copy(update={"max_output_tokens": 8192}),
            runtime_versions=versions,
        )
        is None
    )
    assert (
        store.load_summary(
            record,
            summary_settings,
            runtime_versions=versions,
        )
        == summary
    )


def test_previous_validation_versions_remain_saved_but_are_not_reused(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    report_hash = report_sha256(b"synthetic versioned report")
    legacy_analysis_settings = analysis_settings.model_copy(
        update={
            "prompt_version": "report-analysis-prompt-v1",
            "schema_version": "report-analysis-schema-v1",
        }
    )
    legacy_summary_settings = summary_settings.model_copy(
        update={
            "prompt_version": "summary-prompt-v1",
            "schema_version": "summary-schema-v1",
        }
    )
    legacy_analysis = store.save_analysis(
        report_hash,
        legacy_analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    legacy_summary = store.save_summary(
        legacy_analysis,
        legacy_summary_settings,
        _summary(legacy_analysis.result),
        runtime_versions=versions,
    )

    assert (
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=versions,
        )
        is None
    )
    assert (
        store.load_summary(
            legacy_analysis,
            summary_settings,
            runtime_versions=versions,
        )
        is None
    )
    assert store.load_analysis_by_id(legacy_analysis.record_id) == legacy_analysis
    assert legacy_summary.analysis_record_id == legacy_analysis.record_id


def test_summary_is_tied_to_exact_saved_analysis(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    service = PersistedResultService(store, runtime_versions=versions)
    report_hash = report_sha256(b"synthetic report")
    first = service.resolve_analysis(
        report_hash,
        analysis_settings,
        _analysis,
        force=True,
    )
    service.resolve_summary(
        first.record,
        summary_settings,
        lambda: _summary(first.record.result),
        force=True,
    )
    second = service.resolve_analysis(
        report_hash,
        analysis_settings,
        lambda: _analysis(value="91"),
        force=True,
    )

    assert first.record.record_id != second.record.record_id
    assert (
        store.load_summary(
            second.record,
            summary_settings,
            runtime_versions=versions,
        )
        is None
    )


def test_invalid_saved_json_is_rejected_through_pydantic(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
) -> None:
    report_hash = report_sha256(b"synthetic report")
    record = store.save_analysis(
        report_hash,
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE analysis_records SET payload_json = ? WHERE record_id = ?",
            ("{result:", record.record_id),
        )

    with pytest.raises(
        SavedResultValidationError,
        match="failed validation",
    ):
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=versions,
        )


def test_tampered_compatibility_metadata_is_rejected(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
) -> None:
    report_hash = report_sha256(b"synthetic report")
    record = store.save_analysis(
        report_hash,
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM analysis_records WHERE record_id = ?",
                (record.record_id,),
            ).fetchone()[0]
        )
        payload["settings"]["model"] = "tampered-model"
        connection.execute(
            "UPDATE analysis_records SET payload_json = ? WHERE record_id = ?",
            (json.dumps(payload), record.record_id),
        )

    with pytest.raises(
        SavedResultValidationError,
        match="failed validation",
    ):
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=versions,
        )


def test_unsupported_summary_evidence_is_not_saved(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    analysis = store.save_analysis(
        report_sha256(b"synthetic report"),
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    summary = _summary(analysis.result)
    bad_evidence = summary.overview.source_evidence[0].model_copy(
        update={"supporting_excerpt": "unsupported synthetic excerpt"}
    )
    bad_overview = summary.overview.model_copy(
        update={"source_evidence": [bad_evidence]}
    )
    unsupported = summary.model_copy(update={"overview": bad_overview})

    with pytest.raises(SavedResultValidationError, match="source failed validation"):
        store.save_summary(
            analysis,
            summary_settings,
            unsupported,
            runtime_versions=versions,
        )


def test_confirmed_delete_removes_report_analysis_and_cascaded_summary(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    report_hash = report_sha256(b"synthetic report")
    analysis = store.save_analysis(
        report_hash,
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )
    store.save_summary(
        analysis,
        summary_settings,
        _summary(analysis.result),
        runtime_versions=versions,
    )

    deleted = store.delete_report(report_hash)

    assert deleted.analyses_deleted == 1
    assert deleted.summaries_deleted == 1
    assert (
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=versions,
        )
        is None
    )
    with pytest.raises(SavedResultValidationError, match="unavailable"):
        store.load_analysis_by_id(analysis.record_id)


def test_failed_forced_regeneration_preserves_previous_valid_records(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
    summary_settings: SummaryGenerationSettings,
) -> None:
    service = PersistedResultService(store, runtime_versions=versions)
    report_hash = report_sha256(b"synthetic report")
    analysis = service.resolve_analysis(
        report_hash,
        analysis_settings,
        _analysis,
        force=True,
    )
    summary = service.resolve_summary(
        analysis.record,
        summary_settings,
        lambda: _summary(analysis.record.result),
        force=True,
    )

    def fail() -> Any:
        raise RuntimeError("synthetic generation failure")

    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        service.resolve_analysis(
            report_hash,
            analysis_settings,
            fail,
            force=True,
        )
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        service.resolve_summary(
            analysis.record,
            summary_settings,
            fail,
            force=True,
        )

    assert (
        store.load_analysis(
            report_hash,
            analysis_settings,
            runtime_versions=versions,
        )
        == analysis.record
    )
    assert (
        store.load_summary(
            analysis.record,
            summary_settings,
            runtime_versions=versions,
        )
        == summary.record
    )


def test_database_payload_contains_no_pdf_bytes_or_api_key(
    store: SQLiteResultStore,
    versions: RuntimeVersions,
    analysis_settings: AnalysisGenerationSettings,
) -> None:
    raw_pdf_marker = b"%PDF synthetic private bytes"
    api_key_marker = "synthetic-secret-api-key"
    store.save_analysis(
        report_sha256(raw_pdf_marker),
        analysis_settings,
        _analysis(),
        runtime_versions=versions,
    )

    database_bytes = store.path.read_bytes()
    assert raw_pdf_marker not in database_bytes
    assert api_key_marker.encode() not in database_bytes
