"""Offline tests for the second project agent: the Summary Agent."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from clinical_report_assistant.config import GeminiConfig
from clinical_report_assistant.report_analysis_agent import (
    ApplicabilityStatus,
    ExtractedApplicabilityContext,
    FindingAssessment,
    FindingKind,
    NarrativeCategory,
    NarrativeFinding,
    NarrativeStatus,
    ReportAnalysisResult,
    ReportFinding,
    ReviewWarningCategory,
    ReviewWarningGroup,
)
from clinical_report_assistant.summary_agent import (
    MAX_REPEAT_GROUPS,
    MAX_SUMMARY_REFERENCES,
    SUMMARY_DISCLAIMER,
    CrewAISummaryRunner,
    ExtractedReportSummary,
    RepeatResultSelection,
    SanitizedSummaryGeminiCompletion,
    SummaryAgent,
    SummaryCrewAIComponents,
    SummaryIncompleteOutputError,
    SummaryOutputError,
    SummaryReferenceValidationError,
    _build_repeat_groups,
    _redact_dependency_logs,
    _validate_assembled_repeat_groups,
    build_compact_finding_rows,
    build_compact_uncertainty_rows,
    safe_summary_error_message,
)


class StubRunner:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.seen_analysis: ReportAnalysisResult | None = None

    def summarize(self, analysis: ReportAnalysisResult) -> Any:
        self.seen_analysis = analysis
        return self.output


def _measurement(
    finding_id: str,
    *,
    value: str,
    assessment: FindingAssessment,
    date: str | None,
    page: int,
    review_reasons: list[str] | None = None,
    kind: FindingKind = FindingKind.PATIENT_MEASUREMENT,
) -> ReportFinding:
    reasons = review_reasons or []
    return ReportFinding(
        name="Glucose",
        reported_value=value,
        reported_date=date,
        reported_unit="mg/dL",
        reference_text="70-99 mg/dL",
        reference_range="70-99",
        reference_unit="mg/dL",
        specimen_event=(f"synthetic specimen {date}" if date else None),
        applicability_status=(
            ApplicabilityStatus.REVIEW_REQUIRED
            if reasons
            else ApplicabilityStatus.DOCUMENTED
        ),
        applicability_review_reasons=reasons,
        finding_kind=kind,
        finding_id=finding_id,
        source_page=page,
        supporting_excerpt="; ".join(
            part
            for part in (
                f"Glucose {value} mg/dL",
                f"date {date}" if date else None,
                "reference range 70-99 mg/dL",
            )
            if part is not None
        )
        + ".",
        assessment=assessment,
        review_required=bool(reasons),
        source_review_reasons=[],
        numeric_review_reasons=[],
        review_reasons=reasons,
    )


def _narrative(
    finding_id: str,
    *,
    category: NarrativeCategory,
    statement: str,
    status: NarrativeStatus,
    attribution: str | None,
    page: int,
) -> NarrativeFinding:
    attribution_text = attribution or ""
    return NarrativeFinding(
        category=category,
        statement=statement,
        attribution=attribution,
        status=status,
        finding_id=finding_id,
        source_page=page,
        supporting_excerpt=f"{statement} {attribution_text}".strip(),
    )


def _analysis() -> ReportAnalysisResult:
    unknown_fasting = "Fasting status is unknown; range applicability requires review."
    initial = _measurement(
        "measurement-initial",
        value="118",
        assessment=FindingAssessment.ABOVE_RANGE,
        date="2026-01-01",
        page=1,
        review_reasons=[unknown_fasting],
    )
    repeat = _measurement(
        "measurement-repeat",
        value="96",
        assessment=FindingAssessment.WITHIN_RANGE,
        date="2026-02-01",
        page=2,
    )
    symptom = _narrative(
        "narrative-symptom",
        category=NarrativeCategory.SYMPTOM,
        statement="Participant reported occasional thirst.",
        status=NarrativeStatus.REPORTED,
        attribution="participant reported",
        page=3,
    )
    follow_up = _narrative(
        "narrative-follow-up",
        category=NarrativeCategory.MISSING_FOLLOW_UP,
        statement="Follow-up date was not recorded.",
        status=NarrativeStatus.NOT_RECORDED,
        attribution=None,
        page=4,
    )
    control = _measurement(
        "measurement-training-control",
        value="999",
        assessment=FindingAssessment.ABOVE_RANGE,
        date="2026-01-01",
        page=5,
        kind=FindingKind.TRAINING_CONTROL,
    )
    return ReportAnalysisResult(
        filename="synthetic-longitudinal.pdf",
        findings=[initial, repeat],
        training_controls=[control],
        narrative_findings=[symptom, follow_up],
        review_warnings=[
            ReviewWarningGroup(
                category=ReviewWarningCategory.APPLICABILITY,
                reason=unknown_fasting,
                finding_ids=[initial.finding_id],
                source_pages=[1],
                finding_references=["Glucose (page 1, 2026-01-01)"],
            )
        ],
    )


def _long_analysis() -> ReportAnalysisResult:
    base = _analysis()
    initial, repeat = base.findings
    fasting_context = ExtractedApplicabilityContext(
        text="Fasting status was documented for the repeat specimen.",
        source_page=2,
        supporting_excerpt=("Fasting status was documented for the repeat specimen."),
        specimen_event=repeat.specimen_event,
    )
    repeat = repeat.model_copy(
        update={
            "applicability_contexts": [fasting_context],
            "applicability_status": ApplicabilityStatus.DOCUMENTED,
        }
    )

    no_range_reason = (
        "No reference range was supplied; numeric assessment was not performed."
    )
    pulse = _measurement(
        "measurement-pulse-no-range",
        value="92",
        assessment=FindingAssessment.NOT_ASSESSED,
        date="2026-01-01",
        page=3,
    ).model_copy(
        update={
            "name": "Pulse",
            "reported_unit": "beats/min",
            "reference_text": None,
            "reference_range": None,
            "reference_unit": None,
            "specimen_event": "intake vital signs",
            "applicability_status": ApplicabilityStatus.NOT_STATED,
            "numeric_review_reasons": [no_range_reason],
            "review_reasons": [no_range_reason],
            "review_required": True,
            "supporting_excerpt": "Pulse: 92 beats/min. No reference range supplied.",
        }
    )
    respiration = pulse.model_copy(
        update={
            "name": "Respiration",
            "finding_id": "measurement-respiration-no-range",
            "reported_value": "18",
            "reported_unit": "breaths/min",
            "source_page": 3,
            "supporting_excerpt": (
                "Respiration: 18 breaths/min. No reference range supplied."
            ),
        }
    )

    mismatch_reason = (
        "The reported and reference units do not match; numeric assessment was "
        "not performed."
    )
    mismatch = _measurement(
        "measurement-unit-mismatch",
        value="5",
        assessment=FindingAssessment.NOT_ASSESSED,
        date="2026-01-02",
        page=4,
    ).model_copy(
        update={
            "name": "Synthetic marker",
            "reported_unit": "mg/dL",
            "reference_text": "1-3 mmol/L",
            "reference_range": "1-3",
            "reference_unit": "mmol/L",
            "numeric_review_reasons": [mismatch_reason],
            "review_reasons": [mismatch_reason],
            "review_required": True,
            "supporting_excerpt": (
                "Synthetic marker: 5 mg/dL. Reference range: 1-3 mmol/L."
            ),
        }
    )

    unsupported_reason = (
        "The supplied reference range format is unsupported; numeric assessment "
        "was not performed."
    )
    unsupported = _measurement(
        "measurement-unsupported-range",
        value="7",
        assessment=FindingAssessment.NOT_ASSESSED,
        date="2026-01-03",
        page=5,
    ).model_copy(
        update={
            "name": "Age-specific marker",
            "reported_unit": "units",
            "reference_text": "see age-specific worksheet",
            "reference_range": None,
            "reference_unit": None,
            "numeric_review_reasons": [unsupported_reason],
            "review_reasons": [unsupported_reason],
            "review_required": True,
            "supporting_excerpt": (
                "Age-specific marker: 7 units. Reference: see age-specific worksheet."
            ),
        }
    )

    symptom = base.narrative_findings[0]
    medication = _narrative(
        "narrative-medication",
        category=NarrativeCategory.MEDICATION_UNCERTAINTY,
        statement="Participant reported that the medication list was uncertain.",
        status=NarrativeStatus.UNCERTAIN,
        attribution="participant reported",
        page=6,
    )
    allergy = _narrative(
        "narrative-allergy",
        category=NarrativeCategory.ALLERGY_VERIFICATION,
        statement="Allergy verification was incomplete.",
        status=NarrativeStatus.VERIFICATION_INCOMPLETE,
        attribution="reconciliation form",
        page=7,
    )
    follow_up = base.narrative_findings[1].model_copy(update={"source_page": 8})
    return ReportAnalysisResult(
        filename="synthetic-eight-page-report.pdf",
        findings=[initial, repeat, pulse, respiration, mismatch, unsupported],
        training_controls=base.training_controls,
        narrative_findings=[symptom, medication, allergy, follow_up],
        review_warnings=base.review_warnings,
    )


def _long_selection() -> ExtractedReportSummary:
    return ExtractedReportSummary(
        overview_finding_ids=["measurement-initial", "narrative-symptom"],
        important_finding_ids=["measurement-initial"],
        repeat_result_groups=[
            RepeatResultSelection(
                finding_ids=["measurement-initial", "measurement-repeat"]
            )
        ],
        narrative_finding_ids=[],
        uncertainty_finding_ids=[],
    )


def _named_measurement(
    finding_id: str,
    *,
    name: str,
    value: str,
    unit: str,
    reference: str,
    reference_range: str,
    date: str | None,
    specimen_event: str,
    page: int,
) -> ReportFinding:
    finding = _measurement(
        finding_id,
        value=value,
        assessment=FindingAssessment.WITHIN_RANGE,
        date=date,
        page=page,
    )
    date_text = f" Date: {date}." if date else ""
    return finding.model_copy(
        update={
            "name": name,
            "reported_unit": unit,
            "reference_text": reference,
            "reference_range": reference_range,
            "reference_unit": unit,
            "specimen_event": specimen_event,
            "supporting_excerpt": (
                f"{name.strip()}: {value} {unit}.{date_text} "
                f"Reference range: {reference}."
            ),
        }
    )


def _repeat_analysis() -> ReportAnalysisResult:
    base = _analysis()
    glucose_initial, glucose_repeat = base.findings
    hemoglobin_initial = _named_measurement(
        "hemoglobin-initial",
        name="Hemoglobin",
        value="13.2",
        unit="g/dL",
        reference="12-16 g/dL",
        reference_range="12-16",
        date="2026-01-02",
        specimen_event="baseline blood specimen",
        page=2,
    )
    hemoglobin_repeat = _named_measurement(
        "hemoglobin-repeat",
        name="hemoglobin",
        value="13.8",
        unit="g/dL",
        reference="12-16 g/dL",
        reference_range="12-16",
        date="2026-02-02",
        specimen_event="repeat blood specimen",
        page=4,
    )
    hemoglobin_incompatible = _named_measurement(
        "hemoglobin-incompatible-unit",
        name="Hemoglobin",
        value="138",
        unit="g/L",
        reference="120-160 g/L",
        reference_range="120-160",
        date="2026-03-02",
        specimen_event="different-unit blood specimen",
        page=7,
    )
    platelet_initial = _named_measurement(
        "platelet-initial",
        name="Platelet count",
        value="250",
        unit="10^9/L",
        reference="150-450 10^9/L",
        reference_range="150-450",
        date="2026-01-03",
        specimen_event="baseline blood specimen",
        page=2,
    )
    platelet_repeat = _named_measurement(
        "platelet-repeat",
        name="  PLATELET   COUNT ",
        value="265",
        unit="10^9/L",
        reference="150-450 10^9/L",
        reference_range="150-450",
        date="2026-02-03",
        specimen_event="repeat blood specimen",
        page=4,
    )
    potassium_undated = _named_measurement(
        "potassium-undated",
        name="Potassium",
        value="4.1",
        unit="mmol/L",
        reference="3.5-5.1 mmol/L",
        reference_range="3.5-5.1",
        date=None,
        specimen_event="undated chemistry specimen",
        page=3,
    )
    potassium_dated = _named_measurement(
        "potassium-dated",
        name="potassium",
        value="4.3",
        unit="mmol/L",
        reference="3.5-5.1 mmol/L",
        reference_range="3.5-5.1",
        date="2026-02-04",
        specimen_event="dated chemistry specimen",
        page=5,
    )
    platelet_control = _named_measurement(
        "platelet-training-control",
        name="Platelet count",
        value="999",
        unit="10^9/L",
        reference="150-450 10^9/L",
        reference_range="150-450",
        date="2026-01-03",
        specimen_event="training control",
        page=8,
    ).model_copy(update={"finding_kind": FindingKind.TRAINING_CONTROL})
    return ReportAnalysisResult(
        filename="synthetic-repeat-report.pdf",
        findings=[
            glucose_initial,
            hemoglobin_initial,
            platelet_initial,
            potassium_undated,
            glucose_repeat,
            hemoglobin_repeat,
            platelet_repeat,
            potassium_dated,
            hemoglobin_incompatible,
            platelet_control,
        ],
        training_controls=[platelet_control],
        review_warnings=base.review_warnings,
    )


def _repeat_selection(
    repeat_groups: list[RepeatResultSelection] | None = None,
) -> ExtractedReportSummary:
    return ExtractedReportSummary(
        overview_finding_ids=["measurement-initial"],
        important_finding_ids=["measurement-initial"],
        repeat_result_groups=repeat_groups or [],
    )


def _valid_selection() -> ExtractedReportSummary:
    return ExtractedReportSummary(
        overview_finding_ids=["measurement-initial", "narrative-symptom"],
        important_finding_ids=["measurement-initial"],
        repeat_result_groups=[
            RepeatResultSelection(
                finding_ids=["measurement-repeat", "measurement-initial"]
            )
        ],
        narrative_finding_ids=["narrative-symptom", "narrative-follow-up"],
        uncertainty_finding_ids=[
            "measurement-initial",
            "narrative-follow-up",
        ],
    )


def test_valid_summary_preserves_ids_numeric_fidelity_and_sources() -> None:
    analysis = _analysis()
    runner = StubRunner(_valid_selection())

    result = SummaryAgent(runner=runner).summarize(analysis)

    assert runner.seen_analysis is analysis
    important = result.important_findings[0]
    assert important.finding_ids == ["measurement-initial"]
    assert "118 mg/dL" in important.text
    assert "70-99 mg/dL" in important.text
    assert "above" in important.text
    assert important.source_pages == [1]
    assert important.source_evidence[0].supporting_excerpt == (
        analysis.findings[0].supporting_excerpt
    )


def test_crewai_runner_uses_explicit_gemini_and_bounded_execution() -> None:
    calls: dict[str, Any] = {}
    llm_object = object()
    agent_object = object()
    task_object = object()

    def llm_factory(**kwargs: Any) -> object:
        calls["llm"] = kwargs
        return llm_object

    def agent_factory(**kwargs: Any) -> object:
        calls["agent"] = kwargs
        return agent_object

    def task_factory(**kwargs: Any) -> object:
        calls["task"] = kwargs
        return task_object

    class FakeCrew:
        def kickoff(self) -> SimpleNamespace:
            return SimpleNamespace(pydantic=_valid_selection())

    def crew_factory(**kwargs: Any) -> FakeCrew:
        calls["crew"] = kwargs
        return FakeCrew()

    components = SummaryCrewAIComponents(
        llm=llm_factory,
        agent=agent_factory,
        task=task_factory,
        crew=crew_factory,
        sequential_process="sequential",
    )
    runner = CrewAISummaryRunner(
        GeminiConfig(api_key="synthetic-test-key", model="explicit-test-model"),
        timeout_seconds=17,
        max_output_tokens=777,
        components=components,
    )

    output = runner.summarize(_analysis())

    assert output == _valid_selection()
    assert calls["llm"]["model"] == "gemini/explicit-test-model"
    assert calls["llm"]["max_output_tokens"] == 777
    http_options = calls["llm"]["client_params"]["http_options"]
    assert http_options.timeout == 17_000
    assert http_options.retry_options.attempts == 1
    assert calls["agent"]["llm"] is llm_object
    assert calls["agent"]["tools"] == []
    assert calls["agent"]["allow_delegation"] is False
    assert calls["agent"]["max_iter"] == 1
    assert calls["agent"]["max_execution_time"] == 17
    assert calls["agent"]["max_retry_limit"] == 0
    assert calls["agent"]["respect_context_window"] is False
    assert calls["task"]["output_pydantic"] is ExtractedReportSummary
    assert calls["task"]["guardrail_max_retries"] == 0
    assert calls["crew"]["agents"] == [agent_object]


def test_gemini_wire_schema_omits_array_cardinality_constraints() -> None:
    adapter = SanitizedSummaryGeminiCompletion(
        model="gemini-3.5-flash-lite",
        api_key="synthetic-test-key",
        max_output_tokens=4096,
    )
    config = adapter._prepare_generation_config(
        None,
        [],
        ExtractedReportSummary,
    )
    schema = config.model_dump(exclude_none=True)["response_json_schema"]

    def collect_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            keys = set(value)
            for nested in value.values():
                keys.update(collect_keys(nested))
            return keys
        if isinstance(value, list):
            keys: set[str] = set()
            for nested in value:
                keys.update(collect_keys(nested))
            return keys
        return set()

    schema_keys = collect_keys(schema)
    assert "minItems" not in schema_keys
    assert "maxItems" not in schema_keys


def test_summary_cardinality_bounds_remain_enforced_application_side() -> None:
    with pytest.raises(ValidationError):
        ExtractedReportSummary(overview_finding_ids=[])

    assert RepeatResultSelection(finding_ids=["only-one"]).finding_ids == ["only-one"]

    with pytest.raises(ValidationError):
        RepeatResultSelection(finding_ids=[])

    with pytest.raises(ValidationError):
        ExtractedReportSummary(
            overview_finding_ids=["one"],
            important_finding_ids=[
                f"finding-{index}" for index in range(MAX_SUMMARY_REFERENCES + 1)
            ],
        )

    valid_group = RepeatResultSelection(finding_ids=["one", "two"])
    with pytest.raises(ValidationError):
        ExtractedReportSummary(
            overview_finding_ids=["one"],
            repeat_result_groups=[valid_group] * (MAX_REPEAT_GROUPS + 1),
        )


def test_unknown_finding_id_is_rejected_and_training_control_is_excluded() -> None:
    analysis = _analysis()
    unknown = _valid_selection().model_copy(
        update={"overview_finding_ids": ["measurement-training-control"]}
    )

    with pytest.raises(
        SummaryReferenceValidationError,
        match=(
            "category=unknown_or_excluded_references, missing_count=0, "
            "misclassified_count=1"
        ),
    ):
        SummaryAgent(runner=StubRunner(unknown)).summarize(analysis)


def test_repeat_sequence_uses_analysis_order_and_preserves_events() -> None:
    result = SummaryAgent(runner=StubRunner(_valid_selection())).summarize(_analysis())

    repeat = result.repeat_result_changes[0]
    assert repeat.finding_ids == ["measurement-initial", "measurement-repeat"]
    assert "118 mg/dL (2026-01-01, synthetic specimen 2026-01-01)" in repeat.text
    assert "96 mg/dL (2026-02-01, synthetic specimen 2026-02-01)" in repeat.text
    assert "clinical recovery" not in repeat.text
    assert result.disclaimer == SUMMARY_DISCLAIMER
    assert "recovery" in result.disclaimer
    assert "causation" in result.disclaimer
    assert "Supported numeric difference (subsequent minus initial): -22 mg/dL" in (
        repeat.text
    )


def test_omitted_repeat_groups_are_completed_deterministically() -> None:
    analysis = _repeat_analysis()

    result = SummaryAgent(runner=StubRunner(_repeat_selection())).summarize(analysis)

    group_ids = {frozenset(item.finding_ids) for item in result.repeat_result_changes}
    assert group_ids == {
        frozenset({"measurement-initial", "measurement-repeat"}),
        frozenset({"hemoglobin-initial", "hemoglobin-repeat"}),
        frozenset({"platelet-initial", "platelet-repeat"}),
        frozenset({"potassium-undated", "potassium-dated"}),
    }
    all_repeat_ids = set().union(*(set(group) for group in group_ids))
    assert "hemoglobin-incompatible-unit" not in all_repeat_ids
    assert "platelet-training-control" not in all_repeat_ids


def test_repeat_completeness_check_counts_missing_groups_not_findings() -> None:
    analysis = _repeat_analysis()
    assembled = _build_repeat_groups(analysis, [])

    with pytest.raises(
        SummaryReferenceValidationError,
        match=("category=repeat_result_groups, missing_count=1, misclassified_count=0"),
    ):
        _validate_assembled_repeat_groups(analysis, assembled[:-1])


def test_one_member_and_complete_repeat_preferences_only_control_group_order() -> None:
    analysis = _repeat_analysis()
    selection = _repeat_selection(
        [
            RepeatResultSelection(finding_ids=["platelet-repeat"]),
            RepeatResultSelection(
                finding_ids=["hemoglobin-repeat", "hemoglobin-initial"]
            ),
        ]
    )

    result = SummaryAgent(runner=StubRunner(selection)).summarize(analysis)

    assert result.repeat_result_changes[0].finding_ids == [
        "platelet-initial",
        "platelet-repeat",
    ]
    assert result.repeat_result_changes[1].finding_ids == [
        "hemoglobin-initial",
        "hemoglobin-repeat",
    ]
    assert len(result.repeat_result_changes) == 4


def test_unknown_repeat_chronology_uses_report_order_without_difference() -> None:
    analysis = _repeat_analysis()
    result = SummaryAgent(runner=StubRunner(_repeat_selection())).summarize(analysis)
    potassium = next(
        item
        for item in result.repeat_result_changes
        if "potassium-undated" in item.finding_ids
    )

    assert potassium.finding_ids == ["potassium-undated", "potassium-dated"]
    assert "report order; chronology was not established" in potassium.text
    assert "Result 1: 4.1 mmol/L" in potassium.text
    assert "Result 2: 4.3 mmol/L" in potassium.text
    assert "subsequent minus initial" not in potassium.text


def test_repeat_comparison_preserves_ranges_warnings_and_section_overlap() -> None:
    analysis = _repeat_analysis()
    result = SummaryAgent(runner=StubRunner(_repeat_selection())).summarize(analysis)
    glucose = next(
        item
        for item in result.repeat_result_changes
        if "measurement-initial" in item.finding_ids
    )
    uncertainty_ids = {
        finding_id for item in result.uncertainties for finding_id in item.finding_ids
    }

    assert "supplied range: 70-99 mg/dL" in glucose.text
    assert "Fasting status is unknown" in glucose.text
    assert "measurement-initial" in result.important_findings[0].finding_ids
    assert "measurement-initial" in glucose.finding_ids
    assert "measurement-initial" in uncertainty_ids
    assert len(glucose.source_evidence) == 2


def test_compact_findings_cover_all_required_repeats_without_duplicates() -> None:
    analysis = _repeat_analysis()
    result = SummaryAgent(runner=StubRunner(_repeat_selection())).summarize(analysis)

    rows = build_compact_finding_rows(result)
    covered_ids = {finding_id for row in rows for finding_id in row.finding_ids}
    required_ids = {
        finding_id
        for item in result.repeat_result_changes
        for finding_id in item.finding_ids
    }

    assert len(rows) == len(result.repeat_result_changes) == 4
    assert covered_ids == required_ids
    assert "platelet-training-control" not in covered_ids
    assert "hemoglobin-incompatible-unit" not in covered_ids
    glucose = next(row for row in rows if row.finding == "Glucose")
    assert "2026-01-01: 118 mg/dL" in glucose.reported_results
    assert "2026-02-01: 96 mg/dL" in glucose.reported_results
    assert "2026-01-01: 70-99 mg/dL" in glucose.supplied_ranges
    assert "Fasting status is unknown" in glucose.review_warnings
    assert glucose.category == "Outside supplied range + repeat"


def test_compact_presentation_retains_applicability_and_grouped_uncertainties() -> None:
    result = SummaryAgent(runner=StubRunner(_long_selection())).summarize(
        _long_analysis()
    )

    finding_rows = build_compact_finding_rows(result)
    glucose = next(row for row in finding_rows if row.finding == "Glucose")
    assert "Fasting status was documented for the repeat specimen" in (
        glucose.applicability
    )
    assert "Fasting status is unknown" in glucose.review_warnings
    glucose_repeat = next(
        evidence
        for item in result.repeat_result_changes
        for evidence in item.source_evidence
        if evidence.finding_id == "measurement-repeat"
    )
    assert glucose_repeat.applicability_evidence[0].source_page == 2
    assert "documented for the repeat specimen" in (
        glucose_repeat.applicability_evidence[0].text
    )

    uncertainty_rows = build_compact_uncertainty_rows(result)
    affected = {
        finding_id: row for row in uncertainty_rows for finding_id in row.finding_ids
    }
    missing_range = affected["measurement-pulse-no-range"]
    unsupported = affected["measurement-unsupported-range"]
    assert "Pulse (2026-01-01)" in missing_range.affected_findings
    assert "Respiration (2026-01-01)" in missing_range.affected_findings
    assert "No reference range was supplied" in missing_range.issue
    assert "unsupported" not in missing_range.issue
    assert "Age-specific marker (2026-01-03)" in unsupported.affected_findings
    assert "unsupported" in unsupported.issue
    assert "Medication uncertainty (page 6)" in (
        affected["narrative-medication"].affected_findings
    )
    assert "Allergy verification (page 7)" in (
        affected["narrative-allergy"].affected_findings
    )


def test_compact_repeat_vitals_use_not_supplied_not_measurement_evidence() -> None:
    no_range_reason = (
        "No reference range was supplied; this is not an extraction error and "
        "abnormality was not assessed."
    )
    unsupported_value = (
        "Reported value format is unsupported; abnormality was not assessed."
    )

    def blood_pressure(
        finding_id: str,
        value: str,
        date: str,
        page: int,
    ) -> ReportFinding:
        return _measurement(
            finding_id,
            value=value,
            assessment=FindingAssessment.NOT_ASSESSED,
            date=date,
            page=page,
        ).model_copy(
            update={
                "name": "Seated blood pressure",
                "reported_unit": "mmHg",
                "reference_text": None,
                "reference_range": None,
                "reference_unit": None,
                "reference_supporting_excerpt": (
                    "No explicit reference ranges accompany these measurements."
                ),
                "reference_absence_statement": (
                    "No explicit reference ranges accompany these measurements."
                ),
                "supporting_excerpt": (f"Seated blood pressure was {value} mmHg."),
                "numeric_review_reasons": [unsupported_value, no_range_reason],
                "review_reasons": [unsupported_value, no_range_reason],
                "review_required": True,
            }
        )

    first = blood_pressure("bp-initial", "142/88", "2026-09-14", 1)
    repeat = blood_pressure("bp-repeat", "134/82", "2026-09-17", 7)
    analysis = ReportAnalysisResult(
        filename="synthetic-vitals.pdf",
        findings=[first, repeat],
    )
    selection = ExtractedReportSummary(
        overview_finding_ids=["bp-initial", "bp-repeat"],
    )

    result = SummaryAgent(runner=StubRunner(selection)).summarize(analysis)
    row = build_compact_finding_rows(result)[0]

    assert row.reported_results == ("2026-09-14: 142/88 mmHg; 2026-09-17: 134/82 mmHg")
    assert row.supplied_ranges == ("2026-09-14: Not supplied; 2026-09-17: Not supplied")
    assert "blood pressure was" not in row.supplied_ranges.casefold()
    evidence = result.repeat_result_changes[0].source_evidence[0]
    assert evidence.supporting_excerpt.startswith("Seated blood pressure")
    assert evidence.reference_absence_statement is not None
    assert evidence.reference_supporting_excerpt != evidence.supporting_excerpt


def test_conditional_tsh_summary_has_no_invented_missing_unit_issue() -> None:
    conditional_reason = (
        "The supplied reference context is conditional; applicability requires review."
    )
    context = ExtractedApplicabilityContext(
        text="pregnancy-specific intervals differ",
        source_page=4,
        supporting_excerpt=(
            "adult interval 0.4-4.0 mIU/L; pregnancy-specific intervals differ"
        ),
    )
    tsh = _measurement(
        "tsh-conditional",
        value="2.1",
        assessment=FindingAssessment.WITHIN_RANGE,
        date="2026-09-15",
        page=4,
    ).model_copy(
        update={
            "name": "Thyroid-stimulating hormone",
            "reported_unit": "mIU/L",
            "reference_text": (
                "adult interval 0.4-4.0 mIU/L; pregnancy-specific intervals differ"
            ),
            "reference_range": "0.4-4.0",
            "reference_unit": "mIU/L",
            "reference_supporting_excerpt": (
                "Printed reference text: adult interval 0.4-4.0 mIU/L; "
                "pregnancy-specific intervals differ."
            ),
            "applicability_contexts": [context],
            "applicability_status": ApplicabilityStatus.REVIEW_REQUIRED,
            "applicability_review_reasons": [conditional_reason],
            "review_reasons": [conditional_reason],
            "review_required": True,
            "supporting_excerpt": ("Thyroid-stimulating hormone: 2.1 mIU/L."),
        }
    )
    analysis = ReportAnalysisResult(
        filename="synthetic-tsh.pdf",
        findings=[tsh],
    )
    selection = ExtractedReportSummary(
        overview_finding_ids=["tsh-conditional"],
    )

    result = SummaryAgent(runner=StubRunner(selection)).summarize(analysis)

    assert len(result.uncertainties) == 1
    assert conditional_reason in result.uncertainties[0].text
    assert "unit" not in result.uncertainties[0].text.casefold()
    evidence = result.uncertainties[0].source_evidence[0]
    assert evidence.reported_unit == "mIU/L"
    assert evidence.reference_unit == "mIU/L"


def test_single_disclaimer_replaces_repeated_item_disclaimers() -> None:
    result = SummaryAgent(runner=StubRunner(_long_selection())).summarize(
        _long_analysis()
    )
    item_text = " ".join(
        item.text
        for section in (
            result.important_findings,
            result.repeat_result_changes,
            result.narrative_context,
            result.uncertainties,
        )
        for item in section
    )

    assert result.disclaimer == SUMMARY_DISCLAIMER
    assert SUMMARY_DISCLAIMER not in item_text
    assert "not a diagnosis" not in item_text
    assert "clinical recovery" not in item_text


def test_repeat_preference_cannot_mix_analytes_or_incompatible_units() -> None:
    analysis = _repeat_analysis()
    mixed_analytes = _repeat_selection(
        [RepeatResultSelection(finding_ids=["hemoglobin-initial", "platelet-repeat"])]
    )
    incompatible_unit = _repeat_selection(
        [
            RepeatResultSelection(
                finding_ids=[
                    "hemoglobin-initial",
                    "hemoglobin-incompatible-unit",
                ]
            )
        ]
    )

    for selection in (mixed_analytes, incompatible_unit):
        with pytest.raises(
            SummaryReferenceValidationError,
            match=(
                "category=repeat_group_preferences, missing_count=0, "
                "misclassified_count=1"
            ),
        ):
            SummaryAgent(runner=StubRunner(selection)).summarize(analysis)


def test_applicability_participant_attribution_and_missing_information_remain() -> None:
    result = SummaryAgent(runner=StubRunner(_valid_selection())).summarize(_analysis())

    uncertainties = {item.finding_ids[0]: item.text for item in result.uncertainties}
    assert "Fasting status is unknown" in uncertainties["measurement-initial"]
    assert "not_recorded" in uncertainties["narrative-follow-up"]
    assert "not a negative finding" in uncertainties["narrative-follow-up"]
    narratives = {item.finding_ids[0]: item.text for item in result.narrative_context}
    assert "participant reported" in narratives["narrative-symptom"]


def test_omitted_uncertainty_is_added_but_outside_range_omission_is_rejected() -> None:
    missing_uncertainty = _valid_selection().model_copy(
        update={"uncertainty_finding_ids": []}
    )
    result = SummaryAgent(runner=StubRunner(missing_uncertainty)).summarize(_analysis())
    covered_uncertainties = {
        finding_id for item in result.uncertainties for finding_id in item.finding_ids
    }
    assert covered_uncertainties == {
        "measurement-initial",
        "narrative-follow-up",
    }

    missing_important = _valid_selection().model_copy(
        update={"important_finding_ids": []}
    )
    with pytest.raises(
        SummaryReferenceValidationError,
        match=(
            "category=outside_range_findings, missing_count=1, misclassified_count=0"
        ),
    ):
        SummaryAgent(runner=StubRunner(missing_important)).summarize(_analysis())


def test_long_report_uncertainties_are_complete_when_gemini_selects_none() -> None:
    analysis = _long_analysis()

    result = SummaryAgent(runner=StubRunner(_long_selection())).summarize(analysis)

    uncertainty_ids = {
        finding_id for item in result.uncertainties for finding_id in item.finding_ids
    }
    assert uncertainty_ids == {
        "measurement-initial",
        "measurement-pulse-no-range",
        "measurement-respiration-no-range",
        "measurement-unit-mismatch",
        "measurement-unsupported-range",
        "narrative-medication",
        "narrative-allergy",
        "narrative-follow-up",
    }
    assert "measurement-initial" in result.important_findings[0].finding_ids
    assert any(
        "measurement-initial" in item.finding_ids
        and "Fasting status is unknown" in item.text
        for item in result.uncertainties
    )

    no_range = next(
        item
        for item in result.uncertainties
        if "measurement-pulse-no-range" in item.finding_ids
    )
    assert no_range.finding_ids == [
        "measurement-pulse-no-range",
        "measurement-respiration-no-range",
    ]
    assert len(no_range.source_evidence) == 2
    assert "remain unassessed" in no_range.text
    assert "not the same as a normal or abnormal result" in no_range.text

    repeat = analysis.findings[1]
    assert repeat.applicability_status is ApplicabilityStatus.DOCUMENTED
    assert repeat.finding_id not in uncertainty_ids
    assert not result.narrative_context
    assert "measurement(s)" not in result.overview.text
    assert "validated measurements including Glucose" in result.overview.text


def test_long_report_uncertainty_text_preserves_required_categories() -> None:
    result = SummaryAgent(runner=StubRunner(_long_selection())).summarize(
        _long_analysis()
    )
    by_id = {
        finding_id: item.text
        for item in result.uncertainties
        for finding_id in item.finding_ids
    }

    assert "units do not match" in by_id["measurement-unit-mismatch"]
    assert "unsupported" in by_id["measurement-unsupported-range"]
    assert "participant reported" in by_id["narrative-medication"]
    assert "uncertain" in by_id["narrative-medication"]
    assert "incomplete" in by_id["narrative-allergy"]
    assert "not_recorded" in by_id["narrative-follow-up"]
    assert "not a negative finding" in by_id["narrative-follow-up"]


def test_non_uncertain_optional_context_does_not_create_uncertainty() -> None:
    analysis = _long_analysis()
    repeat = analysis.findings[1]
    symptom = analysis.narrative_findings[0]
    focused_analysis = ReportAnalysisResult(
        filename=analysis.filename,
        findings=[repeat],
        narrative_findings=[symptom],
    )
    selection = ExtractedReportSummary(
        overview_finding_ids=[repeat.finding_id, symptom.finding_id],
    )

    result = SummaryAgent(runner=StubRunner(selection)).summarize(focused_analysis)

    assert result.uncertainties == []


def test_uncertainty_preference_rejects_non_uncertain_id_with_safe_counts() -> None:
    analysis = _long_analysis()
    selection = _long_selection().model_copy(
        update={"uncertainty_finding_ids": ["measurement-repeat"]}
    )

    with pytest.raises(
        SummaryReferenceValidationError,
        match=(
            "category=uncertainty_preferences, missing_count=0, misclassified_count=1"
        ),
    ) as error:
        SummaryAgent(runner=StubRunner(selection)).summarize(analysis)

    rendered = str(error.value)
    assert "measurement-repeat" not in rendered
    assert analysis.findings[1].supporting_excerpt not in rendered


def test_unsupported_generated_claim_field_is_rejected() -> None:
    raw = _valid_selection().model_dump()
    raw["diagnosis"] = "SENSITIVE UNSUPPORTED DIAGNOSIS"

    with pytest.raises(SummaryOutputError, match="invalid structured output"):
        SummaryAgent(runner=StubRunner(raw)).summarize(_analysis())


def test_truncated_json_is_rejected_without_exposing_content(caplog: Any) -> None:
    sensitive = "SENSITIVE PARTICIPANT CONTENT"
    adapter = object.__new__(SanitizedSummaryGeminiCompletion)
    truncated = "".join(
        (
            chr(123),
            chr(34),
            "overview_finding_ids",
            chr(34),
            ": [",
            chr(34),
            sensitive,
        )
    )

    with caplog.at_level(logging.WARNING), pytest.raises(SummaryIncompleteOutputError):
        adapter._validate_and_emit_structured_output(
            truncated,
            ExtractedReportSummary,
            [],
            usage={"total_token_count": 42},
            finish_reason="MAX_TOKENS",
        )

    assert sensitive not in caplog.text


def test_token_limit_finish_rejects_partial_summary(monkeypatch: Any) -> None:
    adapter = object.__new__(SanitizedSummaryGeminiCompletion)
    monkeypatch.setattr(
        SanitizedSummaryGeminiCompletion,
        "_extract_finish_reason_and_id",
        lambda self, response: ("MAX_TOKENS", None),
    )

    with pytest.raises(SummaryIncompleteOutputError) as error:
        adapter._process_response_with_tools(
            object(),
            [],
            usage={"total_token_count": 73},
        )

    assert error.value.finish_reason == "MAX_TOKENS"
    assert error.value.token_usage == {"total_token_count": 73}


def test_dependency_logs_are_redacted() -> None:
    sensitive = "SENSITIVE GENERATED FINDING FRAGMENT"
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("crewai.flow.runtime")
    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    try:
        with _redact_dependency_logs():
            logger.error("provider output: %s", sensitive)
    finally:
        logger.removeHandler(handler)

    rendered = " ".join(record.getMessage() for record in records)
    assert sensitive not in rendered
    assert "sanitized" in rendered


def test_unknown_ui_error_is_safe_and_preserves_no_raw_details() -> None:
    message = safe_summary_error_message(
        RuntimeError("SENSITIVE GENERATED FINDING FRAGMENT")
    )

    assert "SENSITIVE" not in message
    assert "existing report analysis was preserved" in message
