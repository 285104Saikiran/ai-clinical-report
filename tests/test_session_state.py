"""Tests for report-analysis state isolation across uploaded documents."""

from clinical_report_assistant.session_state import (
    REPORT_ANALYSIS_CONTEXT_ID_KEY,
    REPORT_ANALYSIS_RECORD_KEY,
    REPORT_ANALYSIS_RESULT_KEY,
    REPORT_ANALYSIS_REUSED_KEY,
    REPORT_DOCUMENT_ID_KEY,
    REPORT_RECOMMENDATION_CONTEXT_ID_KEY,
    REPORT_RECOMMENDATION_RECORD_KEY,
    REPORT_RECOMMENDATION_RESULT_KEY,
    REPORT_RECOMMENDATION_REUSED_KEY,
    REPORT_RECOMMENDATION_REVIEW_KEY,
    REPORT_SUMMARY_CONTEXT_ID_KEY,
    REPORT_SUMMARY_RECORD_KEY,
    REPORT_SUMMARY_RESULT_KEY,
    REPORT_SUMMARY_REUSED_KEY,
    clear_summary_result,
    summary_context_id,
    synchronize_analysis_context,
    synchronize_recommendation_context,
    synchronize_report_document,
    synchronize_summary_context,
)


def test_changed_report_clears_stale_analysis() -> None:
    state = {
        REPORT_DOCUMENT_ID_KEY: "first-report",
        REPORT_ANALYSIS_RESULT_KEY: object(),
        REPORT_SUMMARY_CONTEXT_ID_KEY: "old-context",
        REPORT_SUMMARY_RESULT_KEY: object(),
    }

    synchronize_report_document(state, "second-report")

    assert state[REPORT_DOCUMENT_ID_KEY] == "second-report"
    assert REPORT_ANALYSIS_RESULT_KEY not in state
    assert REPORT_SUMMARY_CONTEXT_ID_KEY not in state
    assert REPORT_SUMMARY_RESULT_KEY not in state


def test_same_report_keeps_analysis_and_removed_upload_clears_it() -> None:
    result = object()
    state = {
        REPORT_DOCUMENT_ID_KEY: "same-report",
        REPORT_ANALYSIS_RESULT_KEY: result,
    }

    synchronize_report_document(state, "same-report")
    assert state[REPORT_ANALYSIS_RESULT_KEY] is result

    synchronize_report_document(state, None)
    assert state[REPORT_DOCUMENT_ID_KEY] is None
    assert REPORT_ANALYSIS_RESULT_KEY not in state


def test_analysis_or_model_change_clears_only_stale_summary() -> None:
    analysis = object()
    summary = object()
    first_id = summary_context_id(
        "analysis-v1",
        model="gemini-model-a",
        max_output_tokens=4096,
    )
    state = {
        REPORT_ANALYSIS_RESULT_KEY: analysis,
        REPORT_SUMMARY_CONTEXT_ID_KEY: first_id,
        REPORT_SUMMARY_RESULT_KEY: summary,
    }

    synchronize_summary_context(state, first_id)
    assert state[REPORT_SUMMARY_RESULT_KEY] is summary

    second_id = summary_context_id(
        "analysis-v1",
        model="gemini-model-b",
        max_output_tokens=4096,
    )
    synchronize_summary_context(state, second_id)

    assert state[REPORT_ANALYSIS_RESULT_KEY] is analysis
    assert REPORT_SUMMARY_RESULT_KEY not in state


def test_summary_failure_cleanup_preserves_valid_analysis() -> None:
    analysis = object()
    state = {
        REPORT_ANALYSIS_RESULT_KEY: analysis,
        REPORT_SUMMARY_CONTEXT_ID_KEY: "context",
        REPORT_SUMMARY_RESULT_KEY: object(),
    }

    clear_summary_result(state)

    assert state[REPORT_ANALYSIS_RESULT_KEY] is analysis
    assert REPORT_SUMMARY_CONTEXT_ID_KEY not in state
    assert REPORT_SUMMARY_RESULT_KEY not in state


def test_generation_context_change_clears_analysis_and_all_dependent_metadata() -> None:
    state = {
        REPORT_ANALYSIS_CONTEXT_ID_KEY: "old-analysis-settings",
        REPORT_ANALYSIS_RESULT_KEY: object(),
        REPORT_ANALYSIS_RECORD_KEY: object(),
        REPORT_ANALYSIS_REUSED_KEY: True,
        REPORT_SUMMARY_CONTEXT_ID_KEY: "old-summary-settings",
        REPORT_SUMMARY_RESULT_KEY: object(),
        REPORT_SUMMARY_RECORD_KEY: object(),
        REPORT_SUMMARY_REUSED_KEY: True,
    }

    synchronize_analysis_context(state, "new-analysis-settings")

    assert state[REPORT_ANALYSIS_CONTEXT_ID_KEY] == "new-analysis-settings"
    assert REPORT_ANALYSIS_RESULT_KEY not in state
    assert REPORT_ANALYSIS_RECORD_KEY not in state
    assert REPORT_ANALYSIS_REUSED_KEY not in state
    assert REPORT_SUMMARY_CONTEXT_ID_KEY not in state
    assert REPORT_SUMMARY_RESULT_KEY not in state
    assert REPORT_SUMMARY_RECORD_KEY not in state
    assert REPORT_SUMMARY_REUSED_KEY not in state


def test_summary_context_change_clears_saved_record_metadata_only() -> None:
    analysis = object()
    state = {
        REPORT_ANALYSIS_RESULT_KEY: analysis,
        REPORT_SUMMARY_CONTEXT_ID_KEY: "old-summary-settings",
        REPORT_SUMMARY_RESULT_KEY: object(),
        REPORT_SUMMARY_RECORD_KEY: object(),
        REPORT_SUMMARY_REUSED_KEY: True,
    }

    synchronize_summary_context(state, "new-summary-settings")

    assert state[REPORT_ANALYSIS_RESULT_KEY] is analysis
    assert state[REPORT_SUMMARY_CONTEXT_ID_KEY] == "new-summary-settings"
    assert REPORT_SUMMARY_RESULT_KEY not in state
    assert REPORT_SUMMARY_RECORD_KEY not in state
    assert REPORT_SUMMARY_REUSED_KEY not in state


def test_recommendation_context_change_preserves_analysis_and_summary() -> None:
    analysis = object()
    summary = object()
    state = {
        REPORT_ANALYSIS_RESULT_KEY: analysis,
        REPORT_SUMMARY_RESULT_KEY: summary,
        REPORT_RECOMMENDATION_CONTEXT_ID_KEY: "old-guideline-fingerprint",
        REPORT_RECOMMENDATION_RESULT_KEY: object(),
        REPORT_RECOMMENDATION_RECORD_KEY: object(),
        REPORT_RECOMMENDATION_REUSED_KEY: True,
        REPORT_RECOMMENDATION_REVIEW_KEY: object(),
    }

    synchronize_recommendation_context(state, "new-guideline-fingerprint")

    assert state[REPORT_ANALYSIS_RESULT_KEY] is analysis
    assert state[REPORT_SUMMARY_RESULT_KEY] is summary
    assert state[REPORT_RECOMMENDATION_CONTEXT_ID_KEY] == "new-guideline-fingerprint"
    assert REPORT_RECOMMENDATION_RESULT_KEY not in state
    assert REPORT_RECOMMENDATION_RECORD_KEY not in state
    assert REPORT_RECOMMENDATION_REUSED_KEY not in state
    assert REPORT_RECOMMENDATION_REVIEW_KEY not in state
