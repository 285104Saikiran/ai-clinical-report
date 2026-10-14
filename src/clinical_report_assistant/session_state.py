"""Framework-neutral helpers for report-analysis and summary session state."""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from typing import Any

REPORT_DOCUMENT_ID_KEY = "report_document_id"
REPORT_ANALYSIS_CONTEXT_ID_KEY = "report_analysis_context_id"
REPORT_ANALYSIS_RESULT_KEY = "report_analysis_result"
REPORT_ANALYSIS_RECORD_KEY = "report_analysis_record"
REPORT_ANALYSIS_REUSED_KEY = "report_analysis_reused"
REPORT_SUMMARY_CONTEXT_ID_KEY = "report_summary_context_id"
REPORT_SUMMARY_RESULT_KEY = "report_summary_result"
REPORT_SUMMARY_RECORD_KEY = "report_summary_record"
REPORT_SUMMARY_REUSED_KEY = "report_summary_reused"
REPORT_RECOMMENDATION_CONTEXT_ID_KEY = "report_recommendation_context_id"
REPORT_RECOMMENDATION_RESULT_KEY = "report_recommendation_result"
REPORT_RECOMMENDATION_RECORD_KEY = "report_recommendation_record"
REPORT_RECOMMENDATION_REUSED_KEY = "report_recommendation_reused"
REPORT_RECOMMENDATION_REVIEW_KEY = "report_recommendation_review"


def synchronize_report_document(
    state: MutableMapping[str, Any],
    document_id: str | None,
) -> None:
    """Clear report-specific results whenever the active upload changes."""

    if state.get(REPORT_DOCUMENT_ID_KEY) == document_id:
        return
    state[REPORT_DOCUMENT_ID_KEY] = document_id
    clear_analysis_result(state)


def synchronize_analysis_context(
    state: MutableMapping[str, Any],
    context_id: str,
) -> None:
    """Clear analysis and summary when generation compatibility changes."""

    if state.get(REPORT_ANALYSIS_CONTEXT_ID_KEY) == context_id:
        return
    clear_analysis_result(state)
    state[REPORT_ANALYSIS_CONTEXT_ID_KEY] = context_id


def summary_context_id(
    analysis_payload: str,
    *,
    model: str,
    max_output_tokens: int,
) -> str:
    """Hash non-secret inputs that determine whether a summary is still current."""

    payload = "\0".join((analysis_payload, model, str(max_output_tokens)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def synchronize_summary_context(
    state: MutableMapping[str, Any],
    context_id: str,
) -> None:
    """Clear only a stale summary when its analysis or model context changes."""

    if state.get(REPORT_SUMMARY_CONTEXT_ID_KEY) == context_id:
        return
    state[REPORT_SUMMARY_CONTEXT_ID_KEY] = context_id
    state.pop(REPORT_SUMMARY_RESULT_KEY, None)
    state.pop(REPORT_SUMMARY_RECORD_KEY, None)
    state.pop(REPORT_SUMMARY_REUSED_KEY, None)
    clear_recommendation_result(state)


def synchronize_recommendation_context(
    state: MutableMapping[str, Any],
    context_id: str,
) -> None:
    """Clear only recommendation state when any exact dependency changes."""

    if state.get(REPORT_RECOMMENDATION_CONTEXT_ID_KEY) == context_id:
        return
    clear_recommendation_result(state)
    state[REPORT_RECOMMENDATION_CONTEXT_ID_KEY] = context_id


def clear_analysis_result(state: MutableMapping[str, Any]) -> None:
    """Clear report analysis and its dependent summary state."""

    state.pop(REPORT_ANALYSIS_CONTEXT_ID_KEY, None)
    state.pop(REPORT_ANALYSIS_RESULT_KEY, None)
    state.pop(REPORT_ANALYSIS_RECORD_KEY, None)
    state.pop(REPORT_ANALYSIS_REUSED_KEY, None)
    clear_summary_result(state)


def clear_summary_result(state: MutableMapping[str, Any]) -> None:
    """Clear summary state without removing a valid report analysis."""

    state.pop(REPORT_SUMMARY_CONTEXT_ID_KEY, None)
    state.pop(REPORT_SUMMARY_RESULT_KEY, None)
    state.pop(REPORT_SUMMARY_RECORD_KEY, None)
    state.pop(REPORT_SUMMARY_REUSED_KEY, None)
    clear_recommendation_result(state)


def clear_recommendation_result(state: MutableMapping[str, Any]) -> None:
    """Clear the dependent draft and exact-draft review from session state."""

    state.pop(REPORT_RECOMMENDATION_CONTEXT_ID_KEY, None)
    state.pop(REPORT_RECOMMENDATION_RESULT_KEY, None)
    state.pop(REPORT_RECOMMENDATION_RECORD_KEY, None)
    state.pop(REPORT_RECOMMENDATION_REUSED_KEY, None)
    state.pop(REPORT_RECOMMENDATION_REVIEW_KEY, None)
