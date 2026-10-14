"""Streamlit interface for ingestion, retrieval, analysis, and summaries."""

from __future__ import annotations

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from clinical_report_assistant.config import load_gemini_config
from clinical_report_assistant.guideline_kb import (
    GuidelineKnowledgeBase,
    GuidelineKnowledgeBaseError,
)
from clinical_report_assistant.pdf_processing import (
    DEFAULT_MAX_PDF_SIZE_BYTES,
    PdfIngestionError,
    process_pdf,
)
from clinical_report_assistant.persistence import (
    AnalysisResolution,
    PersistedResultService,
    PersistenceError,
    RecommendationResolution,
    SavedAnalysisRecord,
    SavedRecommendationDraft,
    SavedReviewDecision,
    SavedSummaryRecord,
    SQLiteResultStore,
    SummaryResolution,
    analysis_compatibility_key,
    analysis_generation_settings,
    recommendation_compatibility_key,
    recommendation_generation_settings,
    report_sha256,
    summary_compatibility_key,
    summary_generation_settings,
)
from clinical_report_assistant.recommendation_agent import (
    SYNTHETIC_GUIDANCE_LABEL,
    RecommendationAgent,
    RecommendationDraftResult,
    RecommendationError,
    RecommendationWorkflow,
    create_edited_draft,
    safe_recommendation_error_message,
)
from clinical_report_assistant.report_analysis_agent import (
    FindingAssessment,
    ReportAnalysisAgent,
    ReportAnalysisResult,
    ReportFinding,
    safe_report_analysis_error_message,
)
from clinical_report_assistant.session_state import (
    REPORT_ANALYSIS_RECORD_KEY,
    REPORT_ANALYSIS_RESULT_KEY,
    REPORT_ANALYSIS_REUSED_KEY,
    REPORT_RECOMMENDATION_RECORD_KEY,
    REPORT_RECOMMENDATION_RESULT_KEY,
    REPORT_RECOMMENDATION_REUSED_KEY,
    REPORT_RECOMMENDATION_REVIEW_KEY,
    REPORT_SUMMARY_RECORD_KEY,
    REPORT_SUMMARY_RESULT_KEY,
    REPORT_SUMMARY_REUSED_KEY,
    clear_analysis_result,
    clear_summary_result,
    synchronize_analysis_context,
    synchronize_recommendation_context,
    synchronize_report_document,
    synchronize_summary_context,
)
from clinical_report_assistant.summary_agent import (
    ReportSummaryResult,
    SummaryAgent,
    SummaryCompactFindingRow,
    SummaryCompactUncertaintyRow,
    SummaryItem,
    build_compact_finding_rows,
    build_compact_uncertainty_rows,
    safe_summary_error_message,
)

_REPORT_DELETION_NOTICE_KEY = "report_saved_results_deletion_notice"


@st.cache_resource
def get_guideline_knowledge_base() -> GuidelineKnowledgeBase:
    """Reuse the local embedding model and Chroma connection across reruns."""

    return GuidelineKnowledgeBase()


def get_streamlit_secrets() -> dict[str, object] | None:
    """Return deployment secrets without requiring a local secrets file."""

    try:
        return st.secrets.to_dict()
    except StreamlitSecretNotFoundError:
        return None


def remember_analysis(resolution: AnalysisResolution) -> None:
    """Replace session analysis only after generation and saving both succeed."""

    st.session_state[REPORT_ANALYSIS_RECORD_KEY] = resolution.record
    st.session_state[REPORT_ANALYSIS_RESULT_KEY] = resolution.record.result
    st.session_state[REPORT_ANALYSIS_REUSED_KEY] = resolution.reused


def remember_summary(resolution: SummaryResolution) -> None:
    """Replace session summary only after generation and saving both succeed."""

    st.session_state[REPORT_SUMMARY_RECORD_KEY] = resolution.record
    st.session_state[REPORT_SUMMARY_RESULT_KEY] = resolution.record.result
    st.session_state[REPORT_SUMMARY_REUSED_KEY] = resolution.reused


def remember_recommendation(resolution: RecommendationResolution) -> None:
    """Replace the session draft only after validation and saving succeed."""

    st.session_state[REPORT_RECOMMENDATION_RECORD_KEY] = resolution.record
    st.session_state[REPORT_RECOMMENDATION_RESULT_KEY] = resolution.record.result
    st.session_state[REPORT_RECOMMENDATION_REUSED_KEY] = resolution.reused
    st.session_state.pop(REPORT_RECOMMENDATION_REVIEW_KEY, None)


def display_saved_status(
    label: str,
    record: SavedAnalysisRecord | SavedSummaryRecord | SavedRecommendationDraft,
    *,
    reused: bool,
) -> None:
    """Clearly label reused versus newly created local results."""

    created_at = record.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    origin = "Reused compatible saved" if reused else "Created and saved new"
    st.caption(f"{origin} {label} from {created_at}.")


def display_report_analysis(result: ReportAnalysisResult) -> None:
    """Render a validated result without placing it in a shared cache."""

    st.subheader("Report Analysis Agent result")
    st.caption(
        "Source validation confirms that cited text exists in the uploaded "
        "report; it does not establish clinical accuracy."
    )
    for warning in result.warnings:
        st.warning(warning)
    if result.review_warnings:
        st.markdown("**Grouped review notes**")
        for group in result.review_warnings:
            references = ", ".join(group.finding_references)
            finding_ids = ", ".join(group.finding_ids)
            pages = ", ".join(str(page) for page in group.source_pages)
            message = (
                f"[{group.category.value}] {group.reason} "
                f"Affected: {references}. Finding IDs: {finding_ids}. "
                f"Source pages: {pages}."
            )
            if group.reason.startswith("No reference range was supplied"):
                st.info(message)
            else:
                st.warning(message)
    st.caption(
        "A measurement with no supplied range is unassessed; that alone does not "
        "mean extraction failed or that the value is abnormal."
    )

    abnormal = [
        finding
        for finding in result.findings
        if finding.assessment
        in {FindingAssessment.ABOVE_RANGE, FindingAssessment.BELOW_RANGE}
    ]
    within_range = [
        finding
        for finding in result.findings
        if finding.assessment is FindingAssessment.WITHIN_RANGE
    ]
    unassessed = [
        finding
        for finding in result.findings
        if finding.assessment is FindingAssessment.NOT_ASSESSED
    ]
    _display_measurement_group("Abnormal measurements", abnormal)
    _display_measurement_group("Within-range measurements", within_range)
    _display_measurement_group("Unassessed measurements", unassessed)

    if result.narrative_findings:
        st.markdown("### Narrative findings")
        for position, finding in enumerate(result.narrative_findings, start=1):
            heading = f"{position}. {finding.category.value} - {finding.status.value}"
            with st.expander(heading, expanded=False):
                st.write(finding.statement)
                st.write(f"Finding ID: {finding.finding_id}")
                if finding.attribution:
                    st.write(f"Attribution: {finding.attribution}")
                st.write(f"Source page: {finding.source_page}")
                st.write("Supporting excerpt")
                st.code(finding.supporting_excerpt, language=None)

    if result.training_controls:
        st.markdown("### Non-patient training controls")
        st.caption(
            "These explicitly labeled controls are excluded from patient findings "
            "that a future Summary Agent may consume."
        )
        for position, finding in enumerate(result.training_controls, start=1):
            _display_measurement(finding, position=position, expanded=False)


def _display_measurement_group(
    title: str,
    findings: list[ReportFinding],
) -> None:
    if not findings:
        return
    st.markdown(f"### {title}")
    for position, finding in enumerate(findings, start=1):
        _display_measurement(finding, position=position, expanded=False)


def _display_measurement(
    finding: ReportFinding,
    *,
    position: int,
    expanded: bool,
) -> None:
    heading = f"{position}. {finding.name} - {finding.assessment.value}"
    with st.expander(heading, expanded=expanded):
        value_text = _with_unit(finding.reported_value, finding.reported_unit)
        original_reference = finding.reference_text or "Not supplied"
        parsed_range = "Not parsed"
        if finding.parsed_numeric_range is not None:
            parsed_range = (
                f"{finding.parsed_numeric_range.lower} to "
                f"{finding.parsed_numeric_range.upper}"
            )
        value_column, reference_column, range_column, page_column = st.columns(4)
        value_column.metric("Reported value", value_text)
        reference_column.metric("Original reference text", original_reference)
        range_column.metric("Parsed numeric range", parsed_range)
        page_column.metric("Source page", finding.source_page)
        st.write(f"Finding ID: {finding.finding_id}")
        if finding.reported_date:
            st.write(f"Reported date: {finding.reported_date}")
        if finding.specimen_event:
            st.write(f"Specimen event: {finding.specimen_event}")
        st.write(f"Applicability status: {finding.applicability_status.value}")
        for context in finding.applicability_contexts:
            context_source = f"page {context.source_page}"
            if context.specimen_event:
                context_source += f", specimen {context.specimen_event}"
            st.info(f"Applicability context ({context_source}): {context.text}")
            st.code(context.supporting_excerpt, language=None)
        if finding.reference_absence_statement:
            st.info(
                "Explicit reference-range absence: "
                f"{finding.reference_absence_statement}"
            )
        if finding.reference_supporting_excerpt:
            st.write("Reference-range evidence")
            st.code(finding.reference_supporting_excerpt, language=None)
        if finding.rejected_applicability_contexts:
            with st.expander("Rejected applicability-context diagnostics"):
                for diagnostic in finding.rejected_applicability_contexts:
                    st.warning(diagnostic.reason)
                    if diagnostic.source_verified:
                        if diagnostic.text:
                            st.write(diagnostic.text)
                        if diagnostic.supporting_excerpt:
                            st.code(diagnostic.supporting_excerpt, language=None)
        if finding.review_reasons:
            with st.expander("Detailed review diagnostics"):
                for reason in finding.review_reasons:
                    st.write(reason)
        st.write("Measurement evidence")
        st.code(finding.supporting_excerpt, language=None)


def _with_unit(value: str | None, unit: str | None) -> str:
    value_text = value or "Missing"
    if unit and unit.casefold() not in value_text.casefold():
        return f"{value_text} {unit}"
    return value_text


def display_report_summary(result: ReportSummaryResult) -> None:
    """Render a validated, session-only report summary."""

    st.subheader("Summary Agent result")
    st.write(result.overview.text)
    st.info(result.disclaimer)

    finding_rows = build_compact_finding_rows(result)
    if finding_rows:
        st.markdown("### Key findings and repeat trends")
        st.dataframe(
            [_compact_finding_table_row(row) for row in finding_rows],
            hide_index=True,
            use_container_width=True,
        )

    uncertainty_rows = build_compact_uncertainty_rows(result)
    if uncertainty_rows:
        st.markdown("### Uncertainties requiring review")
        st.dataframe(
            [_compact_uncertainty_table_row(row) for row in uncertainty_rows],
            hide_index=True,
            use_container_width=True,
        )

    _display_summary_evidence("Overview sources", result.overview)
    _display_summary_section(
        "Outside supplied numeric ranges", result.important_findings
    )
    _display_summary_section("Repeat-result changes", result.repeat_result_changes)
    _display_summary_section("Narrative context", result.narrative_context)
    _display_summary_section(
        "Uncertainties and missing information", result.uncertainties
    )


def _compact_finding_table_row(row: SummaryCompactFindingRow) -> dict[str, str]:
    return {
        "Category": row.category,
        "Finding": row.finding,
        "Dated results": row.reported_results,
        "Supplied ranges": row.supplied_ranges,
        "Assessment": row.assessment,
        "Applicability": row.applicability,
        "Review warnings": row.review_warnings,
        "Sources": row.sources,
    }


def _compact_uncertainty_table_row(
    row: SummaryCompactUncertaintyRow,
) -> dict[str, str]:
    return {
        "Affected findings": row.affected_findings,
        "Issue": row.issue,
        "Sources": row.sources,
    }


def _display_summary_section(title: str, items: list[SummaryItem]) -> None:
    if not items:
        return
    with st.expander(f"Full {title} ({len(items)})", expanded=False):
        for position, item in enumerate(items, start=1):
            st.write(f"{position}. {item.text}")
            _write_summary_evidence(item)


def _display_summary_evidence(label: str, item: SummaryItem) -> None:
    with st.expander(label, expanded=False):
        _write_summary_evidence(item)


def _write_summary_evidence(item: SummaryItem) -> None:
    for evidence in item.source_evidence:
        source = (
            f"{evidence.display_name} - {evidence.finding_id} - "
            f"page {evidence.source_page}"
        )
        if evidence.reported_date:
            source += f" - {evidence.reported_date}"
        if evidence.specimen_event:
            source += f" - {evidence.specimen_event}"
        st.write(source)
        st.write("Measurement evidence")
        st.code(evidence.supporting_excerpt, language=None)
        if evidence.reference_absence_statement:
            st.info(
                "Explicit reference-range absence: "
                f"{evidence.reference_absence_statement}"
            )
        if evidence.reference_supporting_excerpt:
            st.write("Reference-range evidence")
            st.code(evidence.reference_supporting_excerpt, language=None)
        for context in evidence.applicability_evidence:
            context_source = f"Applicability evidence - page {context.source_page}"
            if context.specimen_event:
                context_source += f" - {context.specimen_event}"
            st.write(context_source)
            st.write(context.text)
            st.code(context.supporting_excerpt, language=None)
        for diagnostic in evidence.rejected_applicability_contexts:
            st.warning(f"Rejected applicability context: {diagnostic.reason}")
            if diagnostic.source_verified and diagnostic.supporting_excerpt:
                st.code(diagnostic.supporting_excerpt, language=None)


def display_recommendation_draft(
    result: RecommendationDraftResult,
    review: SavedReviewDecision | None,
    analysis: ReportAnalysisResult,
) -> None:
    """Render a pending draft and exact source evidence."""

    st.subheader("Recommendation Agent draft")
    st.warning(result.status)
    st.caption(result.guidance_label)
    st.info(result.disclaimer)
    if review is None:
        st.write("Human review status: Pending review")
    else:
        reviewed_at = review.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        st.write(
            f"Human review status: {review.decision.capitalize()} at {reviewed_at}"
        )
        if review.notes:
            st.write(f"Review notes: {review.notes}")
    if result.abstention is not None:
        category = result.abstention.category.value.replace("_", " ")
        st.warning(
            f"No draft suggestion was produced ({category}): {result.abstention.reason}"
        )
    for position, suggestion in enumerate(result.suggestions, start=1):
        st.write(f"{position}. {suggestion.text}")
        uncertainty_notes = list(
            dict.fromkeys(
                note
                for evidence in suggestion.patient_evidence
                for note in evidence.uncertainty_notes
            )
        )
        for note in uncertainty_notes:
            st.warning(note)
        with st.expander(f"Evidence for suggestion {position}"):
            st.markdown("**Patient evidence**")
            for evidence in suggestion.patient_evidence:
                st.write(
                    f"{evidence.display_name} - {evidence.finding_id} - "
                    f"page {evidence.source_page}"
                )
                st.code(evidence.supporting_excerpt, language=None)
            st.markdown("**Guideline evidence**")
            for evidence in suggestion.guideline_evidence:
                st.caption(SYNTHETIC_GUIDANCE_LABEL)
                st.write(
                    f"{evidence.document_name} - page {evidence.page_number} - "
                    f"chunk {evidence.chunk_id} - document {evidence.document_hash}"
                )
                st.code(evidence.exact_passage, language=None)
    referenced = {
        finding_id
        for suggestion in result.suggestions
        for finding_id in suggestion.finding_ids
    }
    rejected = [
        (finding, diagnostic)
        for finding in analysis.findings
        if finding.finding_id in referenced
        for diagnostic in finding.rejected_applicability_contexts
    ]
    if rejected:
        with st.expander("Rejected report-context diagnostics", expanded=False):
            st.caption("These diagnostics are not accepted guideline support.")
            for finding, diagnostic in rejected:
                st.warning(f"{finding.name}: {diagnostic.reason}")


st.set_page_config(page_title="Clinical Report PDF Ingestion", page_icon="PDF")

st.title("Clinical Report PDF Ingestion")
st.caption("Synthetic-data capstone demo - Gemini analysis runs only on request")
st.info(
    "Upload synthetic reports only. Original PDF bytes remain in memory and are "
    "not saved. Successfully validated analyses and summaries, including source "
    "excerpts, plus recommendation drafts and review decisions are stored in a "
    "local SQLite database. This local store does not "
    "provide shared-user isolation."
)

st.header("Clinical Report Upload")

uploaded_file = st.file_uploader(
    "Upload one PDF",
    type=["pdf"],
    accept_multiple_files=False,
    help="Maximum size: 10 MiB. Encrypted PDFs are not supported.",
)

document = None
if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    document_identity = report_sha256(uploaded_bytes)
    synchronize_report_document(st.session_state, document_identity)
    try:
        document = process_pdf(
            uploaded_bytes,
            uploaded_file.name,
            max_size_bytes=DEFAULT_MAX_PDF_SIZE_BYTES,
        )
    except PdfIngestionError as exc:
        st.error(str(exc))
    else:
        filename_column, page_count_column = st.columns(2)
        filename_column.metric("Filename", document.filename)
        page_count_column.metric("Pages", document.page_count)

        for warning in document.warnings:
            st.warning(warning)

        st.subheader("Extracted text")
        for page in document.pages:
            with st.expander(f"Page {page.page_number}", expanded=True):
                for warning in page.warnings:
                    st.info(warning)
                if page.text.strip():
                    st.text_area(
                        f"Text from page {page.page_number}",
                        value=page.text,
                        height=220,
                        disabled=True,
                        key=f"page-{page.page_number}",
                    )
                else:
                    st.write("No extractable text was found on this page.")
else:
    synchronize_report_document(st.session_state, None)

if document is not None:
    st.subheader("Analyze extracted report")
    st.caption(
        "A compatible saved analysis is reused without a Gemini call. New analysis "
        "runs only when requested and consumes Gemini API quota. It does not provide "
        "a diagnosis or recommendation."
    )
    try:
        result_store = SQLiteResultStore()
    except PersistenceError as exc:
        st.error(str(exc))
        result_store = None

    if result_store is not None:
        try:
            config = load_gemini_config(
                streamlit_secrets=get_streamlit_secrets(),
            )
        except Exception as exc:  # noqa: BLE001 - sanitized provider boundary
            st.error(safe_report_analysis_error_message(exc))
        else:
            result_service = PersistedResultService(result_store)
            analysis_settings = analysis_generation_settings(config)
            analysis_context = analysis_compatibility_key(
                document_identity,
                analysis_settings,
                result_service.runtime_versions,
            )
            synchronize_analysis_context(st.session_state, analysis_context)

            analysis_record = st.session_state.get(REPORT_ANALYSIS_RECORD_KEY)
            if not isinstance(analysis_record, SavedAnalysisRecord):
                try:
                    compatible_analysis = result_store.load_analysis(
                        document_identity,
                        analysis_settings,
                        runtime_versions=result_service.runtime_versions,
                    )
                except PersistenceError as exc:
                    st.error(str(exc))
                else:
                    if compatible_analysis is not None:
                        remember_analysis(
                            AnalysisResolution(
                                record=compatible_analysis,
                                reused=True,
                            )
                        )

            analysis_record = st.session_state.get(REPORT_ANALYSIS_RECORD_KEY)
            analyze_label = (
                "Reanalyze Report"
                if isinstance(analysis_record, SavedAnalysisRecord)
                else "Analyze Report"
            )
            if st.button(analyze_label, type="primary"):
                with st.spinner("Running the Report Analysis Agent..."):
                    try:
                        resolution = result_service.resolve_analysis(
                            document_identity,
                            analysis_settings,
                            lambda: ReportAnalysisAgent(config).analyze(document),
                            force=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - sanitized UI boundary
                        st.error(safe_report_analysis_error_message(exc))
                    else:
                        remember_analysis(resolution)
                        clear_summary_result(st.session_state)

            analysis_record = st.session_state.get(REPORT_ANALYSIS_RECORD_KEY)
            analysis_result = st.session_state.get(REPORT_ANALYSIS_RESULT_KEY)
            if isinstance(analysis_record, SavedAnalysisRecord) and isinstance(
                analysis_result,
                ReportAnalysisResult,
            ):
                display_saved_status(
                    "analysis",
                    analysis_record,
                    reused=bool(
                        st.session_state.get(REPORT_ANALYSIS_REUSED_KEY, False)
                    ),
                )
                display_report_analysis(analysis_result)
                st.subheader("Generate report summary")
                st.caption(
                    "A compatible summary is reused without a Gemini call. Summary "
                    "generation never reruns report analysis and does not query "
                    "guidelines."
                )
                summary_settings = summary_generation_settings(config)
                summary_context = summary_compatibility_key(
                    analysis_record,
                    summary_settings,
                    result_service.runtime_versions,
                )
                synchronize_summary_context(st.session_state, summary_context)

                summary_record = st.session_state.get(REPORT_SUMMARY_RECORD_KEY)
                if not isinstance(summary_record, SavedSummaryRecord):
                    try:
                        compatible_summary = result_store.load_summary(
                            analysis_record,
                            summary_settings,
                            runtime_versions=result_service.runtime_versions,
                        )
                    except PersistenceError as exc:
                        st.error(str(exc))
                    else:
                        if compatible_summary is not None:
                            remember_summary(
                                SummaryResolution(
                                    record=compatible_summary,
                                    reused=True,
                                )
                            )

                summary_record = st.session_state.get(REPORT_SUMMARY_RECORD_KEY)
                summary_label = (
                    "Regenerate Summary"
                    if isinstance(summary_record, SavedSummaryRecord)
                    else "Generate Summary"
                )
                if st.button(summary_label, type="primary"):
                    with st.spinner("Running the Summary Agent..."):
                        try:
                            summary_resolution = result_service.resolve_summary(
                                analysis_record,
                                summary_settings,
                                lambda: SummaryAgent(config).summarize(analysis_result),
                                force=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - sanitized UI boundary
                            st.error(safe_summary_error_message(exc))
                        else:
                            remember_summary(summary_resolution)

                summary_record = st.session_state.get(REPORT_SUMMARY_RECORD_KEY)
                summary_result = st.session_state.get(REPORT_SUMMARY_RESULT_KEY)
                if isinstance(summary_record, SavedSummaryRecord) and isinstance(
                    summary_result,
                    ReportSummaryResult,
                ):
                    display_saved_status(
                        "summary",
                        summary_record,
                        reused=bool(
                            st.session_state.get(REPORT_SUMMARY_REUSED_KEY, False)
                        ),
                    )
                    display_report_summary(summary_result)

                    st.subheader("Generate draft recommendations")
                    st.caption(
                        "This queries the local guideline index, then runs the third "
                        "and final AI agent only when requested. It never reruns "
                        "analysis or summary. API generation consumes Gemini quota."
                    )
                    try:
                        recommendation_kb = get_guideline_knowledge_base()
                        index_identity = recommendation_kb.index_fingerprint()
                        recommendation_settings = recommendation_generation_settings(
                            config,
                            guideline_index_fingerprint=index_identity.fingerprint,
                            guideline_document_hashes=index_identity.document_hashes,
                        )
                        recommendation_context = recommendation_compatibility_key(
                            analysis_record,
                            summary_record,
                            recommendation_settings,
                            result_service.runtime_versions,
                        )
                        synchronize_recommendation_context(
                            st.session_state, recommendation_context
                        )
                    except (GuidelineKnowledgeBaseError, PersistenceError) as exc:
                        st.error(str(exc))
                        recommendation_settings = None

                    recommendation_record = st.session_state.get(
                        REPORT_RECOMMENDATION_RECORD_KEY
                    )
                    if recommendation_settings is not None and not isinstance(
                        recommendation_record, SavedRecommendationDraft
                    ):
                        try:
                            compatible_draft = result_store.load_recommendation(
                                analysis_record,
                                summary_record,
                                recommendation_settings,
                                runtime_versions=result_service.runtime_versions,
                            )
                        except PersistenceError as exc:
                            st.error(str(exc))
                        else:
                            if compatible_draft is not None:
                                remember_recommendation(
                                    RecommendationResolution(
                                        record=compatible_draft, reused=True
                                    )
                                )

                    recommendation_record = st.session_state.get(
                        REPORT_RECOMMENDATION_RECORD_KEY
                    )
                    generate_label = (
                        "Regenerate Recommendations"
                        if isinstance(recommendation_record, SavedRecommendationDraft)
                        else "Generate Recommendations"
                    )
                    if st.button(
                        generate_label,
                        type="primary",
                        disabled=recommendation_settings is None,
                    ):
                        with st.spinner(
                            "Retrieving guidelines and running the Recommendation Agent..."
                        ):
                            try:
                                workflow = RecommendationWorkflow(
                                    recommendation_kb,
                                    RecommendationAgent(config),
                                    top_k=recommendation_settings.retrieval_top_k,
                                    max_queries=(
                                        recommendation_settings.max_retrieval_queries
                                    ),
                                )
                                recommendation_resolution = (
                                    result_service.resolve_recommendation(
                                        analysis_record,
                                        summary_record,
                                        recommendation_settings,
                                        lambda: workflow.generate(
                                            analysis_result, summary_result
                                        ),
                                        force=True,
                                    )
                                )
                            except Exception as exc:  # noqa: BLE001
                                st.error(safe_recommendation_error_message(exc))
                            else:
                                remember_recommendation(recommendation_resolution)

                    recommendation_record = st.session_state.get(
                        REPORT_RECOMMENDATION_RECORD_KEY
                    )
                    recommendation_result = st.session_state.get(
                        REPORT_RECOMMENDATION_RESULT_KEY
                    )
                    if isinstance(
                        recommendation_record, SavedRecommendationDraft
                    ) and isinstance(recommendation_result, RecommendationDraftResult):
                        display_saved_status(
                            "recommendation draft",
                            recommendation_record,
                            reused=bool(
                                st.session_state.get(
                                    REPORT_RECOMMENDATION_REUSED_KEY, False
                                )
                            ),
                        )
                        try:
                            review = result_store.load_latest_review(
                                recommendation_record
                            )
                        except PersistenceError as exc:
                            st.error(str(exc))
                            review = None
                        st.session_state[REPORT_RECOMMENDATION_REVIEW_KEY] = review
                        display_recommendation_draft(
                            recommendation_result, review, analysis_result
                        )
                        if recommendation_result.suggestions:
                            with st.expander(
                                "Edit draft (creates a new pending version)",
                                expanded=False,
                            ):
                                edited_texts = [
                                    st.text_area(
                                        f"Suggestion {position}",
                                        value=suggestion.text,
                                        max_chars=500,
                                        key=(
                                            f"edit-{recommendation_record.record_id}-"
                                            f"{position}"
                                        ),
                                    )
                                    for position, suggestion in enumerate(
                                        recommendation_result.suggestions,
                                        start=1,
                                    )
                                ]
                                if st.button(
                                    "Save as new draft version",
                                    key=f"edit-save-{recommendation_record.record_id}",
                                ):
                                    try:
                                        edited_result = create_edited_draft(
                                            recommendation_result, edited_texts
                                        )
                                        edited_record = (
                                            result_store.save_edited_recommendation(
                                                recommendation_record,
                                                edited_result,
                                            )
                                        )
                                    except (
                                        PersistenceError,
                                        RecommendationError,
                                        ValueError,
                                    ) as exc:
                                        st.error(safe_recommendation_error_message(exc))
                                    else:
                                        remember_recommendation(
                                            RecommendationResolution(
                                                record=edited_record,
                                                reused=False,
                                            )
                                        )
                                        st.rerun()

                        st.markdown("### Human review")
                        st.caption(
                            "Approve means this exact draft version was reviewed; "
                            "it is not clinical certification. A new or edited "
                            "version requires fresh review."
                        )
                        review_notes = st.text_area(
                            "Review notes",
                            value="",
                            max_chars=2000,
                            key=f"review-notes-{recommendation_record.record_id}",
                        )
                        approve_column, reject_column = st.columns(2)
                        if approve_column.button(
                            "Approve exact draft",
                            key=f"approve-{recommendation_record.record_id}",
                        ):
                            try:
                                result_store.save_review_decision(
                                    recommendation_record,
                                    "approved",
                                    review_notes,
                                )
                            except PersistenceError as exc:
                                st.error(str(exc))
                            else:
                                st.rerun()
                        if reject_column.button(
                            "Reject exact draft",
                            key=f"reject-{recommendation_record.record_id}",
                        ):
                            try:
                                result_store.save_review_decision(
                                    recommendation_record,
                                    "rejected",
                                    review_notes,
                                )
                            except PersistenceError as exc:
                                st.error(str(exc))
                            else:
                                st.rerun()

        st.subheader("Saved local results")
        st.caption(
            "Saved findings and source excerpts contain report data. The database "
            "is local to this app installation and is not a shared-user isolation "
            "boundary."
        )
        deletion_notice = st.session_state.pop(_REPORT_DELETION_NOTICE_KEY, None)
        if isinstance(deletion_notice, str):
            st.success(deletion_notice)
        deletion_confirmed = st.checkbox(
            "I understand this deletes every saved analysis, summary, recommendation "
            "draft, and review decision for this exact report.",
            key=f"delete-results-confirmed-{document_identity}",
        )
        if st.button(
            "Delete saved results for this report",
            disabled=not deletion_confirmed,
        ):
            try:
                deleted = result_store.delete_report(document_identity)
            except PersistenceError as exc:
                st.error(str(exc))
            else:
                clear_analysis_result(st.session_state)
                st.session_state[_REPORT_DELETION_NOTICE_KEY] = (
                    f"Deleted {deleted.analyses_deleted} saved analyses and "
                    f"{deleted.summaries_deleted} saved summaries, "
                    f"{deleted.recommendation_drafts_deleted} drafts, and "
                    f"{deleted.review_decisions_deleted} review decisions for "
                    "this report."
                )
                st.rerun()

st.divider()
st.header("Guideline Knowledge Base")
st.caption(
    "Index selectable-text synthetic guideline PDFs for local retrieval. "
    "Retrieved text is not authoritative medical guidance."
)

try:
    knowledge_base = get_guideline_knowledge_base()
except GuidelineKnowledgeBaseError as exc:
    st.error(str(exc))
    st.stop()

st.write(f"Indexed guideline documents: {knowledge_base.indexed_document_count}")
guideline_file = st.file_uploader(
    "Upload one guideline PDF",
    type=["pdf"],
    accept_multiple_files=False,
    key="guideline-upload",
    help="Selectable-text PDFs only. Guideline chunks persist in local Chroma.",
)

if st.button(
    "Index guideline",
    disabled=guideline_file is None,
    type="primary",
):
    with st.status(
        "Extracting, chunking, and embedding the guideline...",
        expanded=True,
    ) as indexing_status:
        st.write(
            "The first indexing run downloads the local embedding model. "
            "Later runs use the cached model and do not call Gemini."
        )
        try:
            index_result = knowledge_base.index_pdf(
                guideline_file.getvalue(),
                guideline_file.name,
            )
        except (PdfIngestionError, GuidelineKnowledgeBaseError) as exc:
            indexing_status.update(label="Indexing failed", state="error")
            st.error(str(exc))
        else:
            indexing_status.update(label="Indexing complete", state="complete")
            if index_result.duplicate:
                st.info("This exact guideline is already indexed; nothing was added.")
            else:
                st.success(
                    f"Indexed {index_result.chunks_added} chunks from "
                    f"{index_result.page_count} pages."
                )

st.subheader("Inspect retrieved passages")
retrieval_query = st.text_input(
    "Retrieval query",
    placeholder="Enter a topic to search in indexed guidelines",
)
top_k = st.slider("Passages to return", min_value=1, max_value=10, value=4)

if st.button("Retrieve passages"):
    try:
        passages = knowledge_base.retrieve(retrieval_query, top_k=top_k)
    except GuidelineKnowledgeBaseError as exc:
        st.error(str(exc))
    else:
        if not passages:
            st.info("No passages were returned for this query.")
        for position, passage in enumerate(passages, start=1):
            source = (
                f"{passage.document_name} - page {passage.page_number} - "
                f"chunk {passage.chunk_id}"
            )
            with st.expander(f"Passage {position}: {source}", expanded=True):
                st.write(passage.text)
