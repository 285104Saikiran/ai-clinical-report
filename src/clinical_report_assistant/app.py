'''Minimal Streamlit interface for offline PDF ingestion.'''

from __future__ import annotations

import streamlit as st

from clinical_report_assistant.pdf_processing import (
    DEFAULT_MAX_PDF_SIZE_BYTES,
    PdfIngestionError,
    process_pdf,
)

st.set_page_config(page_title='Clinical Report PDF Ingestion', page_icon='PDF')

st.title('Clinical Report PDF Ingestion')
st.caption('Synthetic-data capstone demo — offline extraction only')
st.info(
    'Upload synthetic reports only. Files remain in memory and are not saved. '
    'This milestone does not use Gemini, agents, RAG, or any paid service.'
)

uploaded_file = st.file_uploader(
    'Upload one PDF',
    type=['pdf'],
    accept_multiple_files=False,
    help='Maximum size: 10 MiB. Encrypted PDFs are not supported.',
)

if uploaded_file is not None:
    try:
        document = process_pdf(
            uploaded_file.getvalue(),
            uploaded_file.name,
            max_size_bytes=DEFAULT_MAX_PDF_SIZE_BYTES,
        )
    except PdfIngestionError as exc:
        st.error(str(exc))
    else:
        filename_column, page_count_column = st.columns(2)
        filename_column.metric('Filename', document.filename)
        page_count_column.metric('Pages', document.page_count)

        for warning in document.warnings:
            st.warning(warning)

        st.subheader('Extracted text')
        for page in document.pages:
            with st.expander(f'Page {page.page_number}', expanded=True):
                for warning in page.warnings:
                    st.info(warning)
                if page.text.strip():
                    st.text_area(
                        f'Text from page {page.page_number}',
                        value=page.text,
                        height=220,
                        disabled=True,
                        key=f'page-{page.page_number}',
                    )
                else:
                    st.write('No extractable text was found on this page.')
