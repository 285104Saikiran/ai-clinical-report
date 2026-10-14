"""Optional real-model integration test; skipped during normal offline checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clinical_report_assistant.guideline_kb import GuidelineKnowledgeBase


@pytest.mark.skipif(
    os.getenv("RUN_GUIDELINE_INTEGRATION") != "1",
    reason="Set RUN_GUIDELINE_INTEGRATION=1 to use the real local embedding model.",
)
def test_real_embedding_model_with_temporary_chroma(
    tmp_path: Path,
    synthetic_guideline_pdf: bytes,
) -> None:
    knowledge_base = GuidelineKnowledgeBase(storage_path=tmp_path)

    result = knowledge_base.index_pdf(
        synthetic_guideline_pdf,
        "synthetic-guideline.pdf",
    )
    passages = knowledge_base.retrieve("fictional hydration", top_k=1)

    assert result.chunks_added > 0
    assert len(passages) == 1
    assert passages[0].document_name == "synthetic-guideline.pdf"
    assert passages[0].page_number in {1, 2}
