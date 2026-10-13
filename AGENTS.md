# Project guidance

## Stack

- Python 3.12 with a uv-managed src-layout project
- Streamlit for the local UI
- Pydantic for boundary models
- pypdf for offline PDF extraction
- pytest and Ruff for verification
- Future agreed components: CrewAI, Gemini API, LangChain, ChromaDB, SQLite
- Ollama is not part of this project and must not be added

## Module boundaries

- `pdf_processing.py` owns byte-level validation, pypdf parsing, extraction,
  warnings, exceptions, and Pydantic result models. It must not import
  Streamlit or make network calls.
- `app.py` is a thin presentation layer. It keeps uploads in memory and calls
  the processing module; extraction rules do not belong in the UI.
- `tests/` uses synthetic data only. Generate PDF fixtures in memory and never
  add real clinical documents.
- Future Gemini, agent, retrieval, and persistence modules must remain
  separate from PDF ingestion and require their own reviewed milestone.

## Privacy and safety

- Do not log uploaded bytes or extracted document contents.
- Do not persist uploads unless a future requirement explicitly authorizes it.
- Keep secrets in ignored local configuration; `.env.example` contains only
  placeholders.
- Do not enable paid services automatically.
- OCR is out of scope for the current milestone; explain the limitation when
  a PDF has no extractable text.

## Required checks

Run these before handing off changes:

```powershell
uv run pytest
uv run ruff check .
uv lock --check
```

For UI changes, also start Streamlit locally, confirm that its health endpoint
responds, and stop the server afterward.
