# AI Clinical Report Summarization Assistant

This shared capstone processes **synthetic clinical reports only**. The first
milestone safely validates PDFs and extracts their text locally, page by page.
Uploaded files stay in memory and are never written by the application.

## Current milestone

Included:

- PDF content validation with a configurable 10 MiB limit
- clear errors for empty, oversized, non-PDF, corrupt, and encrypted uploads
- page-by-page extraction with `pypdf`
- warnings for blank or image-only pages that may need OCR
- Pydantic result models
- a minimal Streamlit uploader
- synthetic pytest fixtures and tests

Not included yet: Gemini calls, CrewAI agents, RAG, Chroma workflows, SQLite
persistence, OCR, deployment, or paid services.

## Windows 11 setup

Install Python 3.12, uv, and Git before continuing. Open PowerShell in this
repository and confirm that Windows can find them:

```powershell
Get-Command uv
Get-Command git
uv --version
git --version
```

On the development machine used for this milestone, the existing executables
were found at:

```text
C:\Users\21r21\.local\bin\uv.exe
C:\Program Files\Git\cmd\git.exe
```

If `Get-Command` cannot find a tool, do not reinstall it immediately. Locate
the existing executable, then either invoke it by its full path or add its
containing directory to the user PATH through **Settings > System > About >
Advanced system settings > Environment Variables**. Open a new PowerShell
window afterward. The project does not modify permanent PATH settings.

Create or update the project environment from the lockfile:

```powershell
uv sync
uv lock --check
uv sync --check
```

If uv is not yet on PATH, the equivalent command format is:

```powershell
& 'C:\Users\21r21\.local\bin\uv.exe' sync
```

## Run the application

```powershell
uv run streamlit run src/clinical_report_assistant/app.py
```

Open the local URL printed by Streamlit, upload one synthetic PDF, and review
the metadata, warnings, and extracted text. Press `Ctrl+C` in PowerShell to
stop the server.

## Run the checks

```powershell
uv run pytest
uv run ruff check .
uv lock --check
```

The tests generate PDFs in memory with synthetic phrases. They do not require
real clinical data, network access, API keys, Gemini, or any paid service.

## Project layout

```text
src/clinical_report_assistant/pdf_processing.py  validation and extraction
src/clinical_report_assistant/app.py             Streamlit presentation layer
tests/                                           synthetic fixtures and tests
.env.example                                     placeholder-only environment file
```

`pdf_processing.py` is reusable and contains no Streamlit code. The UI reads
an upload into bytes and passes those bytes to the processing module. Neither
layer logs document text.

## PDF behavior and limitations

- The filename is display metadata; the module checks the uploaded bytes and
  parses them as a PDF.
- Encrypted PDFs are rejected even if they could be opened with an empty
  password.
- A readable blank or image-only PDF is accepted with an OCR warning.
- OCR is intentionally not implemented. Scanned reports will not yield text
  until a future, separately reviewed OCR milestone.
- PDF extraction is not clinical interpretation and must not be treated as
  medical advice.

## Secrets and local data

Copy `.env.example` to `.env` only for a future milestone that needs local
configuration. `.env`, Streamlit secrets, uploaded documents, outputs,
databases, Chroma storage, and tool caches are ignored by Git. Never put real
patient data or a real API key in this capstone repository.
