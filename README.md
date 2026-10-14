# AI Clinical Report Summarization Assistant

This shared capstone processes **synthetic clinical reports only**. The first
milestone safely validates PDFs and extracts their text locally, page by page.
Uploaded PDF bytes stay in memory and are never written by the application.
Successfully validated analyses, summaries, recommendation drafts, and human
review decisions are saved in an ignored local SQLite database. Saved records
may contain report-derived text and source excerpts.

## Current milestone

Included:

- PDF content validation with a configurable 10 MiB limit
- clear errors for empty, oversized, non-PDF, corrupt, and encrypted uploads
- page-by-page extraction with `pypdf`
- warnings for blank or image-only pages that may need OCR
- Pydantic result models
- a minimal Streamlit uploader
- synthetic pytest fixtures and tests
- compatible local reuse of all validated agent results through SQLite

Milestone 2 adds a small Gemini connection layer and an explicit synthetic
smoke-test command. It does not connect uploaded PDFs or the Streamlit UI to
Gemini.

Milestone 3 adds persistent local guideline indexing and retrieval with
LangChain, Chroma, and `sentence-transformers/all-MiniLM-L6-v2`. This model
creates embeddings only; Gemini remains the generation provider.

Milestone 4 adds the Report Analysis Agent, milestone 5 adds the Summary Agent,
and milestone 6 adds the Recommendation Agent plus human review. These are the
project's exactly three AI agents. PDF processing, Gemini access, retrieval,
state management, and persistence are supporting components, not agents.

Not included: autonomous clinical decision-making, real clinical guidance,
OCR, deployment, shared-user isolation, or clinical certification.

## Report Analysis Agent

The Streamlit app reuses an exactly compatible saved analysis when one exists.
Otherwise it analyzes a report only after **Analyze Report** is pressed. Report
identity is the SHA-256 of uploaded bytes, not the filename: renamed identical
bytes match, while different bytes with the same filename do not. Result objects
stay in the current browser session and are never placed in a shared Streamlit
cache.

The agent uses one sequential CrewAI task with the explicitly configured
native Gemini provider. Delegation, tools, memory, planning, tracing, and agent
retries are disabled; iterations, request time, and output tokens are bounded.
Report text is serialized as untrusted data and is never treated as agent
instructions. No OpenAI model is selected and no embedding provider is
initialized by report analysis.

CrewAI returns Pydantic-structured candidate measurements and a bounded set of
narrative facts. Deterministic Python verifies cited pages, excerpts, reported
values, dates, units, reference text, applicability context, narrative
statements, and attribution against the extracted report. The original
reference text is retained separately from parsed numeric bounds and the final
assessment. Text such as `see age-specific worksheet` is therefore preserved
instead of being replaced with a fabricated missing-range phrase.

Plain decimal values are compared only with source-backed two-sided ranges
written as `lower-upper`, `lower to upper`, or with a supported dash. Numeric
comparison and applicability are separate: for example, `118 mg/dL` compared
with `70-99 mg/dL` remains numerically above range even when unknown fasting
status produces an independent applicability-review warning. This arithmetic is
not a diagnosis. Missing ranges, unsupported formats, reversed bounds, and
incomplete or mismatched units remain unassessed rather than guessed. No
supplied range means only that the measurement cannot be classified by this
program; it is not automatically an extraction error or an abnormal result.

Measurement evidence is stored separately from reference evidence. A
value-bearing excerpt cannot become a supplied range merely because it appears
on the same page. When no range is supplied, numeric range fields remain empty,
the UI displays `Not supplied`, and any verified explicit absence statement is
retained separately. Reported units are checked in measurement evidence;
reference-range units are checked in reference evidence. A missing range,
unsupported compound value, missing reported unit, missing range unit, unit
mismatch, and conditional applicability therefore remain distinct conditions.

Each applicability item has its own cited page and excerpt. It normally names
the analyte using token-bounded evidence. A context that directly follows a
measurement may also be associated when source position is unique and a
source-verified specimen event distinguishes the repeated analyte. A shared page
or specimen event alone is not enough. This avoids short names such as AST
matching text inside a word such as fasting. Source-supported context returned
under the wrong finding is reassociated only to the specifically evidenced
result. Ambiguous context is not guessed. Rejected, source-verified context is
kept in detailed diagnostics but does not affect applicability. Absent context
is represented by an empty collection and creates no warning.

The same single agent may also extract symptoms, reported history, medication
uncertainty, allergy-verification status, and explicitly missing follow-up.
These narrative facts retain attribution and distinguish `not recorded` from a
negative finding. They stay separate from numeric measurements. Explicitly
labeled training controls are also displayed separately and excluded from the
patient-measurement list consumed by the Summary Agent. Repeated patient
measurements from distinct dates or specimens are not deduplicated.

Source validation establishes only that the output is grounded in extracted
report text. It does not establish that PDF extraction, the report itself, the
model output, or any clinical interpretation is accurate. This capstone does
not provide diagnoses or recommendations.

Official CrewAI references:

- [CrewAI LLM configuration](https://docs.crewai.com/en/concepts/llms)
- [CrewAI agents and execution controls](https://docs.crewai.com/en/concepts/agents)
- [CrewAI Pydantic task output](https://docs.crewai.com/en/concepts/tasks)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini finish reasons and usage metadata](https://ai.google.dev/api/generate-content)

Report analysis has a separate output budget from the short Gemini smoke test.
It defaults to `8192` tokens and can be set in the process environment,
Streamlit secrets, or local `.env`:

```dotenv
REPORT_ANALYSIS_MAX_OUTPUT_TOKENS=8192
```

The same configuration precedence applies: process environment, then
Streamlit secrets, then `.env`. The accepted range is 1 through 65,536; the
configured value must also fit the selected model's documented output limit.
The current default is within Gemini 3.5 Flash-Lite's documented 65,536-token
limit. This setting does not change the direct smoke test's 512-token budget.
The analysis request timeout remains 60 seconds, SDK retries remain disabled,
and the app does not switch models automatically.

If Gemini reports `MAX_TOKENS`, `CONTINUATION`, or `MALFORMED_RESPONSE`, or if
Pydantic sees truncated JSON, the entire analysis is rejected. The application
does not append missing JSON delimiters and never displays partial findings as
a complete result. Logs retain only safe categories, finish reason, token counts,
and elapsed time when those values are available.

## Summary Agent

After a valid analysis exists, **Generate Summary** runs the second CrewAI
agent only when no compatible saved summary exists. It does not rerun Report
Analysis and does not query the guideline index. The analysis remains available
if summary generation fails. Results are never placed in a shared Streamlit
cache. A changed report or analysis clears the session summary; changing the
configured summary model, output budget, prompt/schema version, schema, or
relevant dependency versions prevents incompatible saved reuse.

The Summary Agent receives validated patient measurements, applicability
context, narrative findings, and grouped review warnings. Explicitly labeled
training controls are omitted from its input. The model returns only bounded
lists of existing finding IDs. It cannot supply free-form clinical claims.
Deterministic Python then reconstructs displayed values, units, supplied ranges,
dates, specimen events, attribution, uncertainty text, source pages, and
evidence from the validated Report Analysis result.

All outside-range patient measurements must be selected by the model or the
whole summary is rejected. Narrative, repeat-group, and uncertainty selections
are optional organizational choices. Deterministic Python derives every
mandatory uncertainty and compatible repeat group from the validated analysis,
then adds every required member even when the model selects none. A finding may
therefore appear in outside-range, repeat-comparison, and applicability
uncertainty sections.

Repeat identity uses conservatively normalized analyte names and matching
reported units; units are never converted. Different analytes, incompatible
units, unitless measurements, and training controls are not combined. Complete,
unique ISO dates establish chronological order. Otherwise, results remain in
report order and are explicitly labeled as having unknown chronology. Each
comparison retains values, units, dates, specimen events, supplied ranges,
review reasons, finding IDs, pages, and evidence. A simple numeric difference
is shown only for two dated, plain-numeric, compatible-unit results and does not
claim recovery or causation.

Similar uncertainties are grouped, but every finding ID, source page, and
evidence item is retained. Measurements without a supplied range remain
unassessed and are not described as normal or abnormal.

The default Streamlit view shows the overview, one global safety disclaimer, a
compact findings/trends table, and grouped uncertainties with readable finding
names. Applicable warnings remain visible in those tables. Full outside-range
items, repeat groups, narratives, uncertainties, primary evidence, and separate
applicability evidence remain available in collapsed expanders.

Unknown IDs, control IDs, category mismatches, malformed or truncated JSON,
and token-limited responses are rejected. Safe reference errors include only a
category and missing/misclassified counts, never report content. Numeric
comparisons remain comparisons with report-supplied ranges, not diagnoses.
Unknown fasting status is displayed alongside the affected initial glucose
finding, and `not recorded` remains missing information rather than a negative
finding. Repeat-result changes are descriptive and do not claim that treatment
or another cause produced them.

Array-size bounds are enforced by Pydantic after the response rather than sent
as cumulative `minItems` and `maxItems` schema constraints. Some Gemini models
reject otherwise valid schemas when those cumulative constraints exceed an
undocumented request budget. Application-side validation preserves the same
bounds and rejects the whole response rather than accepting excess items.

Summary generation has its own output limit:

```dotenv
SUMMARY_MAX_OUTPUT_TOKENS=4096
```

The accepted range is 1 through 65,536 and the selected model must support the
configured value. The timeout is 60 seconds, SDK and agent retries are disabled,
automatic context summarization is disabled so inputs are not silently reduced,
and the app does not switch models. Gemini quota is consumed only after the
explicit button press. A source link proves only that the displayed evidence
came from the validated analysis; it does not prove clinical accuracy.

## Recommendation Agent and human review

After a valid saved summary exists, **Generate Recommendations** performs
bounded local retrieval and then runs the third and final CrewAI agent. It never
reruns Report Analysis or Summary. The agent receives validated patient
findings, the validated summary, and bounded retrieved passages. Training
controls remain excluded. Report excerpts and guideline passages are framed as
untrusted data, so instructions embedded in either source are not agent
instructions.

Each accepted suggestion cites existing patient finding IDs and retrieved
passage IDs. Deterministic validation checks patient excerpts, exact passage
text, document name, document hash, page, and chunk ID. A retrieved source is
provenance, not proof of relevance or clinical correctness. Drafts cannot add
diagnoses, medication changes, doses, or urgency. Relevant analysis uncertainty
stays visible. When guidance is absent, clearly unrelated, contradictory, or
insufficient, the valid result is an explicit abstention. The River-7 hydration
training guideline is intentionally unrelated to laboratory management and
therefore produces an abstention for the synthetic lab report.

Every generated or edited result is labeled **Draft / Pending review** and
**SYNTHETIC TRAINING GUIDANCE - NOT MEDICAL ADVICE**. **Approve exact draft**
and **Reject exact draft** save a decision and optional notes for that immutable
draft version. Approval means a person reviewed that version; it is not clinical
certification. Editing saves a new version with fresh pending status.
Regeneration also creates a new version and never inherits an earlier approval.

Recommendation generation has its own bounded output limit:

```dotenv
RECOMMENDATION_MAX_OUTPUT_TOKENS=4096
```

Its accepted range is 1 through 65,536 and must fit the selected model. The
request timeout is 60 seconds. SDK retries, agent retries, delegation, tools,
memory, planning, tracing, and automatic model switching are disabled. Local
retrieval does not consume Gemini quota; a Gemini call happens only after the
explicit generation button is pressed and relevant passages reach the agent.

Rejected recommendation output includes metadata-only diagnostics such as
`category`, Pydantic `field` and `code`, error `count`, provider finish reason,
and safe token counts when CrewAI exposes them. Categories distinguish malformed
or truncated JSON, empty or blocked responses, token limits, schema mismatches,
missing CrewAI structured output, unknown finding or passage IDs, unsupported
evidence, and unsupported content. Diagnostics never include Pydantic input
values, report text, guideline passages, raw responses, prompts, or credentials.
If CrewAI returns without a parsed Pydantic object after its SDK boundary has
already completed, finish reason and token usage are no longer available to the
application; that case is reported as `missing_structured_output` without
inventing provider metadata.

Run the synthetic-only structured-output diagnostic tests without any API call:

```powershell
uv run pytest tests/test_recommendation_agent.py -k "structured or finish_reason or missing_crewai or logs_only" -vv
```

For a manual synthetic retest, generate and index the capstone review PDF as
described below, then use the explicit **Regenerate Recommendations** control.
Copy only the sanitized diagnostic category and metadata when reporting a
failure; do not copy report content or retrieved passages.

## Local SQLite result persistence

The app creates `data/local_results/results.sqlite3` on demand. Git ignores this
directory and SQLite sidecar files. The database stores only successfully
validated analysis, summary, and recommendation models; their source evidence;
immutable draft versions; review decisions and notes; creation times; and
non-secret compatibility metadata. It does **not** store the API key or original
PDF bytes. Saved findings, narrative statements, warnings, suggestions, review
notes, guideline passages, and excerpts are still report-related data; use
synthetic inputs only and protect or delete the database as appropriate.

Analysis reuse requires an exact match for the report byte hash, configured
model, output limit, timeout/retry settings, prompt and schema versions, actual
Pydantic schema hash, and recorded Python/application/provider dependency
versions. Summary reuse additionally requires the exact immutable saved analysis
record and analysis-result hash plus matching summary settings. Recommendation
reuse additionally requires the exact analysis and summary records and result
hashes, model and generation settings, recommendation prompt/schema versions,
retrieval settings, complete guideline index fingerprint, and sorted guideline
document hashes. Adding, removing, or changing guideline documents prevents the
old draft from being current. Every saved load is parsed through Pydantic and
source relationships are revalidated. Corrupt or incompatible records are
rejected rather than reused.
Analysis prompt/schema compatibility is currently `v2`, as is affected summary
compatibility. Older rows remain in SQLite but are not silently reused as current.

When a compatible record is found, the UI labels it as reused and shows its
creation time without making a Gemini call. **Reanalyze Report** and
**Regenerate Summary** and **Regenerate Recommendations** each make one new
explicit generation attempt. A failed attempt leaves every prior valid upstream
result and draft displayed and saved. Summary generation cannot rerun analysis;
recommendation generation cannot rerun either upstream agent. The confirmed
**Delete saved results for this report** control deletes all saved analyses for
that byte hash and cascades to summaries, drafts, and review decisions. It does
not delete the in-memory upload because no upload is stored.

This is local application storage, not per-user or shared-user isolation. Anyone
with access to the database file can potentially read its report-derived JSON.
No deployment, encryption-at-rest, access control, multi-user tenancy, or backup
policy is provided by this capstone.

## Guideline knowledge base

The Streamlit app has a separate **Guideline Knowledge Base** section. It:

- accepts selectable-text guideline PDFs through the same validated PDF layer
- chunks within each page and preserves filename, page, hash, and chunk ID
- avoids re-indexing an identical document hash
- persists vectors under `data/chroma_guidelines/`, which Git ignores
- retrieves configurable top-k passages without calling Gemini
- displays source references without presenting scores as confidence

The index manifest records the embedding model, normalization, splitter,
chunk size, overlap, schema version, and successfully indexed document hashes.
If those settings differ from an existing index, the app refuses to open it
and asks you to use a fresh directory rather than mixing incompatible vectors.

The first real index or query downloads
`sentence-transformers/all-MiniLM-L6-v2` from Hugging Face and therefore needs
internet access. Model weights are cached by the Hugging Face tooling. After
that download, embedding and Chroma retrieval run locally and consume no
Gemini API quota.

Official references:

- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [LangChain local Hugging Face embeddings](https://docs.langchain.com/oss/python/integrations/embeddings/huggingfacehub)
- [Chroma collections](https://docs.trychroma.com/docs/collections/manage-collections)

## Gemini configuration

This project uses the supported `google-genai` Python SDK. It has no default
model: you must explicitly set both `GEMINI_API_KEY` and `GEMINI_MODEL`.
Choose a model that is available to your own project in Google AI Studio or
the official Gemini model documentation. This project does not guess which
models have free quota for your account.

For local development, copy `.env.example` to `.env` and replace both
placeholders:

```dotenv
GEMINI_API_KEY=your_local_key
GEMINI_MODEL=your_explicit_model_id
REPORT_ANALYSIS_MAX_OUTPUT_TOKENS=8192
SUMMARY_MAX_OUTPUT_TOKENS=4096
RECOMMENDATION_MAX_OUTPUT_TOKENS=4096
```

For Streamlit deployment, configure server-side secrets instead:

```toml
GEMINI_API_KEY = 'your_deployment_key'
GEMINI_MODEL = 'your_explicit_model_id'
```

Configuration precedence, from highest to lowest, is:

1. Process environment variables
2. Values supplied from Streamlit `st.secrets`
3. Local `.env`

Keys are loaded only on the server side and are never included in provider
errors. `.env` and `.streamlit/secrets.toml` are ignored by Git.

Official references:

- [Gemini API key setup](https://ai.google.dev/gemini-api/docs/api-key)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

## Explicit Gemini smoke test

The following is the only live smoke-test command:

```powershell
uv run gemini-smoke --live
```

The command sends one short, built-in synthetic report. It never reads or
sends an uploaded PDF. Retries are disabled, so each invocation makes at most
one generation request. Omitting `--live` sends nothing.

Gemini API calls consume the configured project's quota and may incur charges
according to that project's account settings. This repository does not enable
billing or paid services; keep paid billing disabled if that is your policy.

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

For one manual synthetic-report analysis, first generate the ignored fixture:

```powershell
uv run python scripts/generate_synthetic_report.py
```

Start Streamlit, upload
`generated/synthetic_reports/synthetic_lab_report.pdf` under **Clinical Report
Upload**, and press **Analyze Report**. Expected grounded findings include an
above-range glucose result sourced to page 1 and an in-range sodium result
sourced to page 2. Measurements are grouped by assessment, while source-backed
narratives and explicitly labeled controls appear in separate sections. Review
messages with the same reason are grouped with their affected page references.
This explicit action sends the synthetic report text to the configured Gemini
model and consumes API quota. Do not use real patient data.

After analysis succeeds, press **Generate Summary** once. The expected summary
has a brief overview, the above-range glucose item with its validated value and
source, no fabricated abnormality for sodium, and expandable finding-ID/page
evidence. This is a second explicit Gemini request and consumes additional
quota. Summary generation must not rerun analysis; if it fails, the validated
analysis remains displayed. The generated two-page fixture has no repeated or
narrative findings, so those summary sections may correctly be empty. Use only
synthetic reports for longer manual tests.

For a manual recommendation test, first index a clearly labeled synthetic
guideline whose text actually addresses a finding in the report:

```powershell
uv run python scripts/generate_synthetic_guidelines.py
```

Index
`generated/synthetic_guidelines/synthetic_capstone_review_guidance.pdf`. Every
page is labeled **SYNTHETIC CAPSTONE REVIEW GUIDANCE - NOT MEDICAL ADVICE.** and
contains one fictional documentation-review rule for glucose fasting status,
missing laboratory ranges, mismatched units, medication uncertainty, incomplete
allergy verification, or missing follow-up. The PDF has selectable text and is
not authoritative medical guidance.

Return to the report section, press **Generate Recommendations**, and confirm
that every draft suggestion expands to both patient evidence and a guideline
document/page/chunk with its document hash. Matching report findings are needed;
the agent must abstain when retrieval is absent, unrelated, or insufficient.
The draft must remain **Pending review** until you press **Approve exact draft**
or **Reject exact draft**. Add optional notes, then edit a suggestion and save it
as a new version; the edited version must return to pending review. Approval is
bound only to the exact immutable draft. Each new recommendation generation can
consume Gemini quota.

As an abstention check, index only
`generated/synthetic_guidelines/synthetic_hydration_guideline.pdf` and use the
synthetic lab report. River-7 hydration material does not support laboratory
management, so the app should display an unrelated-guidance abstention instead
of a recommendation. Retrieval itself runs locally and uses no Gemini quota.

Upload the same bytes again in a new browser session to verify that the app
labels compatible analysis, summary, and recommendation draft records as reused
with their creation times and makes no generation request. Use **Reanalyze
Report**, **Regenerate Summary**, or **Regenerate Recommendations** only when you
intentionally want a new request. To remove all saved
versions for those exact bytes, select the confirmation checkbox and press
**Delete saved results for this report**.

To retest a longer synthetic report, keep it synthetic, start the app with the
same command, upload the PDF, and press **Analyze Report** once. A successful
result must show all findings and their page evidence. If the app reports an
incomplete response, do not use any partial output: increase
`REPORT_ANALYSIS_MAX_OUTPUT_TOKENS` within the configured model's documented
limit, restart Streamlit so configuration reloads, and try one explicit analysis
again. Each attempt consumes quota.

For a manual retrieval test, generate the ignored synthetic examples:

```powershell
uv run python scripts/generate_synthetic_guidelines.py
```

Then open the app, upload a PDF from `generated/synthetic_guidelines/` in the
Guideline Knowledge Base section, press **Index guideline**, and query a phrase
such as `River-7 reconciliation`, `BLUE CUP CHECK`, or `missing Moon-4 diary
times`. Confirm that the result shows the expected document and page reference.
Each example contains multiple fictional passages so chunking and topic-specific
retrieval can be inspected; none of its content is medical guidance. The first
indexing run downloads the embedding model.

## Run the checks

```powershell
uv run pytest
uv run ruff check .
uv lock --check
uv sync --check
```

The tests generate PDFs in memory with synthetic phrases. They do not require
real clinical data, network access, API keys, Gemini, or any paid service.
The normal suite mocks the embedding boundary and does not download weights.

An optional integration test uses real local embeddings and temporary Chroma
storage. It may download the model on its first run:

```powershell
$env:RUN_GUIDELINE_INTEGRATION='1'
uv run pytest tests/test_guideline_integration.py -v
Remove-Item Env:RUN_GUIDELINE_INTEGRATION
```

## Project layout

```text
src/clinical_report_assistant/pdf_processing.py  validation and extraction
src/clinical_report_assistant/app.py             Streamlit presentation layer
src/clinical_report_assistant/config.py          server-side configuration
src/clinical_report_assistant/gemini_provider.py Google Gen AI SDK boundary
src/clinical_report_assistant/gemini_smoke.py    explicit synthetic live check
src/clinical_report_assistant/guideline_kb.py    local indexing and retrieval
src/clinical_report_assistant/report_analysis_agent.py one CrewAI analysis agent
src/clinical_report_assistant/summary_agent.py   one CrewAI summary agent
src/clinical_report_assistant/recommendation_agent.py final CrewAI agent
src/clinical_report_assistant/session_state.py   stale result clearing
src/clinical_report_assistant/persistence.py     validated local SQLite results
scripts/generate_synthetic_report.py             synthetic analysis fixture
scripts/generate_synthetic_guidelines.py         synthetic manual-test PDFs
tests/                                           synthetic fixtures and tests
.env.example                                     placeholder-only environment file
```

`pdf_processing.py` is reusable and contains no Streamlit or Gemini code. The
UI reads an upload into bytes and passes those bytes to the processing module.
The Gemini provider remains separate from PDF processing. Uploading a report
does not call Gemini; only the explicit analysis, summary, and recommendation
buttons can do so. Guideline retrieval stays separate and runs locally. These
layers do not log document text, prompts, retrieved passage text, credentials,
or raw provider responses.

## Analysis limitations

- Agent extraction and source matching can still omit or misinterpret content.
- An 8,192-token output budget is bounded but cannot guarantee that every
  unusually dense report will fit. Incomplete or token-limited output is rejected
  rather than repaired or accepted partially.
- Source matching proves textual provenance, not clinical correctness.
- Deterministic abnormality checks support only plain numeric values and
  explicit two-sided numeric ranges. Inequalities, qualitative ranges,
  age/sex-dependent ranges, complex tables, and conversions require review.
- Units are never converted. Missing, incomplete, or mismatched units prevent
  deterministic classification.
- Analysis and summary do not use guideline retrieval. Recommendation consumes
  their already validated saved results and cannot rerun either agent.
- Summary selection is model-generated and may fail validation or omit nuance;
  accepted display text is reconstructed from validated analysis records, but
  that validation does not establish clinical correctness.
- Each explicit live agent generation is a separate Gemini API operation that
  consumes the configured project's quota. This repository does not enable paid
  billing.

## Recommendation limitations

- Retrieval similarity is only a search mechanism, not clinical confidence or
  proof that a passage applies.
- Exact source validation proves provenance, not that a suggestion is clinically
  correct, complete, or safe.
- Conservative lexical screening handles clearly unrelated material but cannot
  establish medical relevance, resolve contradictory guidelines, or evaluate
  every applicability condition.
- Human approval records review of one exact draft version. It is not medical
  authorization, clinical certification, or a replacement for professional
  judgment.
- This capstone uses synthetic training guidelines only. It must not be used for
  patient care or presented as authoritative medicine.

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

## Retrieval limitations

- Only selectable text is indexed; scanned guidelines require future OCR.
- Chunking is character-based within page boundaries. Tables and complex PDF
  layouts may extract or split imperfectly.
- Retrieval returns semantically similar passages, not recommendations,
  validated clinical answers, or confidence percentages.
- Guideline Chroma storage is intended for a single-machine capstone. Index
  migration and deployment are not implemented.
- Changing the embedding model or chunk settings requires a fresh index.
- Synthetic examples are test material and must not be treated as
  authoritative medicine.

## Secrets and local data

Use `.env.example` only as a placeholder template and keep real local settings
in `.env`. `.env`, Streamlit secrets, uploaded documents, generated examples,
databases, Chroma storage, and tool caches are ignored by Git. Never put real
patient data or a real API key in this capstone repository. The ignored SQLite
results database contains saved findings, summaries, warnings, recommendation
drafts, review notes, guideline passages, and evidence excerpts; it is local
storage and does not provide shared-user isolation.
