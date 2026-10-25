# Lector RAG API

This is the independently runnable Python service foundation for Lector. It
provides versioned request/response schemas, a health endpoint, deterministic
chunking, BM25 and semantic retrieval, hybrid retrieval, an evidence-backed
story graph, context assembly, and answer generation. Canonical book content,
graph snapshots, and available semantic vectors are stored locally in SQLite.
Embedding configuration is optional. BookNLP is an optional local graph
processor; its Python package and model assets are not installed or downloaded
by default.

## Requirements and setup

Use Python 3.12 (the supported development runtime). From this directory, create and activate a virtual
environment, then install the small API and test dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

On macOS/Linux, use `python3.12 -m venv .venv` and
`source .venv/bin/activate`.
The optional test extra supplies HTTPX, which FastAPI's `TestClient` uses.

## Run

```shell
python -m uvicorn lector_rag_api.main:create_app --factory --reload --port 7130
```

The health endpoint is `GET http://127.0.0.1:7130/api/v1/health`. FastAPI's
interactive schema page is at `/docs`.

For the Flow reader running at port 7127, browser access is restricted to
`http://localhost:7127` and `http://127.0.0.1:7127` by default. Configure
`LECTOR_CORS_ORIGINS` as a comma-separated list when using another local
origin. Set `NEXT_PUBLIC_LECTOR_API_URL` in the reader environment to the
backend base URL (default `http://127.0.0.1:7130`). Do not put provider keys
in the frontend environment.

## Test

```shell
python -m unittest discover -s tests -v
```

## API contract

- `GET /api/v1/health` reports HTTP service readiness only; it does not imply
  that retrieval, models, or answer generation are ready.
- `PUT /api/v1/books/{book_id}/epub` streams the original EPUB to the local
  backend library. The backend validates its SHA-256 content version and ZIP
  container and assigns its own generated disk filename. `GET /api/v1/books`
  lists saved books; `GET /api/v1/books/{book_id}/epub` restores the file.
- `POST /api/v1/books/{book_id}/source-units` accepts a content version and
  ordered source units from Flow's already-loaded EPUB section documents. It
  validates the handoff, chunks units, and replaces that book's local
  chunks. Stable source IDs, kind, text, section/unit order, and optional href
  and CFI are preserved. Its `202` response reports the book/version, accepted
  source-unit count, and produced chunk count. Source text is sent only to the
  local backend; selected eligible excerpts are sent to Groq only for final
  answer generation when configured.
- `POST /api/v1/books/{book_id}/query` accepts a question, retrieval mode,
  optional content version, and a required explicitly confirmed inclusive
  reading boundary. Current viewport position alone is not a confirmation.
- `POST /api/v1/query` searches all indexed books only when `all_books: true`
  is explicit and a confirmed boundary is provided for every indexed book.
- Query responses contain ranked evidence chunks with provenance, score,
  retrieval method, and matched keyword terms. Hybrid results also include
  each component's rank and score. Graph mode retains hybrid results and can
  append bounded graph-supported chunks with fact and evidence metadata.
- Validation and not-implemented responses use `{ "error": { "code", "message", "details?" } }`.

All paths are versioned under `/api/v1`. Ingestion, BM25, semantic retrieval,
and hybrid retrieval are implemented. Semantic, hybrid, and graph queries require a
configured embedding provider when semantic weight is positive; no model is
configured by default, so semantic queries return `503` until one is configured.
Setting semantic weight to zero explicitly disables that component. Data is
persisted locally in SQLite as described below.

## Retrieval behavior

Keyword search NFKC-normalizes and casefolds Unicode text. Unicode words are
tokens; internal straight and typographic apostrophes remain part of a token,
while hyphens, underscores, and other punctuation separate tokens. BM25 uses
`k1=1.5`, `b=0.75`, Robertson/Sparck Jones IDF, and counts duplicate query terms
once. Boundary-eligible documents determine BM25 document frequencies and
average length before scoring.

Semantic search uses cosine similarity. `EmbeddingProvider` declares a fixed
vector dimension and accepts batches for both chunk and query embeddings.
Vectors are checked for dimensionality and finite values. An optional local
provider can be configured as described below; there is no default model or
model download.

Hybrid retrieval uses Reciprocal Rank Fusion, `sum(weight / (k + rank))`, with
one-based component ranks and default `k=60` and equal source weights. The
optional `hybrid_options` request object can set `k`, `keyword_weight`, and
`semantic_weight`; weights must be finite and non-negative, and at least one
must be positive. Component scores are reported for explanation but are not
compared when fusing. A disabled (zero-weight) source contributes no results.

All retrieval methods accept only a confirmed boundary. Chunks containing any
source unit after that boundary are excluded in full before ranking; this is
deliberately conservative when a chunk spans the boundary. For all-books search,
every indexed book must have its own valid boundary. This restricts use of
future source material but does not detect semantic spoilers in eligible text.
The hybrid service obtains both component lists with the same scope and
boundary, then rechecks fused chunks against that boundary. Flow confirmation
behavior is not integrated, so this is not an end-to-end spoiler guarantee.

## Story graph and graph retrieval

The extraction model receives bounded canonical chunk batches and must return
validated entity, event, and relationship candidates with chunk and source-unit
evidence references. The backend validates every reference and derives graph
IDs itself; extractor-local references are only used to connect candidates
within one extraction batch. Normal imports build a complete graph only for
books whose per-book Graph RAG preference is enabled. The legacy
`extract_story_graph(book_id=..., reading_boundary=...)` service method remains
for compatibility, but reader progress confirmation does not invoke it.

Entities retain evidence-backed aliases. Conservative resolution joins only
exact NFKC/casefolded canonical names with the same entity kind when the
extractor marks identity `resolved`; `uncertain` candidates remain separate
using evidence-scoped IDs. Aliases and similar names do not trigger merges.
Facts retain book/version,
section/chunk order, source units, href/CFI, explicit-versus-inferred status,
and extractor name/version. A graph fact is eligible only if all of its
supporting evidence is within the supplied boundary; relationship endpoints
and event participants must also be eligible. Graph snapshots are persisted
locally and replaced when the content version changes. Graph retrieval is available as
`retrieval_mode: "graph"` on query endpoints. It starts with hybrid RRF
candidates, traverses only eligible facts, and returns canonical supporting
chunks. Defaults cap traversal at depth 2, 12 facts, 3 facts per entity, and 8
additional chunks. Hybrid results stay first in RRF order; graph additions
are ordered by shallowest fact depth and source order. Graph distance is not
narrative importance.

Automatic full-book jobs are scheduled after keyword indexing succeeds. Their
independent state is returned by the preparation status endpoint.
`POST /api/v1/books/{book_id}/preparation/graph/retry?content_version_id=...`
retries a failed full-book job. The legacy confirmed-boundary graph routes
remain local-only. Extraction uses 2-chunk batches by default
(`LECTOR_GRAPH_BATCH_SIZE`, allowed range 1–100),
with at most 2 attempts per batch (`LECTOR_GRAPH_BATCH_MAX_ATTEMPTS`). Model
calls happen outside the service lock and SQLite transactions, so reading and
text retrieval remain available. Missing extractor configuration returns a
structured `503`; an invalid batch is logged and retried only within that
bounded attempt count. A failed batch does not erase committed earlier batches.

Schema-valid empty extraction results are allowed and still mark those source
chunks processed. Graph snapshots and processed chunk IDs commit together. A
version-6 additive SQLite migration adds per-book opt-in and independent graph
progress without rebuilding or deleting existing books. Existing evidence-bearing graph snapshots seed
processed markers where their evidence points to current canonical chunks.
Unfinished `processing` claims return to `queued` at backend startup and are
resumed; retries skip chunks committed by earlier batches. New content versions
cascade away old jobs and markers with their version-scoped book records.
Repeated requests for the same book/version return the existing full-book job;
confirmed reading positions affect graph retrieval eligibility, not graph
construction. One process-local worker runs jobs
serially; this local service is designed for a single API process, not multiple
independent Uvicorn workers sharing the same SQLite file.

The frontend reports queued/processing/completed/failed state and can retry a
failed or unavailable job. Progress counts chunks in the full indexed book.
Existing graph facts remain queryable while extraction is running, and keyword
retrieval does not depend on graph completion. No confirmed boundary is inferred
from navigation.
Graph retrieval returns a structured unavailable error when the selected scope
has no graph facts; it does not quietly report hybrid-only results as graph
expansion. In an all-books scope, graph evidence may exist for only some books.
Health reports whether providers are configured, not whether the local runtime
or model is currently reachable.

## Local embeddings and BookNLP

The backend makes no model request at startup. Ollama remains an optional
loopback provider for embeddings only. Configure
`LECTOR_OLLAMA_BASE_URL`, `LECTOR_EMBEDDING_MODEL`, and
`LECTOR_EMBEDDING_DIMENSIONS`. Embedding dimensions are validated on every
response and against persisted vector metadata. Keyword retrieval remains
available when embeddings are missing or misconfigured.

For a Windows computer with 16 GB RAM, an i7 CPU, and Iris Xe graphics,
[`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) is a
reasonable embedding starting point (Ollama currently lists a 274 MB model
artifact and 768 dimensions). Story graph construction uses BookNLP 1.0.8's
English `small` weights with its `entity,quote,coref` pipeline, adapted from
the supplied experiment. Install the optional runtime from `apps/rag-api`
with `python -m pip install -e ".[booknlp]"`; compatible spaCy, PyTorch, and
transformer runtime assets must be available locally. BookNLP model files are
stored by default under the repository's `models/booknlp` directory, resolved
from the backend source location rather than the current working directory.
Override this with the absolute `LECTOR_BOOKNLP_MODEL_DIR` environment
variable. The old `~/booknlp_models` default is no longer used. The existing
`models/ollama` directory is unrelated and is not modified.

After installing the optional runtime, explicitly run this one-time setup
command from `apps/rag-api` (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m lector_rag_api.booknlp_models
```

The setup creates the configured directory, obtains the three official
BookNLP small-model files, validates each staged PyTorch state dictionary,
and atomically installs it. Existing valid files are skipped; interrupted or
invalid staged downloads are removed. The exact required filenames are
`entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model`,
`coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model`, and
`speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model`.
The command prints the destination and installed file sizes. Re-run it to
verify setup; validated assets are reported as already installed and are not
downloaded again. The asset URLs are the Berkeley-hosted URLs used by BookNLP's
upstream loader; upstream currently specifies HTTP and publishes no checksums,
so installation is explicit rather than automatic. Lector never downloads
these assets during startup or ordinary graph preparation. BookNLP is
initialized in its documented `custom` model mode with the exact three paths,
preventing its built-in `small` mode from silently looking in
`~/booknlp_models` or downloading missing files. `LECTOR_BOOKNLP_ENABLED`,
`LECTOR_BOOKNLP_MODEL=small`, and `LECTOR_BOOKNLP_MODEL_DIR` control the
adapter. BookNLP's upstream documents the `.book`, `.entities`, and `.tokens`
artifacts, including token byte offsets, which Lector maps to canonical chunk
IDs. Missing assets produce an actionable graph status, while normal reading,
keyword retrieval, embeddings, and existing indexed graph data remain
available.

BookNLP 1.0.8's entity, coreference, and quotation-attribution components each
load a Hugging Face base model; their three `.model` files above are fine-tuned
weights and do not contain the base tokenizer vocabulary or BERT weights. The
base repositories are `google/bert_uncased_L-4_H-256_A-4`,
`google/bert_uncased_L-2_H-256_A-4`, and
`google/bert_uncased_L-8_H-256_A-4`, derived by BookNLP from the corresponding
custom checkpoint filenames. On Windows, Lector normalizes the custom BookNLP paths to forward
slashes because this BookNLP version derives the Hugging Face model ID by
splitting the entity path on `/`. The Hugging Face Hub cache defaults to
`<repository>/models/huggingface` (on this setup, under
`D:\Projects\Lector\flow-main\models`), not the user's C: drive cache. The
cache uses Hugging Face Hub's standard repository snapshot layout; do not copy
model files into an arbitrary folder.

After installing the optional runtime, explicitly prepare the base tokenizer
and model once from `apps/rag-api`:

```powershell
.\.venv\Scripts\python.exe -m lector_rag_api.booknlp_base_model
```

This prepares only `config.json`, `pytorch_model.bin`, and `vocab.txt` from
those three official repositories. The existing complete L-4 snapshot is
reused; only missing component snapshots are fetched when this explicit command
runs. Each snapshot is checked for non-empty required files. To check all three
cached snapshots without network access, run:

```powershell
.\.venv\Scripts\python.exe -m lector_rag_api.booknlp_base_model --check
```

Normal graph inference checks all three snapshots offline and sets `HF_HOME`
and `HF_HUB_CACHE` to the repository's
D: `models/huggingface` location and sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before importing BookNLP/Transformers. The explicit
setup command temporarily enables Hub access only in its own Python process;
it does not change Windows environment variables or enable downloads during
backend startup.

After setup, retry a failed graph preparation from the reader's graph status
panel or call `POST /api/v1/books/{book_id}/preparation/graph/retry` with the
book's content version. A failed preparation resumes from existing job
progress; do not re-index the book or delete the database to install models.

Book text is sent to Ollama only for configured embeddings and passed to local
BookNLP only for books where Graph RAG was explicitly enabled. BookNLP
constructs a graph from the full book; its profiles and relationship evidence
are filtered server-side by the explicitly confirmed reading boundary before
graph display or graph retrieval. Groq remains reserved for final answer
generation.

## Chunking behavior

The Python port uses a default budget of 1,200 UTF-16 code units and overlap of
150 UTF-16 code units (capped at one quarter of custom budgets). It counts the
two-newline separator, keeps source units whole, isolates EPUB sections,
preserves oversized units in their own chunks, and overlaps only whole trailing
units. UTF-16 counting matches TypeScript `String.length`, including two units
for non-BMP characters such as emoji. CFI strings are copied only when supplied;
unavailable and malformed CFI metadata are reported without generating a
location.

## Local persistence

The service uses Python's standard-library SQLite driver. By default the
database is `data/lector.sqlite3` under this backend directory. Set
`LECTOR_RAG_DB_PATH` to choose a different local path. The `data/` directory is
ignored by Git. The production command above uses the app factory, which opens
the durable database; the module-level `app` object is an in-memory instance
kept for import-time inspection and tests.

SQLite schema migrations are tracked by `PRAGMA user_version`; startup creates
a missing database and rejects a database from a newer unsupported schema.
Startup reconstructs BM25 from canonical chunks, restores the story graph,
and loads semantic vectors only when their provider/model identity and
dimensions match the explicitly configured embedding provider. It does not
download models or call providers during recovery. Embedding providers should
expose a stable `identity`, `provider_id`, or `model_id`; providers that do not
yet expose one use a class-and-dimension fallback for compatibility. Missing, malformed, or
incompatible vectors make semantic retrieval fail with an actionable
re-ingestion message; keyword retrieval remains available. Graph evidence that
does not resolve to current canonical chunks fails startup closed.

Ingestion prepares chunks and retrieval state before opening the replacement
transaction. Canonical chunks, graph snapshot, and available vectors commit
together; in-memory indexes update only after SQLite commits. A failed write
rolls back and leaves the prior in-memory and persisted book intact. Repeating
the same content version with identical content is idempotent; reusing a
content-version ID for changed content is rejected so stale graph facts and
vectors cannot be retained. Use a new content-version ID for changed EPUB
content. Reading boundaries remain explicit request inputs and are never
persisted as confirmed state.

To back up the library, stop the service and copy the SQLite database file (and
any associated `-wal`/`-shm` files if present). To reset deliberately, stop the
service, confirm the configured `LECTOR_RAG_DB_PATH`, and remove that database
file yourself; startup never deletes existing user data.

Uploaded EPUB originals are stored under `data/epubs/` by default; override the
location with `LECTOR_EPUB_STORAGE_DIR`. The additive schema migration stores
the book ID, title, original filename, content version, generated storage key,
file size, and uploaded/indexed status. Flow also retains its browser IndexedDB
copy and can recover a missing local copy from the backend library when the
file remains available. A missing original is reported without deleting the
canonical indexed content.

Answer-generation audit records are one UTF-8 `.txt` file per request under
`data/qa-logs/YYYY-MM-DD/` by default; override with `LECTOR_QA_LOG_DIR`. Logs
contain the question, boundary, selected evidence, Groq HTTP status and raw
response body before parsing, plus post-processing outcome. Credentials and
Authorization headers are excluded. These files contain copyrighted excerpts:
keep them local and private, and manually remove old daily folders according
to your retention needs. The API does not serve or upload the audit logs.

Back up SQLite and the EPUB directory together. Stop the backend, copy both to
their configured locations, and restart to restore. Do not delete the database
to resolve model or content-version errors; follow the actionable error and
re-index only when the configured embedding identity requires it.

## Answer generation and privacy

Indexing, embeddings, keyword retrieval, and graph data remain local. Groq is
used only for final answer generation. No Groq calls occur during ingestion,
embedding, graph extraction, or retrieval.

Set `GROQ_API_KEY` and `GROQ_MODEL` in the backend process environment before
starting the service. For PowerShell:

```powershell
$env:GROQ_API_KEY = "your-key"
$env:GROQ_MODEL = "your-selected-model-id"
python -m uvicorn lector_rag_api.main:create_app --factory --reload --port 7130
```

[`.env.example`](.env.example) contains placeholders only. The API reads
environment variables directly and does not load dotenv files. Never commit
credentials.

`POST /api/v1/answer` accepts a question, either a single `book_id` and an
explicit confirmed boundary, or `all_books: true` with one boundary per
indexed book. It returns `answered`, `insufficient_evidence`, or `abstained`,
validated citations, and source metadata including available CFIs. Context
assembly sends only selected passage text, minimal section/chunk/href
orientation metadata, and eligible graph fact labels with passage IDs. The
full EPUB, index, and graph are not sent. Selected book
excerpts and graph labels are transmitted to Groq for each answer.

The Groq system prompt treats the supplied excerpts as partial, potentially
disconnected novel evidence, asks for direct answers or coherent recap
summaries, and distinguishes explicit facts from interpretations. It preserves
the strict JSON response contract and exact passage-ID citation validation.
JSON-generation HTTP 400s are classified separately; HTTP 429 responses expose
a retry delay when Groq supplies `Retry-After`. Lector does not automatically
repeat answer requests.

Evidence is revalidated after generation; if indexed sources change during a
request, the API fails closed. Unknown model-generated citation IDs result in
abstention. Citation validation confirms that cited passages were supplied;
it does not prove every answer sentence is entailed by those passages. Answer
responses expose this limitation in their `uncertainty` field when the model
does not provide a more specific qualification. Spoiler
protection depends on Flow supplying a reliable confirmed boundary, source
provenance, and model behavior. It does not detect every semantic spoiler in
otherwise eligible passages.

## Per-book BookNLP character graph preparation

On EPUB import, Lector asks whether Graph RAG should be enabled for that book.
Keyword indexing and embeddings remain automatic either way. When enabled,
the backend runs BookNLP once over the reconstructed full-book text, then
persists its character profiles, mention mappings, co-mention relationships,
candidate verbs and sentence evidence in canonical chunk batches. The default
assembly batch size is 2 chunks (`LECTOR_GRAPH_BATCH_SIZE`). BookNLP analysis
itself is whole-book processing, not incremental coreference. It uses the
small English model and does not use Qwen, Ollama, or Groq to create graph
records. BookNLP is optional and must be installed with model assets before an
opted-in graph can complete.

Graph construction covers the full indexed book so it does not need to be
restarted as reading progress changes. The reader's explicitly confirmed
position remains a separate query-time filter: graph facts and their complete
supporting evidence are checked against that boundary at every traversal step.
This cannot prevent every semantic spoiler in otherwise eligible text.

The per-book preference and progress remain in SQLite. An additive schema
migration stores BookNLP profiles, mentions, edges, evidence and resumable
artifacts separately from legacy generated graph snapshots. Old graph records
are not deleted, re-run, or re-labeled. Existing books are not automatically
re-indexed or scheduled for BookNLP. A disabled book does not schedule graph
work. If local BookNLP or its model cache is unavailable, indexing and keyword
retrieval remain usable. Groq (`GROQ_API_KEY`, `GROQ_MODEL`) is only for final
answer generation.

The SQLite-backed graph worker is designed for one backend process sharing the
database. Do not run multiple Uvicorn worker processes against the same local
database; the process-local worker lock does not coordinate across processes.
The worker persists BookNLP output and batch progress, so restarting the
single backend process resumes graph assembly without repeating completed
database batches. A crash before the full-book analysis artifacts have been
persisted can require that analysis to run again.
