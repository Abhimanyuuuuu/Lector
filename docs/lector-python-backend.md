# Lector Python backend architecture

## Language boundary

Flow remains the TypeScript/React EPUB reader. It owns EPUB loading, rendering,
navigation, CFI creation/resolution, reader state, the future confirmed-boundary
interaction, and the future assistant UI. It should send serializable extracted
source units from the already-loaded section documents; Python must not receive
browser DOM objects or reparse a second EPUB copy.

The separate `apps/rag-api` Python service owns production chunking, keyword
and semantic retrieval, hybrid retrieval, and evidence-backed story graph
foundations. It is intended to own reranking, local embeddings, full story
extraction and graph updates, spoiler-aware candidate filtering,
evidence/context assembly, and the final answer request.
The frontend will call a versioned HTTP API and render citations by resolving
the returned source IDs/CFI metadata in Flow.

## API handoff

The API is under `/api/v1`. Ingestion identifies a `book_id` in the path, a
`content_version_id`, and ordered source units with stable `source_unit_id`,
`section_index`, `unit_index`, `kind`, text, and optional section href, CFI, and
heading. A book query uses `/api/v1/books/{book_id}/query`; it selects keyword,
semantic, or hybrid mode, optional content version, and a required
`confirmed: true` inclusive boundary pinned to source ID and section/unit
order. All-books query uses `/api/v1/query`, requires explicit
`all_books: true`, and provides a separate confirmed boundary for every
indexed book. Retrieval responses contain ranked chunks, method-specific
scores, and source provenance; they are not generated answers. Errors share a
structured error envelope.

The implementation provides schema validation, health reporting, chunking,
local BM25 retrieval, provider-injected semantic retrieval, and weighted
Reciprocal Rank Fusion. Hybrid requests accept optional `hybrid_options` with
`k` (default 60), `keyword_weight`, and `semantic_weight` (both default 1).
Hybrid results report component ranks and raw scores alongside the fusion
score. Semantic and hybrid API queries return `503` unless an embedding
provider was injected. Query responses are retrieval evidence only. Health
means the HTTP app is up, not that retrieval or models are ready. The API
version allows contract evolution without exposing Python implementation
classes to TypeScript.

Passing extracted source units is preferred over uploading the EPUB: it reuses
Flow's actual DOM extraction and CFI facilities, avoids a second parser and
keeps the navigation source of truth in the reader. A later implementation
should bound request sizes and provide a clear indexing/replacement protocol.

## Spoiler-safety invariant

The viewport/current CFI and confirmed reading boundary are distinct. A jump
must not advance the confirmed boundary automatically. The boundary should
identify an ordered source position; it is an explicit eligibility limit, not
proof of a person's complete reading history and not semantic spoiler
detection.

Future chunk and graph candidate sets must be filtered by book, content version,
and confirmed source position before keyword/vector ranking, reranking, or
context assembly. Context assembly must recheck every selected source before
answer generation and citation emission. Graph facts and edges retain all
supporting source provenance; a fact is eligible only when its required support
is eligible. Explicitly stated facts must remain distinguishable from inferred
relationships. This prevents future-source leakage through retrieval, but
cannot detect every twist or inference already present in eligible text.
Future-content access, if offered, requires a deliberate warned opt-in.

The confirmed-boundary field is supplied by the caller and ingestion does not
mark source units as read. Query validates that the boundary source ID and
section/unit coordinates exist in the indexed version. A chunk is eligible
only when every contained source unit is at or before that boundary; a chunk
crossing it is excluded before BM25 corpus statistics or semantic ranking. An
all-books query requires a separate confirmed boundary for every indexed book.
This prevents future-source candidates from being ranked, but is not semantic
spoiler detection and does not yet guard an answer-generation path.

## Privacy and model policy

The target is local-first: local chunking, indexes, embeddings, graph storage,
and story extraction. No model runner, embedding ID, graph library, or
persistence technology is selected yet. Keep model integrations injectable and
configuration explicit; do not download models as part of setup.

Groq is the only planned hosted service, used only for final answer generation.
It will require a configured model and environment-based API key, must fail
clearly when configuration is absent, and must not be called during indexing,
embedding, or graph extraction. A query may transmit the user's question and
the minimum selected spoiler-eligible source excerpts and graph facts to Groq.
The entire EPUB and full graph must not be sent by default. Final-answer calls
use Groq only when explicit environment configuration is supplied; indexing,
embedding, graph extraction, and retrieval remain local.

The graph foundation is a typed model with an injectable
`StoryExtractor` protocol and validated candidate schemas. Candidates must cite
existing canonical chunks and source units; Python derives persisted IDs rather
than trusting extractor IDs. Resolution joins only exact normalized canonical
names of the same kind only when the extractor marks identity as resolved.
Uncertain candidates remain separate using evidence-scoped IDs. Aliases retain
their own evidence and do not cause automatic merging. Facts preserve extraction version, assertion status, source
positions, hrefs, and CFIs. Graph evidence eligibility requires every supporting
passage to be within the confirmed boundary, and relationships/events also
require eligible endpoint entities. Re-ingesting a new content version clears
the old graph; same-version extraction merges evidence idempotently. Graph
snapshots are persisted alongside canonical chunks in the local SQLite store.
No model or graph database is selected.

## Migration map and roadmap

The existing TypeScript RAG modules are prototypes and regression
specifications, not a second production retrieval engine. Phase 1's
`extract.ts`/`paragraph.ts` remain in the reader for DOM extraction, stable
source IDs, and CFI/source mapping. TypeScript `chunk.ts`, `index.ts`,
`search.ts`, the semantic module, and the existing hybrid prototype remain
reference implementations until Python equivalents have equivalent tests; do
not extend them as production functionality. `types.ts` documents useful
cross-language provenance contracts.

| Migration step                     | Python destination (planned)                    | Scope                                                               |
| ---------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| 1. Scaffold and API contract       | `apps/rag-api/lector_rag_api/{main,schemas}.py` | Implemented: health, validation, ingestion and retrieval contracts  |
| 2. Source/chunk/boundary contracts | `schemas.py` and `chunking.py`                  | Implemented: source metadata, provenance and boundary schema        |
| 3. Chunking                        | `chunking.py`                                   | Implemented: deterministic sections, overlap and UTF-16 sizing      |
| 4. Keyword search                  | `keyword.py`                                    | Implemented: BM25, explicit scope, boundary filtering               |
| 5. Semantic retrieval              | `semantic.py`                                   | Implemented: injected provider interface and cosine search          |
| 6. Hybrid retrieval                | `hybrid.py`                                     | Implemented: weighted RRF over eligible component candidates        |
| 7. Story graph foundation          | `story_graph.py`, `service.py`                  | Implemented: validated extractor boundary and evidence-backed graph |
| 8. Graph retrieval                 | `retrieval/graph.py`                            | Evidence-backed, boundary-eligible graph traversal                  |
| 9. Context and answer              | `answer/`                                       | Eligibility recheck, evidence assembly, Groq-only final answer      |
| 10. Local persistence               | `persistence.py` and `service.py`               | Implemented: SQLite chunks, graph, available vectors and recovery     |
| 11. Flow integration               | reader API client and assistant UI              | Ingestion, explicit boundary, citations and CFI navigation          |

Each phase should define focused unit tests before adding the next subsystem.
Keyword indexes are rebuilt from canonical chunks at startup. Semantic vectors
are restored only when provider/model identity and dimensions match; graph
evidence must resolve to current canonical chunks. Confirmed reading boundaries
remain explicit request data and are never persisted.

## Decisions still open

- Local inference runtime and quantized embedding/story-extraction models.
- Graph library versus a small application-owned graph representation.
- Whether a later migration should replace SQLite JSON graph/vector payloads
  with normalized tables or a graph-specific store.
- Groq model ID, timeout/retry policy, and user-facing disclosure/consent UX.
- Exact boundary UX and how non-linear reading progress is confirmed.
