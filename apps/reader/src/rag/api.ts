import type { BookParagraph } from './types'

declare const process: { env: { NEXT_PUBLIC_LECTOR_API_URL?: string } }

export type RetrievalMode =
  | 'keyword'
  | 'semantic'
  | 'hybrid'
  | 'hybrid_graph'
  | 'graph'

export interface ConfirmedBoundary {
  confirmed: true
  contentVersionId: string
  sectionIndex: number
  unitIndex: number
  sourceUnitId: string
  cfi?: string
}

export interface AnswerCitation {
  passage_id: string
  book_id: string
  content_version_id: string
  section_index: number
  chunk_index: number
  source_unit_ids: string[]
  source_positions: Array<{
    source_unit_id: string
    unit_index: number
    kind: string
    href?: string | null
    cfi?: string | null
  }>
  href?: string | null
  source_cfi?: string | null
  start_cfi?: string | null
  end_cfi?: string | null
  cfi_status: 'complete' | 'partial' | 'unavailable' | 'unreliable'
  excerpt: string
}

export interface AnswerSource {
  passage_id: string
  score: number
  retrieval_method: RetrievalMode
  keyword_rank?: number | null
  keyword_score?: number | null
  semantic_rank?: number | null
  semantic_score?: number | null
  graph_score?: number | null
  graph_depth?: number | null
}

export interface GroundedAnswer {
  status: 'answered' | 'insufficient_evidence' | 'abstained'
  answer: string
  uncertainty?: string | null
  retrieval_mode: RetrievalMode
  citations: AnswerCitation[]
  sources: AnswerSource[]
  graph_fact_ids: string[]
  selected_passage_ids: string[]
}

export interface GraphJob {
  job_id: string
  book_id: string
  content_version_id: string
  boundary_section: number | null
  boundary_unit: number | null
  boundary_source_unit_id: string | null
  total_chunks: number
  processed_chunks: number
  status:
    | 'queued'
    | 'processing'
    | 'persisting'
    | 'completed'
    | 'failed'
    | 'unavailable'
    | 'cancelled'
  error: string | null
  provider: string
  provider_version: string
  job_scope?: 'confirmed_boundary' | 'full_book'
}

export interface CharacterGraphNode {
  id: string
  name: string
  mention_count: number
  aliases: string[]
  pronouns: string[]
  profile_summary?: Record<string, unknown>
}

export interface CharacterGraphEvidence {
  evidence_id: string
  source_chunk_id: string
  section_index: number
  chunk_index: number
  source_unit_ids: string[]
  source_unit_id?: string | null
  unit_index?: number | null
  start_character?: number | null
  end_character?: number | null
  source_cfi?: string | null
  sentence_id: string
  paragraph_id?: string | null
  sentence: string
  character_mentions: string[]
  verbs: string[]
  href?: string | null
  start_cfi?: string | null
  end_cfi?: string | null
  cfi_status: 'complete' | 'partial' | 'unavailable' | 'unreliable'
  source_positions: Array<{
    source_unit_id: string
    unit_index: number
    href?: string | null
    cfi?: string | null
  }>
}

export interface CharacterGraphEdge {
  id: string
  source: string
  target: string
  candidate_category: string
  candidate_verbs: string[]
  weight: number
  evidence_count: number
  evidence_truncated: boolean
  evidence: CharacterGraphEvidence[]
}

export interface CharacterGraphView {
  book_id: string
  content_version_id: string
  view_mode: 'full_book' | 'till_now'
  boundary: {
    section_index: number
    unit_index: number
    source_unit_id: string
  } | null
  nodes: CharacterGraphNode[]
  edges: CharacterGraphEdge[]
}

/**
 * Graph data is keyed by the version already saved with the indexed reader
 * record. Use that identity directly when present so viewing a persisted graph
 * does not require reopening or fingerprinting the original local EPUB.
 */
export async function resolveGraphContentVersionId(
  storedContentVersionId: string | undefined,
  fingerprintLocalEpub: () => Promise<string>,
): Promise<string> {
  if (storedContentVersionId?.trim()) return storedContentVersionId
  return fingerprintLocalEpub()
}

export interface SavedBook {
  book_id: string
  title: string
  original_filename: string
  content_version_id: string
  file_size: number
  indexing_status: 'uploaded' | 'indexed' | 'stale'
  preparation_indexing_status:
    | 'pending'
    | 'queued'
    | 'processing'
    | 'complete'
    | 'failed'
  indexing_progress: number
  indexing_completed: number
  indexing_total: number
  indexing_error?: string | null
  embedding_status:
    | 'pending'
    | 'processing'
    | 'complete'
    | 'failed'
    | 'unavailable'
    | 'not_applicable'
  embedding_progress: number
  embedding_completed: number
  embedding_total: number
  embedding_error?: string | null
  graph_rag_enabled: boolean
  graph_status:
    | 'not_enabled'
    | 'pending'
    | 'queued'
    | 'processing'
    | 'complete'
    | 'failed'
    | 'unavailable'
  graph_progress: number
  graph_completed: number
  graph_total: number
  graph_error?: string | null
  file_available: boolean
  updated_at: string
}

export interface BookPreparation {
  book_id: string
  content_version_id: string
  indexing_status: 'queued' | 'processing' | 'complete' | 'failed'
  embedding_status: SavedBook['embedding_status']
  indexing_completed: number
  indexing_total: number
  embedding_completed: number
  embedding_total: number
  indexing_error: string | null
  embedding_error: string | null
  graph_rag_enabled: boolean
  graph_status: SavedBook['graph_status']
  graph_completed: number
  graph_total: number
  graph_error: string | null
  updated_at: string
}

export class RagApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status?: number,
  ) {
    super(message)
    this.name = 'RagApiError'
  }
}

const baseUrl = () =>
  (process.env.NEXT_PUBLIC_LECTOR_API_URL || 'http://127.0.0.1:7130').replace(
    /\/$/,
    '',
  )

const isRecord = (value: unknown): value is Record<string, any> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

export async function getSavedBookLibrary(
  fetchImpl: typeof fetch = fetch,
): Promise<SavedBook[]> {
  let response: Response
  try {
    response = await fetchImpl(`${baseUrl()}/api/v1/books`)
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  const data: unknown = await response.json().catch(() => undefined)
  if (!response.ok) throw apiErrorFromResponse(response, data)
  if (
    !isRecord(data) ||
    !Array.isArray(data.books) ||
    !data.books.every(isSavedBook)
  )
    throw new RagApiError(
      'The saved-book response did not match its contract.',
      'invalid_response',
      response.status,
    )
  return data.books
}

export async function downloadSavedEpub(
  bookId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<Blob> {
  let response: Response
  try {
    response = await fetchImpl(
      `${baseUrl()}/api/v1/books/${encodeURIComponent(bookId)}/epub`,
    )
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => undefined)
    throw apiErrorFromResponse(response, data)
  }
  return response.blob()
}

export async function deleteSavedBook(
  bookId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  let response: Response
  try {
    response = await fetchImpl(
      `${baseUrl()}/api/v1/books/${encodeURIComponent(bookId)}`,
      { method: 'DELETE' },
    )
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  const data: unknown = await response.json().catch(() => undefined)
  if (!response.ok) throw apiErrorFromResponse(response, data)
  if (!isRecord(data) || data.book_id !== bookId || data.deleted !== true)
    throw new RagApiError(
      'The delete response did not match its contract.',
      'invalid_response',
      response.status,
    )
}

export async function saveEpubToBackend(
  input: {
    bookId: string
    contentVersionId: string
    title: string
    originalFilename: string
    file: File
  },
  fetchImpl: typeof fetch = fetch,
): Promise<void> {
  const query = new URLSearchParams({
    content_version_id: input.contentVersionId,
    title: input.title,
    original_filename: input.originalFilename,
  })
  let response: Response
  try {
    response = await fetchImpl(
      `${baseUrl()}/api/v1/books/${encodeURIComponent(
        input.bookId,
      )}/epub?${query}`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/epub+zip' },
        body: input.file,
      },
    )
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  const data: unknown = await response.json().catch(() => undefined)
  if (!response.ok) throw apiErrorFromResponse(response, data)
  if (
    !isRecord(data) ||
    data.book_id !== input.bookId ||
    data.content_version_id !== input.contentVersionId
  )
    throw new RagApiError(
      'The saved EPUB response did not match its contract.',
      'invalid_response',
      response.status,
    )
}

function isSavedBook(value: unknown): value is SavedBook {
  return (
    isRecord(value) &&
    typeof value.book_id === 'string' &&
    typeof value.title === 'string' &&
    typeof value.original_filename === 'string' &&
    typeof value.content_version_id === 'string' &&
    Number.isInteger(value.file_size) &&
    ['uploaded', 'indexed', 'stale'].includes(value.indexing_status) &&
    ['pending', 'queued', 'processing', 'complete', 'failed'].includes(
      value.preparation_indexing_status,
    ) &&
    typeof value.indexing_progress === 'number' &&
    Number.isInteger(value.indexing_completed) &&
    Number.isInteger(value.indexing_total) &&
    [
      'pending',
      'processing',
      'complete',
      'failed',
      'unavailable',
      'not_applicable',
    ].includes(value.embedding_status) &&
    typeof value.embedding_progress === 'number' &&
    Number.isInteger(value.embedding_completed) &&
    Number.isInteger(value.embedding_total) &&
    typeof value.graph_rag_enabled === 'boolean' &&
    [
      'not_enabled',
      'pending',
      'queued',
      'processing',
      'complete',
      'failed',
      'unavailable',
    ].includes(value.graph_status) &&
    typeof value.graph_progress === 'number' &&
    Number.isInteger(value.graph_completed) &&
    Number.isInteger(value.graph_total) &&
    typeof value.file_available === 'boolean' &&
    typeof value.updated_at === 'string'
  )
}

const isBookPreparation = (value: unknown): value is BookPreparation =>
  isRecord(value) &&
  typeof value.book_id === 'string' &&
  typeof value.content_version_id === 'string' &&
  ['queued', 'processing', 'complete', 'failed'].includes(
    value.indexing_status,
  ) &&
  [
    'pending',
    'processing',
    'complete',
    'failed',
    'unavailable',
    'not_applicable',
  ].includes(value.embedding_status) &&
  Number.isInteger(value.indexing_completed) &&
  Number.isInteger(value.indexing_total) &&
  Number.isInteger(value.embedding_completed) &&
  Number.isInteger(value.embedding_total) &&
  (typeof value.indexing_error === 'string' || value.indexing_error === null) &&
  (typeof value.embedding_error === 'string' ||
    value.embedding_error === null) &&
  typeof value.graph_rag_enabled === 'boolean' &&
  [
    'not_enabled',
    'pending',
    'queued',
    'processing',
    'complete',
    'failed',
    'unavailable',
  ].includes(value.graph_status) &&
  Number.isInteger(value.graph_completed) &&
  Number.isInteger(value.graph_total) &&
  (typeof value.graph_error === 'string' || value.graph_error === null) &&
  typeof value.updated_at === 'string'

export async function getBookPreparation(
  bookId: string,
  contentVersionId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<BookPreparation> {
  let response: Response
  try {
    response = await fetchImpl(
      `${baseUrl()}/api/v1/books/${encodeURIComponent(
        bookId,
      )}/preparation?content_version_id=${encodeURIComponent(
        contentVersionId,
      )}`,
    )
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  const data: unknown = await response.json().catch(() => undefined)
  if (!response.ok) throw apiErrorFromResponse(response, data)
  if (!isBookPreparation(data))
    throw new RagApiError(
      'The preparation response did not match its contract.',
      'invalid_response',
      response.status,
    )
  return data
}

export async function prepareBookSourceUnits(
  bookId: string,
  contentVersionId: string,
  paragraphs: readonly BookParagraph[],
  fetchImpl: typeof fetch = fetch,
  graphRagEnabled?: boolean,
): Promise<BookPreparation> {
  return postJson(
    `/books/${encodeURIComponent(bookId)}/prepare`,
    {
      content_version_id: contentVersionId,
      ...(graphRagEnabled === undefined
        ? {}
        : { graph_rag_enabled: graphRagEnabled }),
      source_units: paragraphs.map((paragraph) => ({
        source_unit_id: paragraph.id,
        section_index: paragraph.sectionIndex,
        unit_index: paragraph.paragraphIndex,
        kind: paragraph.kind,
        text: paragraph.text,
        href: paragraph.href,
        cfi: paragraph.cfi,
        heading: paragraph.heading,
      })),
    },
    isBookPreparation,
    fetchImpl,
  )
}

function apiErrorFromResponse(response: Response, data: unknown): RagApiError {
  const error = isRecord(data) && isRecord(data.error) ? data.error : undefined
  return new RagApiError(
    typeof error?.message === 'string'
      ? error.message
      : `RAG request failed (${response.status}).`,
    typeof error?.code === 'string' ? error.code : 'request_failed',
    response.status,
  )
}

async function postJson<T>(
  path: string,
  body: unknown,
  validate: (value: unknown) => value is T,
  fetchImpl: typeof fetch = fetch,
): Promise<T> {
  let response: Response
  try {
    response = await fetchImpl(`${baseUrl()}/api/v1${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable. Start the Python backend and try again.',
      'backend_unavailable',
    )
  }

  let data: unknown
  try {
    data = await response.json()
  } catch {
    throw new RagApiError(
      'The RAG API returned an invalid response.',
      'invalid_response',
      response.status,
    )
  }
  if (!response.ok) {
    const error =
      isRecord(data) && isRecord(data.error) ? data.error : undefined
    throw new RagApiError(
      typeof error?.message === 'string'
        ? error.message
        : `RAG request failed (${response.status}).`,
      typeof error?.code === 'string' ? error.code : 'request_failed',
      response.status,
    )
  }
  if (!validate(data)) {
    throw new RagApiError(
      'The RAG API response did not match its contract.',
      'invalid_response',
      response.status,
    )
  }
  return data
}

const isIndexResponse = (
  value: unknown,
): value is {
  book_id: string
  content_version_id: string
  accepted_source_units: number
  chunk_count: number
} =>
  isRecord(value) &&
  typeof value.book_id === 'string' &&
  typeof value.content_version_id === 'string' &&
  Number.isInteger(value.accepted_source_units) &&
  Number.isInteger(value.chunk_count)

export async function indexSourceUnits(
  bookId: string,
  contentVersionId: string,
  paragraphs: readonly BookParagraph[],
  fetchImpl: typeof fetch = fetch,
): Promise<{ chunkCount: number; sourceUnitCount: number }> {
  const response = await postJson(
    `/books/${encodeURIComponent(bookId)}/source-units`,
    {
      content_version_id: contentVersionId,
      source_units: paragraphs.map((paragraph) => ({
        source_unit_id: paragraph.id,
        section_index: paragraph.sectionIndex,
        unit_index: paragraph.paragraphIndex,
        kind: paragraph.kind,
        text: paragraph.text,
        href: paragraph.href,
        cfi: paragraph.cfi,
        heading: paragraph.heading,
      })),
    },
    isIndexResponse,
    fetchImpl,
  )
  if (
    response.book_id !== bookId ||
    response.content_version_id !== contentVersionId
  ) {
    throw new RagApiError(
      'The backend indexed a different book revision.',
      'version_mismatch',
    )
  }
  return {
    chunkCount: response.chunk_count,
    sourceUnitCount: response.accepted_source_units,
  }
}

export async function extractStoryGraph(
  bookId: string,
  boundary: ConfirmedBoundary,
  fetchImpl: typeof fetch = fetch,
  retry = false,
): Promise<GraphJob> {
  const response = await postJson(
    `/books/${encodeURIComponent(bookId)}/graph/extract${
      retry ? '/retry' : ''
    }`,
    {
      confirmed: true,
      section_index: boundary.sectionIndex,
      unit_index: boundary.unitIndex,
      source_unit_id: boundary.sourceUnitId,
      cfi: boundary.cfi,
    },
    isGraphJob,
    fetchImpl,
  )
  if (
    response.book_id !== bookId ||
    response.content_version_id !== boundary.contentVersionId ||
    response.boundary_section !== boundary.sectionIndex ||
    response.boundary_unit !== boundary.unitIndex ||
    response.boundary_source_unit_id !== boundary.sourceUnitId
  ) {
    throw new RagApiError(
      'The graph job did not match the confirmed book boundary.',
      'version_mismatch',
    )
  }
  return {
    ...response,
    provider: response.provider ?? 'booknlp',
    provider_version: response.provider_version ?? '',
  }
}

export async function retryFullBookGraph(
  bookId: string,
  contentVersionId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<GraphJob> {
  const response = await postJson(
    `/books/${encodeURIComponent(
      bookId,
    )}/preparation/graph/retry?content_version_id=${encodeURIComponent(
      contentVersionId,
    )}`,
    {},
    isGraphJob,
    fetchImpl,
  )
  if (
    response.book_id !== bookId ||
    response.content_version_id !== contentVersionId ||
    response.job_scope !== 'full_book'
  )
    throw new RagApiError(
      'The graph retry did not match the selected book revision.',
      'version_mismatch',
    )
  return response
}

export async function getStoryGraphStatus(
  bookId: string,
  contentVersionId: string,
  fetchImpl: typeof fetch = fetch,
): Promise<GraphJob> {
  let response: Response
  try {
    response = await fetchImpl(
      `${baseUrl()}/api/v1/books/${encodeURIComponent(
        bookId,
      )}/graph/status?content_version_id=${encodeURIComponent(
        contentVersionId,
      )}`,
    )
  } catch {
    throw new RagApiError(
      'The local Lector RAG API is unavailable.',
      'backend_unavailable',
    )
  }
  const data: unknown = await response.json().catch(() => undefined)
  if (!response.ok) {
    const error =
      isRecord(data) && isRecord(data.error) ? data.error : undefined
    throw new RagApiError(
      typeof error?.message === 'string'
        ? error.message
        : `RAG request failed (${response.status}).`,
      typeof error?.code === 'string' ? error.code : 'request_failed',
      response.status,
    )
  }
  if (
    !isGraphJob(data) ||
    data.book_id !== bookId ||
    data.content_version_id !== contentVersionId
  )
    throw new RagApiError(
      'The graph status response did not match its contract.',
      'invalid_response',
      response.status,
    )
  return {
    ...data,
    provider: data.provider ?? 'booknlp',
    provider_version: data.provider_version ?? '',
  }
}

export async function getCharacterGraphView(
  bookId: string,
  contentVersionId: string,
  boundary?: ConfirmedBoundary,
  fetchImpl: typeof fetch = fetch,
  viewMode: 'full_book' | 'till_now' = 'till_now',
): Promise<CharacterGraphView> {
  if (
    viewMode === 'till_now' &&
    (!boundary?.confirmed || boundary.contentVersionId !== contentVersionId)
  )
    throw new RagApiError(
      'Confirm reading progress for this book revision before opening its graph.',
      'invalid_boundary',
    )
  const response = await postJson(
    `/books/${encodeURIComponent(bookId)}/graph/view`,
    {
      content_version_id: contentVersionId,
      view_mode: viewMode,
      ...(viewMode === 'till_now' && boundary
        ? {
            section_index: boundary.sectionIndex,
            unit_index: boundary.unitIndex,
            source_unit_id: boundary.sourceUnitId,
            cfi: boundary.cfi,
            confirmed: true,
          }
        : {}),
    },
    isCharacterGraphView,
    fetchImpl,
  )
  if (
    response.book_id !== bookId ||
    response.content_version_id !== contentVersionId ||
    response.view_mode !== viewMode ||
    (viewMode === 'till_now' &&
      (!boundary ||
        !response.boundary ||
        response.boundary.section_index !== boundary.sectionIndex ||
        response.boundary.unit_index !== boundary.unitIndex ||
        response.boundary.source_unit_id !== boundary.sourceUnitId)) ||
    (viewMode === 'full_book' && response.boundary !== null)
  )
    throw new RagApiError(
      'The graph response did not match the confirmed book boundary.',
      'invalid_response',
    )
  return response
}

function isCharacterGraphView(value: unknown): value is CharacterGraphView {
  if (
    !isRecord(value) ||
    typeof value.book_id !== 'string' ||
    typeof value.content_version_id !== 'string' ||
    !['full_book', 'till_now'].includes(String(value.view_mode)) ||
    !(value.boundary === null || isRecord(value.boundary)) ||
    !Array.isArray(value.nodes) ||
    !Array.isArray(value.edges)
  )
    return false
  return (
    (value.view_mode === 'full_book'
      ? value.boundary === null
      : isRecord(value.boundary) &&
        Number.isInteger(value.boundary.section_index) &&
        Number.isInteger(value.boundary.unit_index) &&
        typeof value.boundary.source_unit_id === 'string') &&
    value.nodes.every(
      (node: unknown) =>
        isRecord(node) &&
        typeof node.id === 'string' &&
        typeof node.name === 'string' &&
        Number.isInteger(node.mention_count) &&
        Array.isArray(node.aliases) &&
        node.aliases.every((alias: unknown) => typeof alias === 'string') &&
        Array.isArray(node.pronouns) &&
        node.pronouns.every(
          (pronoun: unknown) => typeof pronoun === 'string',
        ) &&
        (node.profile_summary === undefined || isRecord(node.profile_summary)),
    ) &&
    value.edges.every(
      (edge: unknown) =>
        isRecord(edge) &&
        typeof edge.id === 'string' &&
        typeof edge.source === 'string' &&
        typeof edge.target === 'string' &&
        typeof edge.candidate_category === 'string' &&
        Array.isArray(edge.candidate_verbs) &&
        Number.isInteger(edge.evidence_count) &&
        Array.isArray(edge.evidence) &&
        edge.evidence.every(
          (item: unknown) =>
            isRecord(item) &&
            typeof item.source_chunk_id === 'string' &&
            typeof item.sentence === 'string' &&
            Number.isInteger(item.section_index) &&
            Number.isInteger(item.chunk_index),
        ),
    )
  )
}

function isGraphJob(value: unknown): value is GraphJob {
  return (
    isRecord(value) &&
    typeof value.job_id === 'string' &&
    typeof value.book_id === 'string' &&
    typeof value.content_version_id === 'string' &&
    (Number.isInteger(value.boundary_section) ||
      value.boundary_section === null) &&
    (Number.isInteger(value.boundary_unit) || value.boundary_unit === null) &&
    (typeof value.boundary_source_unit_id === 'string' ||
      value.boundary_source_unit_id === null) &&
    Number.isInteger(value.total_chunks) &&
    Number.isInteger(value.processed_chunks) &&
    [
      'queued',
      'processing',
      'persisting',
      'completed',
      'failed',
      'unavailable',
      'cancelled',
    ].includes(value.status) &&
    (value.error === null || typeof value.error === 'string') &&
    (typeof value.provider === 'string' || value.provider === undefined) &&
    (typeof value.provider_version === 'string' ||
      value.provider_version === undefined) &&
    (value.job_scope === undefined ||
      value.job_scope === 'confirmed_boundary' ||
      value.job_scope === 'full_book')
  )
}

export async function askBook(
  input: {
    bookId: string
    contentVersionId: string
    query: string
    retrievalMode: RetrievalMode
    boundary: ConfirmedBoundary
  },
  fetchImpl: typeof fetch = fetch,
): Promise<GroundedAnswer> {
  const response = await postJson(
    '/answer',
    {
      book_id: input.bookId,
      content_version_id: input.contentVersionId,
      query: input.query,
      retrieval_mode: input.retrievalMode,
      reading_boundary: {
        confirmed: true,
        section_index: input.boundary.sectionIndex,
        unit_index: input.boundary.unitIndex,
        source_unit_id: input.boundary.sourceUnitId,
        cfi: input.boundary.cfi,
      },
    },
    isAnswerResponse,
    fetchImpl,
  )
  if (
    response.retrieval_mode !== input.retrievalMode ||
    response.citations.some(
      (citation) =>
        !response.selected_passage_ids.includes(citation.passage_id),
    ) ||
    response.citations.some(
      (citation) =>
        citation.book_id !== input.bookId ||
        citation.content_version_id !== input.contentVersionId,
    )
  ) {
    throw new RagApiError(
      'The answer contained source metadata for another book revision.',
      'invalid_response',
    )
  }
  return response
}

function isAnswerResponse(value: unknown): value is GroundedAnswer {
  if (!isRecord(value)) return false
  const modes: RetrievalMode[] = [
    'keyword',
    'semantic',
    'hybrid',
    'hybrid_graph',
    'graph',
  ]
  if (
    !['answered', 'insufficient_evidence', 'abstained'].includes(
      value.status,
    ) ||
    typeof value.answer !== 'string' ||
    !modes.includes(value.retrieval_mode) ||
    !Array.isArray(value.citations) ||
    !Array.isArray(value.sources) ||
    !Array.isArray(value.graph_fact_ids) ||
    !Array.isArray(value.selected_passage_ids) ||
    (value.uncertainty !== undefined &&
      value.uncertainty !== null &&
      typeof value.uncertainty !== 'string') ||
    !value.graph_fact_ids.every((id: unknown) => typeof id === 'string') ||
    !value.selected_passage_ids.every((id: unknown) => typeof id === 'string')
  )
    return false
  return (
    value.citations.every((citation: unknown) => {
      if (!isRecord(citation)) return false
      return (
        typeof citation.passage_id === 'string' &&
        typeof citation.book_id === 'string' &&
        typeof citation.content_version_id === 'string' &&
        Number.isInteger(citation.section_index) &&
        Number.isInteger(citation.chunk_index) &&
        Array.isArray(citation.source_unit_ids) &&
        citation.source_unit_ids.every(
          (id: unknown) => typeof id === 'string',
        ) &&
        Array.isArray(citation.source_positions) &&
        citation.source_positions.every(
          (position: unknown) =>
            isRecord(position) &&
            typeof position.source_unit_id === 'string' &&
            Number.isInteger(position.unit_index) &&
            typeof position.kind === 'string' &&
            (position.href === undefined ||
              position.href === null ||
              typeof position.href === 'string') &&
            (position.cfi === undefined ||
              position.cfi === null ||
              typeof position.cfi === 'string'),
        ) &&
        (citation.href === undefined ||
          citation.href === null ||
          typeof citation.href === 'string') &&
        (citation.start_cfi === undefined ||
          citation.start_cfi === null ||
          typeof citation.start_cfi === 'string') &&
        (citation.end_cfi === undefined ||
          citation.end_cfi === null ||
          typeof citation.end_cfi === 'string') &&
        typeof citation.excerpt === 'string' &&
        ['complete', 'partial', 'unavailable', 'unreliable'].includes(
          citation.cfi_status,
        )
      )
    }) &&
    value.sources.every(
      (source: unknown) =>
        isRecord(source) &&
        typeof source.passage_id === 'string' &&
        typeof source.score === 'number' &&
        Number.isFinite(source.score) &&
        modes.includes(source.retrieval_method) &&
        [
          'keyword_rank',
          'keyword_score',
          'semantic_rank',
          'semantic_score',
          'graph_score',
          'graph_depth',
        ].every(
          (key) =>
            source[key] === undefined ||
            source[key] === null ||
            (typeof source[key] === 'number' && Number.isFinite(source[key])),
        ),
    )
  )
}
