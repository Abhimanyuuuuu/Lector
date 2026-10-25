import type {
  BookChunk,
  ReadingBoundary,
  SemanticSearchOptions,
  SemanticSearchResult,
} from './types'

/**
 * Provider-neutral embedding boundary. `embed` accepts arbitrary texts, so the
 * same injected implementation can embed book chunks and user queries. It must
 * return one finite, non-empty vector per input, all with `dimensions` values.
 * This interface does not choose a model or make network calls by itself.
 */
export interface EmbeddingProvider {
  readonly dimensions: number
  embed(texts: readonly string[]): Promise<readonly (readonly number[])[]>
}

interface SemanticDocument {
  chunk: BookChunk
  vector: readonly number[]
}

interface SearchScope {
  bookId?: string
  bookVersionId?: string
  boundary?: ReadingBoundary
  allBooks: boolean
  limit: number
}

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const isNonNegativeSafeInteger = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0

const validateDimensions = (dimensions: unknown): number => {
  if (!Number.isSafeInteger(dimensions) || (dimensions as number) <= 0) {
    throw new RangeError('embedding dimensions must be a positive safe integer')
  }
  return dimensions as number
}

const validateVector = (
  vector: readonly number[],
  dimensions: number,
  label: string,
): number[] => {
  if (!Array.isArray(vector)) {
    throw new TypeError(`${label} must be an array of numbers`)
  }
  if (vector.length !== dimensions) {
    throw new RangeError(
      `${label} has ${vector.length} dimensions; expected ${dimensions}`,
    )
  }
  if (
    !vector.every(
      (value) => typeof value === 'number' && Number.isFinite(value),
    )
  ) {
    throw new TypeError(`${label} must contain only finite numbers`)
  }
  return [...vector]
}

/** Cosine similarity in [-1, 1]; a zero-length direction scores as 0. */
export const cosineSimilarity = (
  left: readonly number[],
  right: readonly number[],
): number => {
  if (!Array.isArray(left) || !Array.isArray(right)) {
    throw new TypeError('cosine similarity requires two vector arrays')
  }
  if (left.length === 0 || right.length === 0) {
    throw new RangeError('cosine similarity vectors must not be empty')
  }
  if (left.length !== right.length) {
    throw new RangeError('cosine similarity vectors must have equal dimensions')
  }
  validateVector(left, left.length, 'left vector')
  validateVector(right, right.length, 'right vector')

  let leftScale = 0
  let rightScale = 0
  for (let index = 0; index < left.length; index += 1) {
    leftScale = Math.max(leftScale, Math.abs(left[index]!))
    rightScale = Math.max(rightScale, Math.abs(right[index]!))
  }
  if (leftScale === 0 || rightScale === 0) return 0

  let dotProduct = 0
  let leftMagnitude = 0
  let rightMagnitude = 0
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index]! / leftScale
    const rightValue = right[index]! / rightScale
    dotProduct += leftValue * rightValue
    leftMagnitude += leftValue * leftValue
    rightMagnitude += rightValue * rightValue
  }
  return Math.max(
    -1,
    Math.min(1, dotProduct / Math.sqrt(leftMagnitude * rightMagnitude)),
  )
}

const cloneChunk = (chunk: BookChunk): BookChunk => ({
  ...chunk,
  sourceUnitIds: [...chunk.sourceUnitIds],
  sourcePositions: chunk.sourcePositions.map((position) => ({ ...position })),
})

const validateChunk = (chunk: unknown, bookId: string): BookChunk => {
  if (typeof chunk !== 'object' || chunk === null) {
    throw new TypeError('each indexed chunk must be an object')
  }
  const candidate = chunk as BookChunk
  if (!isNonEmptyString(candidate.id)) {
    throw new TypeError('chunk id must be a non-empty string')
  }
  if (candidate.bookId !== bookId) {
    throw new TypeError('all chunks must match the bookId passed to indexBook')
  }
  if (!isNonEmptyString(candidate.bookVersionId)) {
    throw new TypeError('chunk bookVersionId must be a non-empty string')
  }
  if (
    !isNonNegativeSafeInteger(candidate.sectionIndex) ||
    !isNonNegativeSafeInteger(candidate.chunkIndex) ||
    !isNonNegativeSafeInteger(candidate.startParagraphIndex) ||
    !isNonNegativeSafeInteger(candidate.endParagraphIndex)
  ) {
    throw new TypeError(
      'chunk source positions must be non-negative safe integers',
    )
  }
  if (typeof candidate.text !== 'string') {
    throw new TypeError('chunk text must be a string')
  }
  if (
    !Array.isArray(candidate.sourceUnitIds) ||
    !candidate.sourceUnitIds.every(isNonEmptyString) ||
    !Array.isArray(candidate.sourcePositions) ||
    !candidate.sourcePositions.every(
      (position) =>
        typeof position === 'object' &&
        position !== null &&
        isNonEmptyString(position.sourceUnitId) &&
        isNonNegativeSafeInteger(position.paragraphIndex),
    )
  ) {
    throw new TypeError(
      'chunk source provenance must contain valid source positions',
    )
  }
  if (candidate.sourceUnitIds.length !== candidate.sourcePositions.length) {
    throw new TypeError(
      'chunk source IDs and source positions must have equal lengths',
    )
  }
  return cloneChunk(candidate)
}

const validateBoundary = (boundary: ReadingBoundary): void => {
  if (
    !isNonEmptyString(boundary.bookId) ||
    !isNonNegativeSafeInteger(boundary.sectionIndex) ||
    !isNonNegativeSafeInteger(boundary.chunkIndex)
  ) {
    throw new TypeError(
      'boundary must identify a book and non-negative source position',
    )
  }
}

const isAtOrBeforeBoundary = (
  chunk: BookChunk,
  boundary: ReadingBoundary,
): boolean =>
  chunk.sectionIndex < boundary.sectionIndex ||
  (chunk.sectionIndex === boundary.sectionIndex &&
    chunk.chunkIndex <= boundary.chunkIndex)

const compareText = (left: string, right: string): number =>
  left < right ? -1 : left > right ? 1 : 0

const validateOptions = (options: SemanticSearchOptions): SearchScope => {
  if (typeof options !== 'object' || options === null) {
    throw new TypeError('search options must be an object')
  }
  const limit = options.limit ?? 10
  if (!isNonNegativeSafeInteger(limit)) {
    throw new RangeError('limit must be a non-negative safe integer')
  }
  if (options.bookId !== undefined && !isNonEmptyString(options.bookId)) {
    throw new TypeError('bookId must be a non-empty string when supplied')
  }
  if (
    options.bookVersionId !== undefined &&
    !isNonEmptyString(options.bookVersionId)
  ) {
    throw new TypeError(
      'bookVersionId must be a non-empty string when supplied',
    )
  }
  if (options.allBooks !== undefined && typeof options.allBooks !== 'boolean') {
    throw new TypeError('allBooks must be a boolean when supplied')
  }
  if (options.allBooks && options.bookId) {
    throw new TypeError('allBooks cannot be combined with bookId')
  }

  const boundary = options.boundary
  if (boundary !== undefined) {
    if (typeof boundary !== 'object' || boundary === null) {
      throw new TypeError('boundary must be an object when supplied')
    }
    validateBoundary(boundary)
    if (options.bookId !== undefined && options.bookId !== boundary.bookId) {
      throw new TypeError('bookId and boundary.bookId must match')
    }
    if (options.allBooks) {
      throw new TypeError('a reading boundary cannot be combined with allBooks')
    }
  }
  if (options.bookVersionId && !options.bookId && !boundary) {
    throw new TypeError('bookVersionId requires a bookId or reading boundary')
  }

  const bookId = boundary?.bookId ?? options.bookId
  if (!bookId && !options.allBooks) {
    throw new TypeError(
      'search requires bookId, boundary, or explicit allBooks: true',
    )
  }

  return {
    bookId,
    bookVersionId: options.bookVersionId,
    boundary,
    allBooks: options.allBooks === true,
    limit,
  }
}

/**
 * In-memory semantic index over existing source-mapped chunks. No embedding
 * provider is selected by default and no text is transmitted unless a caller
 * injects a provider and invokes an embedding-backed method.
 */
export class LocalSemanticIndex {
  private readonly books = new Map<string, SemanticDocument[]>()
  private dimensions?: number

  constructor(private readonly provider?: EmbeddingProvider) {
    if (provider !== undefined) {
      if (typeof provider.embed !== 'function') {
        throw new TypeError('embedding provider must implement embed')
      }
      this.dimensions = validateDimensions(provider.dimensions)
    }
  }

  /** Embeds chunk text with the explicitly injected provider and replaces the book index. */
  async indexBook(bookId: string, chunks: readonly BookChunk[]): Promise<void> {
    const prepared = this.prepareChunks(bookId, chunks)
    const indexable = prepared.filter(({ text }) => text.trim().length > 0)
    if (indexable.length === 0) {
      this.books.delete(bookId)
      return
    }
    if (!this.provider) {
      throw new Error(
        'indexBook requires an explicitly supplied embedding provider',
      )
    }

    const vectors = await this.provider.embed(indexable.map(({ text }) => text))
    this.indexBookWithEmbeddings(bookId, indexable, vectors)
  }

  /** Stores caller-supplied vectors without choosing or invoking a provider. */
  indexBookWithEmbeddings(
    bookId: string,
    chunks: readonly BookChunk[],
    vectors: readonly (readonly number[])[],
  ): void {
    const prepared = this.prepareChunks(bookId, chunks)
    if (!Array.isArray(vectors) || vectors.length !== chunks.length) {
      throw new RangeError('one embedding vector is required per input chunk')
    }
    if (chunks.length === 0) {
      this.books.delete(bookId)
      return
    }

    const dimensions = this.dimensions ?? validateDimensions(vectors[0]?.length)
    const uniqueIds = new Set<string>()
    const documents: SemanticDocument[] = []
    for (let index = 0; index < chunks.length; index += 1) {
      const chunk = prepared[index]!
      const vector = validateVector(
        vectors[index],
        dimensions,
        `embedding ${index}`,
      )
      if (chunk.text.trim().length === 0) continue
      if (uniqueIds.has(chunk.id)) {
        throw new TypeError(`duplicate chunk ID: ${chunk.id}`)
      }
      uniqueIds.add(chunk.id)
      documents.push({ chunk, vector })
    }

    if (documents.length === 0) {
      this.books.delete(bookId)
      return
    }
    documents.sort(
      (left, right) =>
        left.chunk.sectionIndex - right.chunk.sectionIndex ||
        left.chunk.chunkIndex - right.chunk.chunkIndex ||
        compareText(left.chunk.id, right.chunk.id),
    )
    this.dimensions = dimensions
    this.books.set(bookId, documents)
  }

  removeBook(bookId: string): void {
    if (!isNonEmptyString(bookId)) {
      throw new TypeError('bookId must be a non-empty string')
    }
    this.books.delete(bookId)
  }

  /** Embeds the query with the injected provider, then performs cosine search. */
  async search(
    query: string,
    options: SemanticSearchOptions = {},
  ): Promise<SemanticSearchResult[]> {
    if (typeof query !== 'string') {
      throw new TypeError('query must be a string')
    }
    const scope = validateOptions(options)
    if (query.trim().length === 0 || scope.limit === 0) return []

    const candidates = this.getEligibleDocuments(scope)
    if (candidates.length === 0) return []
    if (!this.provider) {
      throw new Error(
        'search requires an explicitly supplied embedding provider',
      )
    }

    const vectors = await this.provider.embed([query])
    if (!Array.isArray(vectors) || vectors.length !== 1) {
      throw new RangeError(
        'embedding provider must return one vector for the query',
      )
    }
    return this.rank(vectors[0]!, candidates, scope.limit)
  }

  /** Searches using an already-generated query vector and makes no provider call. */
  searchVector(
    queryVector: readonly number[],
    options: SemanticSearchOptions = {},
  ): SemanticSearchResult[] {
    const scope = validateOptions(options)
    if (scope.limit === 0) return []
    const candidates = this.getEligibleDocuments(scope)
    if (candidates.length === 0) return []
    if (this.dimensions === undefined) return []
    const vector = validateVector(
      queryVector,
      this.dimensions,
      'query embedding',
    )
    return this.rank(vector, candidates, scope.limit)
  }

  private prepareChunks(
    bookId: string,
    chunks: readonly BookChunk[],
  ): BookChunk[] {
    if (!isNonEmptyString(bookId)) {
      throw new TypeError('bookId must be a non-empty string')
    }
    if (!Array.isArray(chunks)) {
      throw new TypeError('chunks must be an array')
    }
    const seen = new Set<string>()
    let versionId: string | undefined
    const prepared = chunks.map((inputChunk) => {
      const chunk = validateChunk(inputChunk, bookId)
      if (versionId !== undefined && chunk.bookVersionId !== versionId) {
        throw new TypeError(
          'indexBook accepts only one content version per book',
        )
      }
      versionId = chunk.bookVersionId
      if (seen.has(chunk.id))
        throw new TypeError(`duplicate chunk ID: ${chunk.id}`)
      seen.add(chunk.id)
      return chunk
    })
    return prepared
  }

  private getEligibleDocuments(scope: SearchScope): SemanticDocument[] {
    const eligible: SemanticDocument[] = []
    for (const [bookId, documents] of this.books) {
      if (scope.bookId && bookId !== scope.bookId) continue
      for (const document of documents) {
        const chunk = document.chunk
        if (
          scope.bookVersionId &&
          chunk.bookVersionId !== scope.bookVersionId
        ) {
          continue
        }
        if (scope.boundary && !isAtOrBeforeBoundary(chunk, scope.boundary)) {
          continue
        }
        eligible.push(document)
      }
    }
    return eligible
  }

  private rank(
    queryVector: readonly number[],
    candidates: readonly SemanticDocument[],
    limit: number,
  ): SemanticSearchResult[] {
    const vector = validateVector(
      queryVector,
      this.dimensions!,
      'query embedding',
    )
    return candidates
      .map(({ chunk, vector: embedding }) => ({
        chunk,
        similarity: cosineSimilarity(vector, embedding),
      }))
      .sort(
        (left, right) =>
          right.similarity - left.similarity ||
          left.chunk.sectionIndex - right.chunk.sectionIndex ||
          left.chunk.chunkIndex - right.chunk.chunkIndex ||
          compareText(left.chunk.id, right.chunk.id),
      )
      .slice(0, limit)
      .map(({ chunk, similarity }) => ({
        chunk: cloneChunk(chunk),
        similarity,
      }))
  }
}
