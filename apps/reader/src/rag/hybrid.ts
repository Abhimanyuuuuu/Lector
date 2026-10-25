import type {
  BookChunk,
  HybridSearchOptions,
  HybridSearchResult,
  KeywordSearchOptions,
  KeywordSearchResult,
  ReadingBoundary,
  SemanticSearchOptions,
  SemanticSearchResult,
} from './types'

const DEFAULT_RRF_CONSTANT = 60
const DEFAULT_RESULT_LIMIT = 10

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const isNonNegativeSafeInteger = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0

const compareText = (left: string, right: string): number =>
  left < right ? -1 : left > right ? 1 : 0

const cloneChunk = (chunk: BookChunk): BookChunk => ({
  ...chunk,
  sourceUnitIds: [...chunk.sourceUnitIds],
  sourcePositions: chunk.sourcePositions.map((position) => ({ ...position })),
})

/** All fields that define the chunk identity and its source provenance. */
const chunkFingerprint = (chunk: BookChunk): string =>
  JSON.stringify([
    chunk.id,
    chunk.bookId,
    chunk.bookVersionId,
    chunk.sectionIndex,
    chunk.chunkIndex,
    chunk.text,
    chunk.sourceUnitIds,
    chunk.sourcePositions.map((position) => [
      position.sourceUnitId,
      position.paragraphIndex,
      position.kind,
      position.href,
      position.cfi,
    ]),
    chunk.startParagraphIndex,
    chunk.endParagraphIndex,
    chunk.href,
    chunk.heading,
    chunk.startCfi,
    chunk.endCfi,
    chunk.cfiStatus,
  ])

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

const validateOptions = (
  options: HybridSearchOptions,
): {
  keywordEnabled: boolean
  semanticEnabled: boolean
  keywordWeight: number
  semanticWeight: number
  rrfConstant: number
  resultLimit: number
  candidateLimit: number
  sourceOptions: KeywordSearchOptions & SemanticSearchOptions
} => {
  if (typeof options !== 'object' || options === null) {
    throw new TypeError('hybrid search options must be an object')
  }

  const limit = options.limit ?? DEFAULT_RESULT_LIMIT
  const candidateLimit = options.candidateLimit ?? limit
  if (!isNonNegativeSafeInteger(limit)) {
    throw new RangeError('limit must be a non-negative safe integer')
  }
  if (!isNonNegativeSafeInteger(candidateLimit)) {
    throw new RangeError('candidateLimit must be a non-negative safe integer')
  }
  const rrfConstant = options.rrfConstant ?? DEFAULT_RRF_CONSTANT
  if (!isNonNegativeSafeInteger(rrfConstant)) {
    throw new RangeError('rrfConstant must be a non-negative safe integer')
  }

  const keywordWeight = options.keywordWeight ?? 1
  const semanticWeight = options.semanticWeight ?? 1
  if (!Number.isFinite(keywordWeight) || keywordWeight <= 0) {
    throw new RangeError('keywordWeight must be a finite positive number')
  }
  if (!Number.isFinite(semanticWeight) || semanticWeight <= 0) {
    throw new RangeError('semanticWeight must be a finite positive number')
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
  if (!options.bookId && !boundary && !options.allBooks) {
    throw new TypeError(
      'hybrid search requires bookId, boundary, or explicit allBooks: true',
    )
  }

  const keywordEnabled =
    options.useKeyword ?? options.keywordIndex !== undefined
  const semanticEnabled =
    options.useSemantic ?? options.semanticIndex !== undefined
  if (keywordEnabled && !options.keywordIndex) {
    throw new TypeError(
      'keyword retrieval is enabled but keywordIndex is missing',
    )
  }
  if (semanticEnabled && !options.semanticIndex) {
    throw new TypeError(
      'semantic retrieval is enabled but semanticIndex is missing',
    )
  }
  if (!keywordEnabled && !semanticEnabled) {
    throw new TypeError('at least one hybrid retrieval source must be enabled')
  }

  const sourceOptions: KeywordSearchOptions & SemanticSearchOptions = {
    bookId: options.bookId,
    bookVersionId: options.bookVersionId,
    allBooks: options.allBooks,
    boundary,
    limit: candidateLimit,
  }
  return {
    keywordEnabled,
    semanticEnabled,
    keywordWeight,
    semanticWeight,
    rrfConstant,
    resultLimit: limit,
    candidateLimit,
    sourceOptions,
  }
}

const isEligible = (
  chunk: BookChunk,
  options: HybridSearchOptions,
): boolean => {
  if (options.bookId && chunk.bookId !== options.bookId) return false
  if (options.bookVersionId && chunk.bookVersionId !== options.bookVersionId) {
    return false
  }
  const boundary = options.boundary
  if (!boundary) return true
  return (
    chunk.bookId === boundary.bookId &&
    (chunk.sectionIndex < boundary.sectionIndex ||
      (chunk.sectionIndex === boundary.sectionIndex &&
        chunk.chunkIndex <= boundary.chunkIndex))
  )
}

interface RankedEntry {
  chunk: BookChunk
  score: number
  keyword?: HybridSearchResult['keyword']
  semantic?: HybridSearchResult['semantic']
}

const addKeywordResults = (
  entries: Map<string, RankedEntry>,
  results: readonly KeywordSearchResult[],
  weight: number,
  rrfConstant: number,
  options: HybridSearchOptions,
): void => {
  const seen = new Map<string, string>()
  results.forEach((result, index) => {
    const { chunk } = result
    if (!isEligible(chunk, options)) return
    if (!isNonEmptyString(chunk.id) || !Number.isFinite(result.score)) {
      throw new TypeError(
        'keyword results must include a chunk ID and finite score',
      )
    }
    const fingerprint = chunkFingerprint(chunk)
    const prior = seen.get(chunk.id)
    if (prior !== undefined) {
      if (prior !== fingerprint) {
        throw new TypeError(`conflicting provenance for chunk ID: ${chunk.id}`)
      }
      return
    }
    seen.set(chunk.id, fingerprint)

    const existing = entries.get(chunk.id)
    if (existing && chunkFingerprint(existing.chunk) !== fingerprint) {
      throw new TypeError(`conflicting provenance for chunk ID: ${chunk.id}`)
    }
    const rank = index + 1
    const component = {
      rank,
      score: result.score,
      matchedTerms: result.matchedTerms.map((match) => ({ ...match })),
    }
    if (existing) {
      existing.score += weight / (rrfConstant + rank)
      existing.keyword = component
    } else {
      entries.set(chunk.id, {
        chunk,
        score: weight / (rrfConstant + rank),
        keyword: component,
      })
    }
  })
}

const addSemanticResults = (
  entries: Map<string, RankedEntry>,
  results: readonly SemanticSearchResult[],
  weight: number,
  rrfConstant: number,
  options: HybridSearchOptions,
): void => {
  const seen = new Map<string, string>()
  results.forEach((result, index) => {
    const { chunk } = result
    if (!isEligible(chunk, options)) return
    if (
      !isNonEmptyString(chunk.id) ||
      !Number.isFinite(result.similarity) ||
      result.similarity < -1 ||
      result.similarity > 1
    ) {
      throw new TypeError(
        'semantic results must include a chunk ID and cosine score in [-1, 1]',
      )
    }
    const fingerprint = chunkFingerprint(chunk)
    const prior = seen.get(chunk.id)
    if (prior !== undefined) {
      if (prior !== fingerprint) {
        throw new TypeError(`conflicting provenance for chunk ID: ${chunk.id}`)
      }
      return
    }
    seen.set(chunk.id, fingerprint)

    const existing = entries.get(chunk.id)
    if (existing && chunkFingerprint(existing.chunk) !== fingerprint) {
      throw new TypeError(`conflicting provenance for chunk ID: ${chunk.id}`)
    }
    const rank = index + 1
    const component = { rank, similarity: result.similarity }
    if (existing) {
      existing.score += weight / (rrfConstant + rank)
      existing.semantic = component
    } else {
      entries.set(chunk.id, {
        chunk,
        score: weight / (rrfConstant + rank),
        semantic: component,
      })
    }
  })
}

/**
 * Runs the enabled independent retrieval sources and combines their ranked
 * candidates with weighted reciprocal rank fusion. RRF is
 * `sum(weight / (rrfConstant + oneBasedRank))`; the default constant is 60 and
 * both source weights default to 1. A reading boundary is forwarded to each
 * index so it filters before source ranking, then eligibility is checked again
 * defensively before fusion.
 */
export const hybridSearch = async (
  query: string,
  options: HybridSearchOptions,
): Promise<HybridSearchResult[]> => {
  if (typeof query !== 'string') {
    throw new TypeError('query must be a string')
  }
  const config = validateOptions(options)
  if (query.trim().length === 0 || config.resultLimit === 0) return []

  const keywordPromise = config.keywordEnabled
    ? Promise.resolve(options.keywordIndex!.search(query, config.sourceOptions))
    : Promise.resolve([] as KeywordSearchResult[])
  const semanticPromise = config.semanticEnabled
    ? options.semanticIndex!.search(query, config.sourceOptions)
    : Promise.resolve([] as SemanticSearchResult[])
  const [keywordResults, semanticResults] = await Promise.all([
    keywordPromise,
    semanticPromise,
  ])

  const entries = new Map<string, RankedEntry>()
  if (config.keywordEnabled) {
    addKeywordResults(
      entries,
      keywordResults,
      config.keywordWeight,
      config.rrfConstant,
      options,
    )
  }
  if (config.semanticEnabled) {
    addSemanticResults(
      entries,
      semanticResults,
      config.semanticWeight,
      config.rrfConstant,
      options,
    )
  }

  return [...entries.values()]
    .sort(
      (left, right) =>
        right.score - left.score ||
        left.chunk.sectionIndex - right.chunk.sectionIndex ||
        left.chunk.chunkIndex - right.chunk.chunkIndex ||
        compareText(left.chunk.bookId, right.chunk.bookId) ||
        compareText(left.chunk.bookVersionId, right.chunk.bookVersionId) ||
        compareText(left.chunk.id, right.chunk.id),
    )
    .slice(0, config.resultLimit)
    .map(({ chunk, ...result }) => ({ chunk: cloneChunk(chunk), ...result }))
}
