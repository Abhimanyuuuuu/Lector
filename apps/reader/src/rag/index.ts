import type {
  BookChunk,
  KeywordSearchOptions,
  KeywordSearchResult,
  ReadingBoundary,
} from './types'
import { rankKeywordDocuments, tokenize, type KeywordDocument } from './search'

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const isNonNegativeSafeInteger = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0

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

/**
 * In-memory, local keyword index. Each indexBook call replaces that book's
 * complete indexed revision; it never merges an old revision into the new one.
 */
export class LocalKeywordIndex {
  private readonly books = new Map<string, KeywordDocument[]>()

  /** Replaces a book's index atomically; an empty array removes its prior data. */
  indexBook(bookId: string, chunks: readonly BookChunk[]): void {
    if (!isNonEmptyString(bookId)) {
      throw new TypeError('bookId must be a non-empty string')
    }
    if (!Array.isArray(chunks)) {
      throw new TypeError('chunks must be an array')
    }

    const unique = new Map<string, BookChunk>()
    let bookVersionId: string | undefined
    for (const inputChunk of chunks) {
      const chunk = validateChunk(inputChunk, bookId)
      if (
        bookVersionId !== undefined &&
        chunk.bookVersionId !== bookVersionId
      ) {
        throw new TypeError(
          'indexBook accepts only one content version per book',
        )
      }
      bookVersionId = chunk.bookVersionId

      const existing = unique.get(chunk.id)
      if (existing) {
        if (JSON.stringify(existing) !== JSON.stringify(chunk)) {
          throw new TypeError(
            `duplicate chunk ID has conflicting content: ${chunk.id}`,
          )
        }
        continue
      }
      unique.set(chunk.id, chunk)
    }

    const documents = [...unique.values()]
      .filter(({ text }) => text.trim().length > 0)
      .sort(
        (left, right) =>
          left.sectionIndex - right.sectionIndex ||
          left.chunkIndex - right.chunkIndex ||
          compareText(left.id, right.id),
      )
      .map((chunk) => {
        const termFrequency = new Map<string, number>()
        const terms = tokenize(chunk.text)
        for (const term of terms) {
          termFrequency.set(term, (termFrequency.get(term) ?? 0) + 1)
        }
        return { chunk, termFrequency, length: terms.length }
      })

    if (documents.length === 0) this.books.delete(bookId)
    else this.books.set(bookId, documents)
  }

  removeBook(bookId: string): void {
    if (!isNonEmptyString(bookId)) {
      throw new TypeError('bookId must be a non-empty string')
    }
    this.books.delete(bookId)
  }

  /**
   * Search one book by default. Searching a multi-book index requires the
   * explicit allBooks option. A boundary implies its book and is applied
   * before BM25 statistics and ranking are calculated.
   */
  search(
    query: string,
    options: KeywordSearchOptions = {},
  ): KeywordSearchResult[] {
    if (typeof query !== 'string') {
      throw new TypeError('query must be a string')
    }
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
    if (
      options.allBooks !== undefined &&
      typeof options.allBooks !== 'boolean'
    ) {
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
        throw new TypeError(
          'a reading boundary cannot be combined with allBooks',
        )
      }
    }
    if (options.bookVersionId && !options.bookId && !boundary) {
      throw new TypeError('bookVersionId requires a bookId or reading boundary')
    }

    const queryTerms = tokenize(query)
    if (queryTerms.length === 0 || limit === 0) return []

    const scopedBookId = boundary?.bookId ?? options.bookId
    if (!scopedBookId && !options.allBooks) {
      throw new TypeError(
        'search requires bookId, boundary, or explicit allBooks: true',
      )
    }

    const eligibleDocuments: KeywordDocument[] = []
    for (const [bookId, documents] of this.books) {
      if (scopedBookId && bookId !== scopedBookId) continue
      for (const document of documents) {
        const chunk = document.chunk
        if (
          options.bookVersionId &&
          chunk.bookVersionId !== options.bookVersionId
        ) {
          continue
        }
        if (boundary && !isAtOrBeforeBoundary(chunk, boundary)) continue
        eligibleDocuments.push(document)
      }
    }

    return rankKeywordDocuments(eligibleDocuments, query, limit).map(
      (result) => ({
        ...result,
        chunk: cloneChunk(result.chunk),
        matchedTerms: result.matchedTerms.map((match) => ({ ...match })),
      }),
    )
  }
}
