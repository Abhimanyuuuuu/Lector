export type ParagraphKind =
  | 'paragraph'
  | 'heading'
  | 'list-item'
  | 'blockquote'
  | 'preformatted'
  | 'other'

export interface BookParagraph {
  id: string

  bookId: string

  /** Identifies the EPUB content revision used to create this paragraph. */
  bookVersionId: string

  sectionIndex: number

  /** Zero-based order among extracted units in this section. */
  paragraphIndex: number

  text: string

  kind: ParagraphKind

  href?: string

  /** epub.js CFI range; omitted when the source range could not be created. */
  cfi?: string

  /** The most recent heading in this section, when one is available. */
  heading?: string
}

export type ChunkCfiStatus =
  | 'complete'
  | 'partial'
  | 'unavailable'
  | 'unreliable'

/** Source position for a unit included in a chunk, including overlap units. */
export interface ChunkSourcePosition {
  sourceUnitId: string
  paragraphIndex: number
  kind: ParagraphKind
  href?: string
  /** Original source value, if supplied. It is not synthesized or repaired. */
  cfi?: string
}

/** A deterministic, section-local group of source units for later retrieval. */
export interface BookChunk {
  id: string
  bookId: string
  bookVersionId: string
  sectionIndex: number
  /** Zero-based chunk order within the section. */
  chunkIndex: number
  text: string
  sourceUnitIds: string[]
  sourcePositions: ChunkSourcePosition[]
  startParagraphIndex: number
  endParagraphIndex: number
  href?: string
  heading?: string
  startCfi?: string
  endCfi?: string
  cfiStatus: ChunkCfiStatus
}

/** Inclusive, explicitly confirmed reading limit for one book. */
export interface ReadingBoundary {
  bookId: string
  sectionIndex: number
  chunkIndex: number
}

export interface KeywordSearchOptions {
  /** Search only this book. Without it, set allBooks to search the full index. */
  bookId?: string
  /** Explicitly opt into searching across all indexed books in this index. */
  allBooks?: boolean
  /** Optionally restrict a single-book search to its currently indexed revision. */
  bookVersionId?: string
  /** Filters eligible chunks before IDF calculation and ranking. */
  boundary?: ReadingBoundary
  /** Maximum results; defaults to 10. Zero returns no results. */
  limit?: number
}

export interface KeywordTermMatch {
  term: string
  frequency: number
  score: number
}

export interface KeywordSearchResult {
  chunk: BookChunk
  score: number
  matchedTerms: KeywordTermMatch[]
}

export interface SemanticSearchOptions extends KeywordSearchOptions {}

export interface SemanticSearchResult {
  chunk: BookChunk
  /** Cosine similarity, from -1 (opposite) through 1 (identical). */
  similarity: number
}

export interface HybridKeywordSearchSource {
  search(query: string, options?: KeywordSearchOptions): KeywordSearchResult[]
}

export interface HybridSemanticSearchSource {
  search(
    query: string,
    options?: SemanticSearchOptions,
  ): Promise<SemanticSearchResult[]>
}

export interface HybridSearchOptions extends KeywordSearchOptions {
  /** Existing independent keyword index. Enabled by default when supplied. */
  keywordIndex?: HybridKeywordSearchSource
  /** Existing independent semantic index. Enabled by default when supplied. */
  semanticIndex?: HybridSemanticSearchSource
  useKeyword?: boolean
  useSemantic?: boolean
  keywordWeight?: number
  semanticWeight?: number
  /** RRF's additive rank constant; the default is 60. */
  rrfConstant?: number
  /** Maximum candidates requested from each enabled retrieval source. */
  candidateLimit?: number
}

export interface HybridKeywordComponent {
  rank: number
  score: number
  matchedTerms: KeywordTermMatch[]
}

export interface HybridSemanticComponent {
  rank: number
  similarity: number
}

export interface HybridSearchResult {
  chunk: BookChunk
  /** Weighted reciprocal-rank-fusion score. */
  score: number
  keyword?: HybridKeywordComponent
  semantic?: HybridSemanticComponent
}

export interface BookSection {
  bookId: string

  bookVersionId: string

  sectionIndex: number

  href?: string

  title?: string

  paragraphs: BookParagraph[]
}

/** Raw DOM-derived content before text normalization and deterministic IDs. */
export interface ExtractedTextUnit {
  sectionIndex: number
  paragraphIndex: number
  text: string
  kind: ParagraphKind
  href?: string
  cfi?: string
  heading?: string
}

/** Minimal structural shape of Flow's loaded epub.js spine sections. */
export interface LoadedEpubSection {
  index: number
  href?: string
  document?: Document
  cfiFromRange?: (range: Range) => string
  navitem?: { label?: string }
}
