import type {
  BookChunk,
  BookParagraph,
  ChunkCfiStatus,
  ChunkSourcePosition,
} from './types'

/** Character budget uses JavaScript string.length (UTF-16 code units), including separators. */
export const DEFAULT_MAX_CHUNK_SIZE = 1200
/** Default overlap; for smaller custom budgets it is capped at one quarter. */
export const DEFAULT_CHUNK_OVERLAP = 150

export interface ChunkOptions {
  maxChunkSize?: number
  overlap?: number
}

const compareText = (left: string, right: string): number =>
  left < right ? -1 : left > right ? 1 : 0

const serializedText = (paragraphs: BookParagraph[]): string =>
  paragraphs.map(({ text }) => text).join('\n\n')

const validateOptions = (
  options: ChunkOptions,
): { maxChunkSize: number; overlap: number } => {
  const maxChunkSize = options.maxChunkSize ?? DEFAULT_MAX_CHUNK_SIZE
  const overlap =
    options.overlap ??
    Math.min(DEFAULT_CHUNK_OVERLAP, Math.floor(maxChunkSize / 4))

  if (!Number.isSafeInteger(maxChunkSize) || maxChunkSize <= 0) {
    throw new RangeError('maxChunkSize must be a positive safe integer')
  }
  if (
    !Number.isSafeInteger(overlap) ||
    overlap < 0 ||
    overlap >= maxChunkSize
  ) {
    throw new RangeError(
      'overlap must be a non-negative integer smaller than maxChunkSize',
    )
  }

  return { maxChunkSize, overlap }
}

const isCfi = (value: string | undefined): value is string =>
  value !== undefined && /^epubcfi\(.+\)$/.test(value)

const sourceRangeStatus = (
  paragraphs: BookParagraph[],
): { status: ChunkCfiStatus; startCfi?: string; endCfi?: string } => {
  const supplied = paragraphs
    .map(({ cfi }) => cfi)
    .filter((cfi) => cfi !== undefined)
  const hasUnreliable = supplied.some((cfi) => !isCfi(cfi))
  const validCount = supplied.filter(isCfi).length
  let status: ChunkCfiStatus

  if (hasUnreliable) status = 'unreliable'
  else if (validCount === 0) status = 'unavailable'
  else if (validCount === paragraphs.length) status = 'complete'
  else status = 'partial'

  const firstCfi = paragraphs[0]?.cfi
  const lastCfi = paragraphs[paragraphs.length - 1]?.cfi
  return {
    status,
    startCfi: isCfi(firstCfi) ? firstCfi : undefined,
    endCfi: isCfi(lastCfi) ? lastCfi : undefined,
  }
}

const createChunk = (
  paragraphs: BookParagraph[],
  chunkIndex: number,
  maxChunkSize: number,
  overlap: number,
): BookChunk => {
  const first = paragraphs[0]!
  const last = paragraphs[paragraphs.length - 1]!
  const sourcePositions: ChunkSourcePosition[] = paragraphs.map(
    ({ id, paragraphIndex, kind, href, cfi }) => ({
      sourceUnitId: id,
      paragraphIndex,
      kind,
      href,
      cfi,
    }),
  )
  let heading = first.heading
  for (const paragraph of paragraphs) {
    if (paragraph.kind === 'heading') heading = paragraph.text
  }
  const cfi = sourceRangeStatus(paragraphs)
  const sourceIds = paragraphs.map(({ id }) => id)
  const idParts = [
    first.bookId,
    first.bookVersionId,
    String(first.sectionIndex),
    String(chunkIndex),
    String(maxChunkSize),
    String(overlap),
    ...sourceIds,
  ]

  return {
    id: `chunk:${idParts.map(encodeURIComponent).join(':')}`,
    bookId: first.bookId,
    bookVersionId: first.bookVersionId,
    sectionIndex: first.sectionIndex,
    chunkIndex,
    text: serializedText(paragraphs),
    sourceUnitIds: sourceIds,
    sourcePositions,
    startParagraphIndex: first.paragraphIndex,
    endParagraphIndex: last.paragraphIndex,
    href: first.href,
    heading,
    startCfi: cfi.startCfi,
    endCfi: cfi.endCfi,
    cfiStatus: cfi.status,
  }
}

const makeOverlap = (
  paragraphs: BookParagraph[],
  overlap: number,
  maxChunkSize: number,
): BookParagraph[] => {
  if (overlap === 0 || paragraphs.length === 0) return []

  const suffix: BookParagraph[] = []
  for (let index = paragraphs.length - 1; index >= 0; index -= 1) {
    const paragraph = paragraphs[index]!
    if (paragraph.text.length > maxChunkSize) break
    const candidate = [paragraph, ...suffix]
    if (serializedText(candidate).length > overlap) break
    suffix.unshift(paragraph)
  }
  return suffix
}

const compareParagraphs = (left: BookParagraph, right: BookParagraph): number =>
  left.paragraphIndex - right.paragraphIndex ||
  compareText(left.id, right.id) ||
  compareText(left.text, right.text) ||
  compareText(left.kind, right.kind) ||
  compareText(left.href ?? '', right.href ?? '') ||
  compareText(left.cfi ?? '', right.cfi ?? '') ||
  compareText(left.heading ?? '', right.heading ?? '')

/**
 * Combines non-empty paragraph units into deterministic chunks. Sections and
 * book-content identities are isolated. Units are kept whole; a unit larger
 * than the budget becomes a chunk by itself. Overlap copies whole trailing
 * units and is measured using the same joined text representation as the
 * chunk budget. No reading-state or spoiler eligibility is inferred here.
 */
export const chunkParagraphs = (
  input: readonly BookParagraph[],
  options: ChunkOptions = {},
): BookChunk[] => {
  const { maxChunkSize, overlap } = validateOptions(options)
  const groups = new Map<string, BookParagraph[]>()

  for (const paragraph of input) {
    if (paragraph.text.length === 0) continue
    const key = JSON.stringify([
      paragraph.bookId,
      paragraph.bookVersionId,
      paragraph.sectionIndex,
    ])
    const group = groups.get(key)
    if (group) group.push(paragraph)
    else groups.set(key, [paragraph])
  }

  const orderedGroups = [...groups.values()].sort((left, right) => {
    const a = left[0]!
    const b = right[0]!
    return (
      compareText(a.bookId, b.bookId) ||
      compareText(a.bookVersionId, b.bookVersionId) ||
      a.sectionIndex - b.sectionIndex
    )
  })
  const chunks: BookChunk[] = []

  for (const group of orderedGroups) {
    const paragraphs = [...group].sort(compareParagraphs)
    let cursor = 0
    let chunkIndex = 0
    let carry: BookParagraph[] = []

    while (cursor < paragraphs.length) {
      let current = [...carry]
      carry = []
      let freshCount = 0
      let oversizedEmitted = false

      while (cursor < paragraphs.length) {
        const next = paragraphs[cursor]!
        const hasBody = current.some(({ kind }) => kind !== 'heading')

        // Start a new chunk at headings where possible, keeping following
        // content with the heading instead of stranding it at a prior end.
        if (next.kind === 'heading' && current.length > 0 && hasBody) break

        const candidate = [...current, next]
        if (serializedText(candidate).length > maxChunkSize) {
          if (freshCount === 0 && current.length > 0) {
            // An overlap-only prefix left no room for new content. Drop it and
            // retry this same unit so we never emit duplicate-only chunks.
            current = []
            continue
          }
          if (current.length === 0) {
            // Oversized units remain intact and never become overlap sources.
            chunks.push(createChunk([next], chunkIndex, maxChunkSize, overlap))
            chunkIndex += 1
            cursor += 1
            oversizedEmitted = true
          }
          break
        }

        current.push(next)
        cursor += 1
        freshCount += 1
      }

      if (oversizedEmitted) continue
      if (freshCount === 0) {
        // A heading boundary can be encountered while current contains only
        // carried units. Discard that carry and begin at the heading.
        if (current.length > 0) continue
        break
      }

      chunks.push(createChunk(current, chunkIndex, maxChunkSize, overlap))
      chunkIndex += 1
      if (cursor < paragraphs.length) {
        carry = makeOverlap(current, overlap, maxChunkSize)
      }
    }
  }

  return chunks
}
