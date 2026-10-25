import type { AnswerCitation, ConfirmedBoundary } from './api'
import { extractBookSections } from './extract'
import type { BookParagraph, LoadedEpubSection } from './types'

export const MAX_RAG_SOURCE_UNITS = 100_000
export const MAX_RAG_TEXT_CODE_UNITS = 10_000_000

export interface ExtractionProgress {
  completedSections: number
  totalSections: number
  sourceUnits: number
}

export type FlowSection = LoadedEpubSection

/** Extract the full loaded spine without displaying sections or changing location. */
export async function extractLoadedBook(
  bookId: string,
  contentVersionId: string,
  sections: readonly FlowSection[],
  onProgress: (progress: ExtractionProgress) => void = () => undefined,
): Promise<BookParagraph[]> {
  if (sections.length === 0)
    throw new Error('The EPUB spine has no readable sections.')
  const ordered = sections
    .slice()
    .sort((left, right) => left.index - right.index)
  const paragraphs: BookParagraph[] = []
  let textCodeUnits = 0
  for (let index = 0; index < ordered.length; index++) {
    const section = ordered[index]!
    if (!section.document) {
      throw new Error(
        `Section ${
          section.index + 1
        } is not fully loaded; no content was indexed.`,
      )
    }
    const extracted = extractBookSections(bookId, contentVersionId, [
      section,
    ])[0]!
    paragraphs.push(...extracted.paragraphs)
    textCodeUnits += extracted.paragraphs.reduce(
      (sum, paragraph) => sum + paragraph.text.length,
      0,
    )
    if (
      paragraphs.length > MAX_RAG_SOURCE_UNITS ||
      textCodeUnits > MAX_RAG_TEXT_CODE_UNITS
    ) {
      throw new Error(
        'This EPUB exceeds the safe indexing size limit; no partial book was sent.',
      )
    }
    onProgress({
      completedSections: index + 1,
      totalSections: ordered.length,
      sourceUnits: paragraphs.length,
    })
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
  }
  return paragraphs
}

/** Map a displayed CFI to the last whole source unit completed before it. */
export function boundaryAtOrBefore(
  paragraphs: readonly BookParagraph[],
  currentCfi: string,
  endOfRange: (cfi: string) => string,
  compareCfi: (left: string, right: string) => number,
): Omit<ConfirmedBoundary, 'contentVersionId' | 'confirmed'> | undefined {
  let boundary: BookParagraph | undefined
  for (const paragraph of paragraphs) {
    if (!paragraph.cfi) continue
    try {
      if (compareCfi(endOfRange(paragraph.cfi), currentCfi) <= 0) {
        boundary = paragraph
      }
    } catch {
      // Invalid ranges are not eligible for a confirmed-position mapping.
    }
  }
  if (!boundary) return undefined
  return {
    sectionIndex: boundary.sectionIndex,
    unitIndex: boundary.paragraphIndex,
    sourceUnitId: boundary.id,
    cfi: boundary.cfi,
  }
}



export function createBoundaryForLocation(
  paragraphs: readonly BookParagraph[],
  currentCfi: string,
  cfiApi: {
    parse(cfi: string): unknown
    compare(left: string, right: string): number
  },
): Omit<ConfirmedBoundary, 'contentVersionId' | 'confirmed'> | undefined {
  return boundaryAtOrBefore(
    paragraphs,
    currentCfi,
    (sourceCfi) => sourceCfi,
    (left, right) => cfiApi.compare(left, right),
  )
}



export type CitationNavigation = 'exact' | 'section' | 'unavailable'

/** Navigate only to validated citation CFIs or an href from the loaded spine. */
export async function navigateToCitation(
  citation: AnswerCitation,
  sections: readonly { index: number; href?: string }[],
  isValidCfi: (cfi: string) => boolean,
  display: (target: string) => Promise<unknown>,
): Promise<CitationNavigation> {
  const sourceCfi = citation.source_positions.find(
    (position) => position.cfi,
  )?.cfi
  const cfi =
    citation.cfi_status === 'unreliable' ||
    citation.cfi_status === 'unavailable'
      ? undefined
      : sourceCfi || citation.start_cfi || undefined
  if (cfi && isValidCfi(cfi)) {
    try {
      await display(cfi)
      return 'exact'
    } catch {
      // Try the known section below; do not claim an exact jump succeeded.
    }
  }
  const section = sections.find((item) => item.index === citation.section_index)
  if (section?.href) {
    try {
      await display(section.href)
      return 'section'
    } catch {
      return 'unavailable'
    }
  }
  return 'unavailable'
}
