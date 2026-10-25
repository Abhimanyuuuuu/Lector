import { BookParagraph, ExtractedTextUnit } from './types'

export interface ParagraphIdentity {
  bookId: string
  bookVersionId: string
}

/** Collapse layout whitespace while preserving punctuation and word spacing. */
export function normalizeParagraphText(text: string) {
  return text.replace(/[\s\u00a0]+/g, ' ').trim()
}

/**
 * Create a deterministic ID scoped to a particular EPUB content revision.
 * The caller must provide a revision identity distinct from the mutable book ID.
 */
export function createParagraphId(
  identity: ParagraphIdentity,
  unit: Pick<ExtractedTextUnit, 'sectionIndex' | 'paragraphIndex'>,
) {
  return [
    encodeURIComponent(identity.bookId),
    encodeURIComponent(identity.bookVersionId),
    unit.sectionIndex,
    unit.paragraphIndex,
  ].join(':')
}

export function toBookParagraph(
  identity: ParagraphIdentity,
  unit: ExtractedTextUnit,
): BookParagraph | undefined {
  const text =
    unit.kind === 'preformatted' ? unit.text : normalizeParagraphText(unit.text)
  if (!text || !Number.isInteger(unit.sectionIndex) || unit.sectionIndex < 0) {
    return undefined
  }
  if (!Number.isInteger(unit.paragraphIndex) || unit.paragraphIndex < 0) {
    return undefined
  }

  return {
    id: createParagraphId(identity, unit),
    bookId: identity.bookId,
    bookVersionId: identity.bookVersionId,
    sectionIndex: unit.sectionIndex,
    paragraphIndex: unit.paragraphIndex,
    text,
    kind: unit.kind,
    href: unit.href,
    cfi: unit.cfi,
    heading: unit.heading,
  }
}

export function processExtractedUnits(
  identity: ParagraphIdentity,
  units: ExtractedTextUnit[],
) {
  return units
    .slice()
    .sort(
      (a, b) =>
        a.sectionIndex - b.sectionIndex || a.paragraphIndex - b.paragraphIndex,
    )
    .map((unit) => toBookParagraph(identity, unit))
    .filter((paragraph): paragraph is BookParagraph => !!paragraph)
}
