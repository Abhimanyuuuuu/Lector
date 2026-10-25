import type { BookChunk, KeywordSearchResult, KeywordTermMatch } from './types'

/** Tokenizes Unicode letters and numbers; apostrophes inside words are kept.
 * Hyphens and other punctuation separate tokens. Matching is lowercase. */
export const tokenize = (text: string): string[] => {
  if (typeof text !== 'string') {
    throw new TypeError('text to tokenize must be a string')
  }

  const normalized = text.toLowerCase().replace(/’/g, "'")
  return normalized.match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) ?? []
}

/** Indexed form used by the local index; it contains no external resources. */
export interface KeywordDocument {
  chunk: BookChunk
  termFrequency: Map<string, number>
  length: number
}

const K1 = 1.2
const B = 0.75

const compareText = (left: string, right: string): number =>
  left < right ? -1 : left > right ? 1 : 0

/**
 * Ranks a pre-filtered document set with BM25. The caller applies book and
 * reading-boundary filters before calling so future text cannot affect either
 * candidate selection or ranking statistics.
 */
export const rankKeywordDocuments = (
  documents: readonly KeywordDocument[],
  query: string,
  limit: number,
): KeywordSearchResult[] => {
  const terms = [...new Set(tokenize(query))].sort(compareText)
  if (terms.length === 0 || documents.length === 0 || limit === 0) return []

  const averageLength =
    documents.reduce((total, document) => total + document.length, 0) /
    documents.length
  const documentFrequency = new Map<string, number>()
  for (const term of terms) {
    documentFrequency.set(
      term,
      documents.reduce(
        (count, document) => count + (document.termFrequency.has(term) ? 1 : 0),
        0,
      ),
    )
  }

  const ranked = documents.flatMap((document) => {
    const matchedTerms: KeywordTermMatch[] = []
    for (const term of terms) {
      const frequency = document.termFrequency.get(term) ?? 0
      if (frequency === 0) continue

      const frequencyInDocuments = documentFrequency.get(term) ?? 0
      const inverseDocumentFrequency = Math.log(
        1 +
          (documents.length - frequencyInDocuments + 0.5) /
            (frequencyInDocuments + 0.5),
      )
      const lengthNormalization =
        averageLength === 0 ? 1 : document.length / averageLength
      const termScore =
        (inverseDocumentFrequency * frequency * (K1 + 1)) /
        (frequency + K1 * (1 - B + B * lengthNormalization))

      matchedTerms.push({ term, frequency, score: termScore })
    }

    if (matchedTerms.length === 0) return []
    return [
      {
        chunk: document.chunk,
        score: matchedTerms.reduce((total, match) => total + match.score, 0),
        matchedTerms,
      },
    ]
  })

  ranked.sort(
    (left, right) =>
      right.score - left.score ||
      left.chunk.sectionIndex - right.chunk.sectionIndex ||
      left.chunk.chunkIndex - right.chunk.chunkIndex ||
      compareText(left.chunk.id, right.chunk.id),
  )
  return ranked.slice(0, limit)
}
