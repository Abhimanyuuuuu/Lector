const assert = require('node:assert/strict')
const { test } = require('node:test')

const { LocalKeywordIndex } = require('../../.turbo/lector-rag-tests/index.js')
const { hybridSearch } = require('../../.turbo/lector-rag-tests/hybrid.js')
const {
  LocalSemanticIndex,
} = require('../../.turbo/lector-rag-tests/semantic.js')

function makeChunk(overrides = {}) {
  const chunkIndex = overrides.chunkIndex ?? 0
  const sectionIndex = overrides.sectionIndex ?? 0
  const sourceUnitId = overrides.sourceUnitId ?? `paragraph-${chunkIndex}`
  return {
    id: `chunk-${overrides.bookId ?? 'book'}-${
      overrides.bookVersionId ?? 'revision-a'
    }-${sectionIndex}-${chunkIndex}`,
    bookId: 'book',
    bookVersionId: 'revision-a',
    sectionIndex,
    chunkIndex,
    text: `Passage ${chunkIndex}.`,
    sourceUnitIds: [sourceUnitId],
    sourcePositions: [
      {
        sourceUnitId,
        paragraphIndex: chunkIndex,
        kind: 'paragraph',
        href: `text/section-${sectionIndex}.xhtml`,
        cfi: `epubcfi(/6/${sectionIndex + 2}!/4/${chunkIndex + 1})`,
      },
    ],
    startParagraphIndex: chunkIndex,
    endParagraphIndex: chunkIndex,
    href: `text/section-${sectionIndex}.xhtml`,
    heading: 'Chapter heading',
    startCfi: `epubcfi(/6/${sectionIndex + 2}!/4/${chunkIndex + 1})`,
    endCfi: `epubcfi(/6/${sectionIndex + 2}!/4/${chunkIndex + 1})`,
    cfiStatus: 'complete',
    ...overrides,
  }
}

function keywordSource(results, onSearch = () => {}) {
  return {
    search(query, options) {
      onSearch(query, options)
      return results
    },
  }
}

function semanticSource(results, onSearch = () => {}) {
  return {
    async search(query, options) {
      onSearch(query, options)
      return results
    },
  }
}

function keywordResult(chunk, score = 1) {
  return {
    chunk,
    score,
    matchedTerms: [{ term: 'story', frequency: 1, score }],
  }
}

function semanticResult(chunk, similarity = 0.8) {
  return { chunk, similarity }
}

test('deduplicates a chunk found by both sources and keeps source-only results', async () => {
  const common = makeChunk({ id: 'common', chunkIndex: 0 })
  const keywordOnly = makeChunk({ id: 'keyword-only', chunkIndex: 1 })
  const semanticOnly = makeChunk({ id: 'semantic-only', chunkIndex: 2 })
  const results = await hybridSearch('story', {
    bookId: 'book',
    keywordIndex: keywordSource([
      keywordResult(common, 4),
      keywordResult(keywordOnly, 2),
    ]),
    semanticIndex: semanticSource([
      semanticResult(semanticOnly, 0.7),
      semanticResult(common, 0.9),
    ]),
  })

  assert.deepEqual(
    results.map(({ chunk }) => chunk.id),
    ['common', 'semantic-only', 'keyword-only'],
  )
  assert.equal(results.filter(({ chunk }) => chunk.id === 'common').length, 1)
  assert.deepEqual(results[0].keyword, {
    rank: 1,
    score: 4,
    matchedTerms: [{ term: 'story', frequency: 1, score: 4 }],
  })
  assert.deepEqual(results[0].semantic, { rank: 2, similarity: 0.9 })
})

test('uses weighted reciprocal rank fusion with a configurable constant', async () => {
  const keywordFirst = makeChunk({ id: 'keyword-first', chunkIndex: 0 })
  const semanticFirst = makeChunk({ id: 'semantic-first', chunkIndex: 1 })
  const sources = {
    keywordIndex: keywordSource([
      keywordResult(keywordFirst),
      keywordResult(semanticFirst),
    ]),
    semanticIndex: semanticSource([
      semanticResult(semanticFirst),
      semanticResult(keywordFirst),
    ]),
  }

  const tied = await hybridSearch('story', {
    bookId: 'book',
    ...sources,
    rrfConstant: 60,
  })
  assert.ok(Math.abs(tied[0].score - (1 / 61 + 1 / 62)) < 1e-12)
  assert.equal(tied[0].score, tied[1].score)
  assert.deepEqual(
    tied.map(({ chunk }) => chunk.id),
    ['keyword-first', 'semantic-first'],
  )

  const keywordWeighted = await hybridSearch('story', {
    bookId: 'book',
    ...sources,
    keywordWeight: 3,
    semanticWeight: 1,
    rrfConstant: 0,
  })
  assert.deepEqual(
    keywordWeighted.map(({ chunk }) => chunk.id),
    ['keyword-first', 'semantic-first'],
  )
  assert.equal(keywordWeighted[0].score, 3 + 1 / 2)
  assert.equal(keywordWeighted[1].score, 3 / 2 + 1)
})

test('applies source weights, candidate limits, and deterministic tie breaking', async () => {
  const zChunk = makeChunk({ id: 'z', chunkIndex: 1 })
  const aChunk = makeChunk({ id: 'a', chunkIndex: 0 })
  const keywordCalls = []
  const semanticCalls = []
  const results = await hybridSearch('story', {
    bookId: 'book',
    limit: 5,
    candidateLimit: 2,
    keywordWeight: 2,
    semanticWeight: 1,
    keywordIndex: keywordSource(
      [keywordResult(zChunk), keywordResult(aChunk)],
      (...args) => keywordCalls.push(args),
    ),
    semanticIndex: semanticSource(
      [semanticResult(aChunk), semanticResult(zChunk)],
      (...args) => semanticCalls.push(args),
    ),
  })

  assert.equal(results[0].score, 2 / 61 + 1 / 62)
  assert.equal(results[1].score, 2 / 62 + 1 / 61)
  assert.equal(keywordCalls[0][1].limit, 2)
  assert.equal(semanticCalls[0][1].limit, 2)
  assert.deepEqual(
    results.map(({ chunk }) => chunk.id),
    ['z', 'a'],
  )
})

test('handles empty result sets and allows either source to be disabled', async () => {
  let keywordCalls = 0
  let semanticCalls = 0
  const keywordIndex = keywordSource([], () => (keywordCalls += 1))
  const semanticIndex = semanticSource([], () => (semanticCalls += 1))

  assert.deepEqual(
    await hybridSearch('story', {
      bookId: 'book',
      keywordIndex,
      semanticIndex,
    }),
    [],
  )
  assert.deepEqual(
    await hybridSearch('story', {
      bookId: 'book',
      keywordIndex,
      semanticIndex,
      useSemantic: false,
    }),
    [],
  )
  assert.deepEqual(
    await hybridSearch('story', {
      bookId: 'book',
      keywordIndex,
      semanticIndex,
      useKeyword: false,
    }),
    [],
  )
  assert.deepEqual(
    await hybridSearch('  ', {
      bookId: 'book',
      keywordIndex,
      semanticIndex,
    }),
    [],
  )
  assert.equal(keywordCalls, 2)
  assert.equal(semanticCalls, 2)
})

test('requires explicit book scope and rejects conflicting scope options', async () => {
  const chunk = makeChunk()
  const index = keywordSource([keywordResult(chunk)])

  await assert.rejects(
    hybridSearch('story', { keywordIndex: index }),
    /requires bookId/,
  )
  await assert.rejects(
    hybridSearch('story', {
      bookId: 'book',
      allBooks: true,
      keywordIndex: index,
    }),
    /cannot be combined/,
  )
  await assert.rejects(
    hybridSearch('story', {
      bookId: 'book',
      boundary: { bookId: 'other', sectionIndex: 0, chunkIndex: 0 },
      keywordIndex: index,
    }),
    /must match/,
  )

  const allBooks = await hybridSearch('story', {
    allBooks: true,
    keywordIndex: index,
  })
  assert.equal(allBooks.length, 1)
})

test('rejects conflicting provenance for the same chunk identity', async () => {
  const keywordChunk = makeChunk({ id: 'same', sourceUnitId: 'source-a' })
  const semanticChunk = makeChunk({ id: 'same', sourceUnitId: 'source-b' })

  await assert.rejects(
    hybridSearch('story', {
      bookId: 'book',
      keywordIndex: keywordSource([keywordResult(keywordChunk)]),
      semanticIndex: semanticSource([semanticResult(semanticChunk)]),
    }),
    /conflicting provenance/,
  )
})

test('preserves chunk provenance and never mutates source result arrays', async () => {
  const chunk = makeChunk({
    id: 'mapped',
    sourceUnitIds: ['unit-7', 'unit-8'],
    sourcePositions: [
      {
        sourceUnitId: 'unit-7',
        paragraphIndex: 7,
        kind: 'heading',
        href: 'text/chapter.xhtml',
        cfi: 'epubcfi(/6/2!/4/1)',
      },
      {
        sourceUnitId: 'unit-8',
        paragraphIndex: 8,
        kind: 'paragraph',
        href: 'text/chapter.xhtml',
        cfi: 'epubcfi(/6/2!/4/2)',
      },
    ],
    startParagraphIndex: 7,
    endParagraphIndex: 8,
    startCfi: 'epubcfi(/6/2!/4/1)',
    endCfi: 'epubcfi(/6/2!/4/2)',
  })
  const keywords = [keywordResult(chunk)]
  const semantics = [semanticResult(chunk)]
  const beforeKeywords = structuredClone(keywords)
  const beforeSemantics = structuredClone(semantics)
  const results = await hybridSearch('story', {
    bookId: 'book',
    keywordIndex: keywordSource(keywords),
    semanticIndex: semanticSource(semantics),
  })

  assert.deepEqual(keywords, beforeKeywords)
  assert.deepEqual(semantics, beforeSemantics)
  assert.deepEqual(results[0].chunk.sourceUnitIds, ['unit-7', 'unit-8'])
  assert.equal(results[0].chunk.bookVersionId, 'revision-a')
  assert.equal(results[0].chunk.sectionIndex, 0)
  assert.equal(results[0].chunk.chunkIndex, 0)
  assert.equal(results[0].chunk.startParagraphIndex, 7)
  assert.equal(results[0].chunk.endParagraphIndex, 8)
  assert.equal(results[0].chunk.startCfi, 'epubcfi(/6/2!/4/1)')
  assert.equal(results[0].chunk.endCfi, 'epubcfi(/6/2!/4/2)')
})

test('passes boundaries into both indexes and keeps future chunks out of fusion', async () => {
  const chunks = [
    makeChunk({ id: 'eligible', chunkIndex: 0, text: 'lantern' }),
    makeChunk({
      id: 'future',
      sectionIndex: 2,
      chunkIndex: 0,
      text: 'lantern lantern lantern',
    }),
  ]
  const keywords = new LocalKeywordIndex()
  keywords.indexBook('book', chunks)

  const provider = {
    dimensions: 2,
    async embed(texts) {
      return texts.map(() => [1, 0])
    },
  }
  const semantics = new LocalSemanticIndex(provider)
  semantics.indexBookWithEmbeddings('book', chunks, [
    [0.8, 0.6],
    [1, 0],
  ])

  const boundary = { bookId: 'book', sectionIndex: 0, chunkIndex: 0 }
  const results = await hybridSearch('lantern', {
    bookId: 'book',
    boundary,
    limit: 1,
    candidateLimit: 1,
    keywordIndex: keywords,
    semanticIndex: semantics,
  })

  assert.deepEqual(
    results.map(({ chunk }) => chunk.id),
    ['eligible'],
  )
})
