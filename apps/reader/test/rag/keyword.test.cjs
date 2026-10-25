const assert = require('node:assert/strict')
const { test } = require('node:test')

const { LocalKeywordIndex } = require('../../.turbo/lector-rag-tests/index.js')
const { tokenize } = require('../../.turbo/lector-rag-tests/search.js')

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
    text: 'A passage about the book.',
    sourceUnitIds: [sourceUnitId],
    sourcePositions: [
      {
        sourceUnitId,
        paragraphIndex: chunkIndex,
        kind: 'paragraph',
        href: `text/section-${sectionIndex}.xhtml`,
        cfi: 'epubcfi(/6/2!/4/1)',
      },
    ],
    startParagraphIndex: chunkIndex,
    endParagraphIndex: chunkIndex,
    href: `text/section-${sectionIndex}.xhtml`,
    startCfi: 'epubcfi(/6/2!/4/1)',
    endCfi: 'epubcfi(/6/2!/4/1)',
    cfiStatus: 'complete',
    ...overrides,
  }
}

test('returns no results for an empty index or tokenless query', () => {
  const index = new LocalKeywordIndex()
  assert.deepEqual(index.search('lantern', { bookId: 'book' }), [])
  index.indexBook('book', [makeChunk()])
  assert.deepEqual(index.search('  ... ', { bookId: 'book' }), [])
  assert.deepEqual(index.search('lantern', { bookId: 'book', limit: 0 }), [])
})

test('matches case-insensitively and tokenizes ordinary prose consistently', () => {
  assert.deepEqual(tokenize('MIRA’s well-known plan: river 42!'), [
    "mira's",
    'well',
    'known',
    'plan',
    'river',
    '42',
  ])

  const index = new LocalKeywordIndex()
  index.indexBook('book', [makeChunk({ text: 'Mira found the Silver Key.' })])
  assert.equal(
    index.search('silver KEY', { bookId: 'book' })[0].chunk.id,
    makeChunk().id,
  )
})

test('ranks chunks by BM25 relevance and reports matching terms', () => {
  const index = new LocalKeywordIndex()
  const relevant = makeChunk({
    chunkIndex: 0,
    id: 'relevant',
    text: 'Mira holds the key. Mira hides the key.',
  })
  const partial = makeChunk({
    chunkIndex: 1,
    id: 'partial',
    text: 'Mira reads a letter.',
  })
  const unrelated = makeChunk({
    chunkIndex: 2,
    id: 'unrelated',
    text: 'The boat crossed the river.',
  })
  index.indexBook('book', [unrelated, partial, relevant])

  const results = index.search('MIRA key', { bookId: 'book' })
  assert.deepEqual(
    results.map(({ chunk }) => chunk.id),
    ['relevant', 'partial'],
  )
  assert.deepEqual(
    results[0].matchedTerms.map(({ term }) => term),
    ['key', 'mira'],
  )
  assert.ok(results[0].score > results[1].score)
  assert.ok(
    results[0].matchedTerms.every(
      ({ frequency, score }) => frequency > 0 && score > 0,
    ),
  )
})

test('returns deterministic results and duplicate query terms do not inflate ranking', () => {
  const index = new LocalKeywordIndex()
  const chunks = [
    makeChunk({ id: 'first', chunkIndex: 0, text: 'The lantern glowed.' }),
    makeChunk({ id: 'second', chunkIndex: 1, text: 'A lantern appeared.' }),
  ]
  index.indexBook('book', chunks)
  const normal = index.search('lantern', { bookId: 'book' })
  const duplicated = index.search('LANTERN lantern lantern', { bookId: 'book' })

  assert.deepEqual(normal, index.search('lantern', { bookId: 'book' }))
  assert.deepEqual(normal, duplicated)
  index.indexBook('book', chunks)
  assert.deepEqual(normal, index.search('lantern', { bookId: 'book' }))
})

test('deduplicates identical chunk IDs and rejects conflicting duplicates', () => {
  const index = new LocalKeywordIndex()
  const chunk = makeChunk({ text: 'one lantern' })
  index.indexBook('book', [chunk, chunk])
  assert.equal(index.search('lantern', { bookId: 'book' }).length, 1)
  assert.throws(
    () =>
      index.indexBook('book', [chunk, { ...chunk, text: 'another lantern' }]),
    /duplicate chunk ID has conflicting content/,
  )
  assert.equal(index.search('lantern', { bookId: 'book' }).length, 1)
})

test('isolates books and replaces a book revision without stale chunks', () => {
  const index = new LocalKeywordIndex()
  index.indexBook('book', [
    makeChunk({ id: 'old', text: 'oldword', bookVersionId: 'revision-a' }),
  ])
  index.indexBook('other-book', [
    makeChunk({
      id: 'other',
      bookId: 'other-book',
      text: 'sharedword',
      bookVersionId: 'revision-other',
    }),
  ])

  assert.deepEqual(index.search('sharedword', { bookId: 'book' }), [])
  assert.equal(
    index.search('sharedword', { allBooks: true })[0].chunk.bookId,
    'other-book',
  )
  assert.deepEqual(
    index.search('oldword', { bookId: 'book' }).map(({ chunk }) => chunk.id),
    ['old'],
  )

  index.indexBook('book', [
    makeChunk({ id: 'new', text: 'newword', bookVersionId: 'revision-b' }),
  ])
  assert.deepEqual(index.search('oldword', { bookId: 'book' }), [])
  assert.deepEqual(
    index
      .search('newword', { bookId: 'book' })
      .map(({ chunk }) => [chunk.id, chunk.bookVersionId]),
    [['new', 'revision-b']],
  )
})

test('skips whitespace-only chunks and validates malformed index input', () => {
  const index = new LocalKeywordIndex()
  index.indexBook('book', [makeChunk({ text: '  \n ' })])
  assert.deepEqual(index.search('passage', { bookId: 'book' }), [])

  assert.throws(() => index.indexBook('book', [null]), /must be an object/)
  assert.throws(
    () => index.indexBook('book', [makeChunk({ bookId: 'other-book' })]),
    /must match the bookId/,
  )
  assert.throws(
    () => index.indexBook('book', [makeChunk({ sourcePositions: [] })]),
    /equal lengths/,
  )
  assert.throws(
    () =>
      index.indexBook('book', [
        makeChunk({ bookVersionId: 'revision-a' }),
        makeChunk({ id: 'another-revision', bookVersionId: 'revision-b' }),
      ]),
    /only one content version/,
  )
  assert.throws(
    () => index.search(12, { bookId: 'book' }),
    /query must be a string/,
  )
  assert.throws(() => index.search('passage'), /requires bookId/)
  assert.throws(
    () => index.search('passage', { bookId: 'book', limit: -1 }),
    /limit/,
  )
  assert.throws(
    () => index.search('passage', { bookId: 'book', allBooks: true }),
    /cannot be combined with bookId/,
  )
})

test('filters by an inclusive reading boundary before calculating BM25 scores', () => {
  const eligible = [
    makeChunk({
      id: 'eligible-a',
      chunkIndex: 0,
      text: 'lantern appears once',
    }),
    makeChunk({ id: 'eligible-b', chunkIndex: 1, text: 'a quiet room' }),
  ]
  const future = makeChunk({
    id: 'future',
    sectionIndex: 2,
    chunkIndex: 0,
    text: 'lantern lantern lantern lantern lantern',
  })
  const combinedIndex = new LocalKeywordIndex()
  combinedIndex.indexBook('book', [...eligible, future])
  const eligibleOnlyIndex = new LocalKeywordIndex()
  eligibleOnlyIndex.indexBook('book', [eligible[0]])
  const boundary = { bookId: 'book', sectionIndex: 0, chunkIndex: 0 }

  const boundedResults = combinedIndex.search('lantern', { boundary })
  const eligibleOnlyResults = eligibleOnlyIndex.search('lantern', {
    bookId: 'book',
  })
  assert.deepEqual(boundedResults, eligibleOnlyResults)
  assert.deepEqual(
    boundedResults.map(({ chunk }) => chunk.id),
    ['eligible-a'],
  )
  assert.deepEqual(
    combinedIndex
      .search('lantern', {
        boundary: { bookId: 'book', sectionIndex: 2, chunkIndex: 0 },
      })
      .map(({ chunk }) => chunk.id),
    ['future', 'eligible-a'],
  )
})

test('preserves chunk provenance in results and isolates returned snapshots', () => {
  const index = new LocalKeywordIndex()
  const chunk = makeChunk({
    id: 'mapped',
    text: 'A mapped lantern passage.',
    sourceUnitIds: ['paragraph-4', 'paragraph-5'],
    sourcePositions: [
      {
        sourceUnitId: 'paragraph-4',
        paragraphIndex: 4,
        kind: 'heading',
        href: 'text/chapter.xhtml',
        cfi: 'epubcfi(/6/2!/4/1)',
      },
      {
        sourceUnitId: 'paragraph-5',
        paragraphIndex: 5,
        kind: 'paragraph',
        href: 'text/chapter.xhtml',
        cfi: 'epubcfi(/6/2!/4/2)',
      },
    ],
    startParagraphIndex: 4,
    endParagraphIndex: 5,
    startCfi: 'epubcfi(/6/2!/4/1)',
    endCfi: 'epubcfi(/6/2!/4/2)',
  })
  index.indexBook('book', [chunk])

  const result = index.search('lantern', { bookId: 'book' })[0]
  assert.deepEqual(result.chunk.sourceUnitIds, ['paragraph-4', 'paragraph-5'])
  assert.equal(result.chunk.startParagraphIndex, 4)
  assert.equal(result.chunk.endParagraphIndex, 5)
  assert.equal(result.chunk.sourcePositions[0].kind, 'heading')
  assert.equal(result.chunk.startCfi, 'epubcfi(/6/2!/4/1)')
  result.chunk.sourceUnitIds.pop()
  assert.deepEqual(
    index.search('lantern', { bookId: 'book' })[0].chunk.sourceUnitIds,
    ['paragraph-4', 'paragraph-5'],
  )
})
