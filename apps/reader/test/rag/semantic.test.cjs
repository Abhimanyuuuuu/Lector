const assert = require('node:assert/strict')
const { test } = require('node:test')

const {
  cosineSimilarity,
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
    startCfi: `epubcfi(/6/${sectionIndex + 2}!/4/${chunkIndex + 1})`,
    endCfi: `epubcfi(/6/${sectionIndex + 2}!/4/${chunkIndex + 1})`,
    cfiStatus: 'complete',
    ...overrides,
  }
}

function makeProvider(embeddings, dimensions = 2) {
  const calls = []
  return {
    calls,
    provider: {
      dimensions,
      async embed(texts) {
        calls.push([...texts])
        return texts.map((text) => embeddings[text])
      },
    },
  }
}

test('cosine similarity covers identical, orthogonal, opposite, and zero vectors', () => {
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1)
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0)
  assert.equal(cosineSimilarity([1, 0], [-1, 0]), -1)
  assert.equal(cosineSimilarity([0, 0], [1, 0]), 0)
  assert.equal(cosineSimilarity([0, 0], [0, 0]), 0)
  assert.equal(cosineSimilarity([1e308, 1e308], [1e308, 1e308]), 1)
  assert.throws(() => cosineSimilarity([], []), /must not be empty/)
  assert.throws(() => cosineSimilarity([1], [1, 0]), /equal dimensions/)
  assert.throws(() => cosineSimilarity([Number.NaN], [1]), /finite numbers/)
})

test('embeds chunks and queries through the injected provider and ranks by cosine', async () => {
  const { provider, calls } = makeProvider({
    alpha: [1, 0],
    beta: [0.8, 0.6],
    gamma: [0, 1],
    question: [1, 0],
  })
  const index = new LocalSemanticIndex(provider)
  const chunks = [
    makeChunk({ id: 'gamma', chunkIndex: 2, text: 'gamma' }),
    makeChunk({ id: 'beta', chunkIndex: 1, text: 'beta' }),
    makeChunk({ id: 'alpha', chunkIndex: 0, text: 'alpha' }),
  ]

  await index.indexBook('book', chunks)
  const results = await index.search('question', { bookId: 'book' })

  assert.deepEqual(calls, [['gamma', 'beta', 'alpha'], ['question']])
  assert.deepEqual(
    results.map(({ chunk, similarity }) => [chunk.id, similarity]),
    [
      ['alpha', 1],
      ['beta', 0.8],
      ['gamma', 0],
    ],
  )
})

test('returns empty results for an empty index and avoids embedding empty queries', async () => {
  const { provider, calls } = makeProvider({ query: [1, 0] })
  const index = new LocalSemanticIndex(provider)

  assert.deepEqual(await index.search('query', { bookId: 'book' }), [])
  assert.deepEqual(await index.search('   ', { bookId: 'book' }), [])
  assert.deepEqual(index.searchVector([1, 0], { bookId: 'book' }), [])
  assert.deepEqual(calls, [])
  await index.indexBook('book', [])
  assert.deepEqual(calls, [])
})

test('accepts supplied vectors and validates dimensions and finite values', () => {
  const index = new LocalSemanticIndex()
  const chunk = makeChunk({ text: 'embedded passage' })

  index.indexBookWithEmbeddings('book', [chunk], [[1, 0]])
  assert.equal(index.searchVector([1, 0], { bookId: 'book' })[0].similarity, 1)
  assert.throws(
    () => index.indexBookWithEmbeddings('book', [chunk], [[1]]),
    /dimensions/,
  )
  assert.throws(
    () =>
      index.indexBookWithEmbeddings(
        'book',
        [chunk],
        [[1, Number.POSITIVE_INFINITY]],
      ),
    /finite numbers/,
  )
  assert.throws(() => index.searchVector([1], { bookId: 'book' }), /dimensions/)
  assert.throws(
    () => index.searchVector([1, Number.NaN], { bookId: 'book' }),
    /finite numbers/,
  )
  assert.equal(
    index.searchVector([1, 0], { bookId: 'book' })[0].chunk.id,
    chunk.id,
  )
})

test('validates provider dimensions, output count, and returned vectors', async () => {
  assert.throws(
    () =>
      new LocalSemanticIndex({
        dimensions: 0,
        async embed() {
          return []
        },
      }),
    /dimensions/,
  )

  let response = [[1, 0]]
  const provider = {
    dimensions: 2,
    async embed() {
      return response
    },
  }
  const index = new LocalSemanticIndex(provider)
  const chunk = makeChunk({ text: 'valid content' })
  response = [[1]]
  await assert.rejects(index.indexBook('book', [chunk]), /dimensions/)
  assert.deepEqual(index.searchVector([1, 0], { bookId: 'book' }), [])

  response = [[1, 0]]
  await index.indexBook('book', [chunk])
  response = [
    [Number.NaN, 0],
    [0, 1],
  ]
  await assert.rejects(index.search('query', { bookId: 'book' }), /one vector/)
  response = [[Number.NaN, 0]]
  await assert.rejects(
    index.search('query', { bookId: 'book' }),
    /finite numbers/,
  )
})

test('rejects missing embeddings and duplicate chunk identifiers predictably', async () => {
  const index = new LocalSemanticIndex()
  const chunk = makeChunk({ text: 'content' })

  await assert.rejects(
    index.indexBook('book', [chunk]),
    /requires an explicitly supplied/,
  )
  assert.throws(
    () => index.indexBookWithEmbeddings('book', [chunk], []),
    /one embedding vector is required/,
  )
  assert.throws(
    () =>
      index.indexBookWithEmbeddings(
        'book',
        [chunk, chunk],
        [
          [1, 0],
          [1, 0],
        ],
      ),
    /duplicate chunk ID/,
  )
  index.indexBookWithEmbeddings('book', [chunk], [[1, 0]])
  assert.throws(
    () => index.indexBookWithEmbeddings('book', [chunk], [[1, 0, 0]]),
    /dimensions/,
  )
})

test('isolates books and replaces a book content version', () => {
  const index = new LocalSemanticIndex()
  const oldChunk = makeChunk({
    id: 'old',
    bookVersionId: 'revision-a',
    text: 'old content',
  })
  index.indexBookWithEmbeddings('book', [oldChunk], [[1, 0]])
  index.indexBookWithEmbeddings(
    'other-book',
    [makeChunk({ id: 'other', bookId: 'other-book', text: 'other content' })],
    [[1, 0]],
  )

  assert.deepEqual(
    index
      .searchVector([1, 0], { bookId: 'other-book' })
      .map(({ chunk }) => chunk.id),
    ['other'],
  )
  assert.deepEqual(
    index.searchVector([1, 0], { bookId: 'book' }).map(({ chunk }) => chunk.id),
    ['old'],
  )

  const newChunk = makeChunk({
    id: 'new',
    bookVersionId: 'revision-b',
    text: 'new content',
  })
  index.indexBookWithEmbeddings('book', [newChunk], [[0, 1]])
  assert.deepEqual(
    index
      .searchVector([1, 0], { bookId: 'book' })
      .map(({ chunk }) => [chunk.id, chunk.bookVersionId]),
    [['new', 'revision-b']],
  )
})

test('filters boundary-ineligible chunks before semantic ranking', () => {
  const index = new LocalSemanticIndex()
  const eligible = makeChunk({
    id: 'eligible',
    chunkIndex: 0,
    text: 'eligible',
  })
  const future = makeChunk({
    id: 'future',
    sectionIndex: 2,
    chunkIndex: 0,
    text: 'future',
  })
  index.indexBookWithEmbeddings(
    'book',
    [eligible, future],
    [
      [0.8, 0.6],
      [1, 0],
    ],
  )

  const results = index.searchVector([1, 0], {
    boundary: { bookId: 'book', sectionIndex: 0, chunkIndex: 0 },
  })
  assert.deepEqual(
    results.map(({ chunk }) => chunk.id),
    ['eligible'],
  )
  assert.equal(results[0].similarity, 0.8)
})

test('preserves source mapping in results and returns detached snapshots', () => {
  const index = new LocalSemanticIndex()
  const chunk = makeChunk({
    id: 'mapped',
    text: 'source mapped passage',
    sourceUnitIds: ['source-4', 'source-5'],
    sourcePositions: [
      {
        sourceUnitId: 'source-4',
        paragraphIndex: 4,
        kind: 'heading',
        cfi: 'epubcfi(/4/1)',
      },
      {
        sourceUnitId: 'source-5',
        paragraphIndex: 5,
        kind: 'paragraph',
        cfi: 'epubcfi(/4/2)',
      },
    ],
    startParagraphIndex: 4,
    endParagraphIndex: 5,
  })
  index.indexBookWithEmbeddings('book', [chunk], [[1, 0]])

  const result = index.searchVector([1, 0], { bookId: 'book' })[0]
  assert.deepEqual(result.chunk.sourceUnitIds, ['source-4', 'source-5'])
  assert.equal(result.chunk.sourcePositions[0].kind, 'heading')
  assert.equal(result.chunk.sourcePositions[1].paragraphIndex, 5)
  assert.equal(result.chunk.startParagraphIndex, 4)
  assert.equal(result.chunk.endParagraphIndex, 5)
  assert.equal(result.chunk.startCfi, chunk.startCfi)
  result.chunk.sourcePositions.pop()
  assert.equal(
    index.searchVector([1, 0], { bookId: 'book' })[0].chunk.sourcePositions
      .length,
    2,
  )
})
