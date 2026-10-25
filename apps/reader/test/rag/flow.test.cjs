const assert = require('node:assert/strict')
const { test } = require('node:test')
const { webcrypto } = require('node:crypto')
const {
  askBook,
  indexSourceUnits,
  RagApiError,
} = require('../../.turbo/lector-rag-tests/api.js')
const { fingerprintEpub } = require('../../.turbo/lector-rag-tests/identity.js')
const {
  extractLoadedBook,
  createBoundaryForLocation,
  navigateToCitation,
} = require('../../.turbo/lector-rag-tests/flow.js')

class ElementFixture {
  constructor(tagName, text = '') {
    this.nodeType = 1
    this.tagName = tagName.toUpperCase()
    this.nodeValue = null
    this.childNodes = text
      ? [{ nodeType: 3, nodeValue: text, childNodes: [] }]
      : []
    this.childNodes.forEach((node) => {
      node.parentElement = this
    })
  }
  descendants() {
    return this.childNodes.flatMap((child) => [
      ...(child.nodeType === 1 ? [child] : []),
      ...(child.nodeType === 1 ? child.descendants() : []),
    ])
  }
  querySelectorAll(selector) {
    const tags = selector.split(',').map((tag) => tag.trim().toUpperCase())
    return this.descendants().filter((element) =>
      tags.includes(element.tagName),
    )
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null
  }
  closest() {
    return null
  }
}

function makeSection(index, text) {
  const body = new ElementFixture('body')
  body.childNodes = [new ElementFixture('p', text)]
  const document = {
    body,
    createRange() {
      return {
        setStart(node) {
          this.startContainer = node
        },
        setEnd(node) {
          this.endContainer = node
        },
      }
    },
  }
  return {
    index,
    href: `text/chapter-${index}.xhtml`,
    document,
    cfiFromRange(range) {
      return `epubcfi(/6/${index + 1}!/${range.startContainer.nodeValue})`
    },
  }
}

test('SHA-256 content versions are stable and distinguish changed EPUB bytes', async () => {
  const a = await fingerprintEpub(new Blob(['book A']), webcrypto)
  assert.equal(a, await fingerprintEpub(new Blob(['book A']), webcrypto))
  assert.notEqual(a, await fingerprintEpub(new Blob(['book B']), webcrypto))
  assert.match(a, /^sha256:[a-f0-9]{64}$/)
})

test('full-spine extraction sorts sections and retains source mapping without a navigation API', async () => {
  const sections = [
    makeSection(1, 'Second chapter.'),
    makeSection(0, 'First chapter.'),
  ]
  const progress = []
  const units = await extractLoadedBook(
    'book-id',
    'sha256:version',
    sections,
    (item) => progress.push(item),
  )
  assert.deepEqual(
    units.map((unit) => [unit.sectionIndex, unit.text]),
    [
      [0, 'First chapter.'],
      [1, 'Second chapter.'],
    ],
  )
  assert.equal(units[0].bookId, 'book-id')
  assert.equal(units[0].bookVersionId, 'sha256:version')
  assert.equal(units[0].href, 'text/chapter-0.xhtml')
  assert.match(units[0].cfi, /^epubcfi\(/)
  assert.deepEqual(
    progress.map((item) => item.completedSections),
    [1, 2],
  )
  await assert.rejects(
    extractLoadedBook('book-id', 'version', [{ index: 0 }]),
    /not fully loaded/,
  )
})

test('confirmed boundary maps to the last completed source-unit range', () => {
  const paragraphs = [
    { id: 'one', sectionIndex: 0, paragraphIndex: 0, cfi: 'range-1' },
    { id: 'bad', sectionIndex: 0, paragraphIndex: 1, cfi: 'invalid' },
    { id: 'two', sectionIndex: 1, paragraphIndex: 0, cfi: 'range-2' },
  ]
  const api = {
    parse(cfi) {
      if (cfi === 'invalid') throw new Error('bad CFI')
      return {
        collapse(toStart) {
          this.value = `${cfi}:${toStart ? 'start' : 'end'}`
        },
        toString() {
          return this.value
        },
      }
    },
    compare(left, right) {
      if (left === 'invalid' || right === 'invalid')
        throw new Error('invalid CFI')
      const number = (value) => Number(/range-(\d+)/.exec(value)?.[1] ?? 0)
      return number(left) - number(right)
    },
  }
  assert.deepEqual(
    createBoundaryForLocation(paragraphs, 'range-1:start', api),
    {
      sectionIndex: 0,
      unitIndex: 0,
      sourceUnitId: 'one',
      cfi: 'range-1',
    },
  )
})

test('citation navigation uses valid exact CFIs and falls back only to known section hrefs', async () => {
  const savedBoundary = Object.freeze({
    contentVersionId: 'version',
    sectionIndex: 0,
    unitIndex: 2,
    sourceUnitId: 'unit-2',
  })
  const citation = {
    passage_id: 'chunk',
    book_id: 'book',
    content_version_id: 'version',
    section_index: 2,
    chunk_index: 0,
    source_unit_ids: ['unit'],
    source_positions: [
      {
        source_unit_id: 'unit',
        unit_index: 0,
        kind: 'paragraph',
        cfi: 'bad-cfi',
      },
    ],
    cfi_status: 'partial',
    excerpt: 'passage',
  }
  const shown = []
  assert.equal(
    await navigateToCitation(
      citation,
      [{ index: 2, href: 'chapter.xhtml' }],
      () => false,
      async (target) => shown.push(target),
    ),
    'section',
  )
  assert.deepEqual(shown, ['chapter.xhtml'])
  citation.source_positions[0].cfi = 'epubcfi(/6/4!/4/2)'
  assert.equal(
    await navigateToCitation(
      citation,
      [{ index: 2, href: 'chapter.xhtml' }],
      (cfi) => cfi.startsWith('epubcfi('),
      async (target) => shown.push(target),
    ),
    'exact',
  )
  assert.equal(
    await navigateToCitation(
      citation,
      [],
      () => false,
      async () => undefined,
    ),
    'unavailable',
  )
  assert.deepEqual(savedBoundary, {
    contentVersionId: 'version',
    sectionIndex: 0,
    unitIndex: 2,
    sourceUnitId: 'unit-2',
  })
})

const boundary = {
  confirmed: true,
  contentVersionId: 'version',
  sectionIndex: 0,
  unitIndex: 1,
  sourceUnitId: 'unit-1',
}
const citationResponse = {
  passage_id: 'chunk-1',
  book_id: 'book',
  content_version_id: 'version',
  section_index: 0,
  chunk_index: 0,
  source_unit_ids: ['unit-1'],
  source_positions: [
    {
      source_unit_id: 'unit-1',
      unit_index: 1,
      kind: 'paragraph',
      href: 'chapter.xhtml',
      cfi: 'epubcfi(/6/2!/4/2)',
    },
  ],
  href: 'chapter.xhtml',
  start_cfi: 'epubcfi(/6/2!/4/2)',
  end_cfi: null,
  cfi_status: 'complete',
  excerpt: 'Supported passage.',
}

test('answer client sends the actual confirmed-boundary schema and validates scoped citations', async () => {
  let request
  const fetchMock = async (url, options) => {
    request = { url, options }
    return new Response(
      JSON.stringify({
        status: 'answered',
        answer: 'Grounded answer.',
        uncertainty: null,
        retrieval_mode: 'keyword',
        citations: [citationResponse],
        sources: [
          { passage_id: 'chunk-1', score: 1, retrieval_method: 'keyword' },
        ],
        graph_fact_ids: [],
        selected_passage_ids: ['chunk-1'],
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )
  }
  const result = await askBook(
    {
      bookId: 'book',
      contentVersionId: 'version',
      query: 'Who?',
      retrievalMode: 'keyword',
      boundary,
    },
    fetchMock,
  )
  assert.equal(result.answer, 'Grounded answer.')
  assert.match(request.url, /\/api\/v1\/answer$/)
  assert.deepEqual(JSON.parse(request.options.body).reading_boundary, {
    confirmed: true,
    section_index: 0,
    unit_index: 1,
    source_unit_id: 'unit-1',
  })
})

test('index client sends ordered source units and reports structured backend errors', async () => {
  let body
  const paragraphs = [
    {
      id: 'unit-0',
      bookId: 'book',
      bookVersionId: 'version',
      sectionIndex: 0,
      paragraphIndex: 0,
      text: 'Text.',
      kind: 'paragraph',
      href: 'chapter.xhtml',
      cfi: 'cfi',
    },
  ]
  const result = await indexSourceUnits(
    'book',
    'version',
    paragraphs,
    async (_url, options) => {
      body = JSON.parse(options.body)
      return new Response(
        JSON.stringify({
          book_id: 'book',
          content_version_id: 'version',
          accepted_source_units: 1,
          chunk_count: 1,
        }),
        { status: 202 },
      )
    },
  )
  assert.deepEqual(result, { chunkCount: 1, sourceUnitCount: 1 })
  assert.equal(body.source_units[0].source_unit_id, 'unit-0')
  await assert.rejects(
    askBook(
      {
        bookId: 'book',
        contentVersionId: 'version',
        query: 'q',
        retrievalMode: 'semantic',
        boundary,
      },
      async () =>
        new Response(
          JSON.stringify({
            error: {
              code: 'local_model_unavailable',
              message: 'Configure a local model.',
            },
          }),
          { status: 503 },
        ),
    ),
    (error) =>
      error instanceof RagApiError &&
      error.code === 'local_model_unavailable' &&
      error.status === 503,
  )
  await assert.rejects(
    askBook(
      {
        bookId: 'book',
        contentVersionId: 'version',
        query: 'q',
        retrievalMode: 'keyword',
        boundary,
      },
      async () => {
        throw new Error('offline')
      },
    ),
    (error) =>
      error instanceof RagApiError && error.code === 'backend_unavailable',
  )
  await assert.rejects(
    askBook(
      {
        bookId: 'book',
        contentVersionId: 'version',
        query: 'q',
        retrievalMode: 'keyword',
        boundary,
      },
      async () => new Response('{bad json', { status: 200 }),
    ),
    (error) =>
      error instanceof RagApiError && error.code === 'invalid_response',
  )
})
