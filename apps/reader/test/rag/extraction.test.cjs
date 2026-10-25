const assert = require('node:assert/strict')
const { test } = require('node:test')
const { chunkParagraphs } = require('../../.turbo/lector-rag-tests/chunk.js')

const {
  extractBookSections,
  extractSectionUnits,
} = require('../../.turbo/lector-rag-tests/extract.js')
const {
  createParagraphId,
  normalizeParagraphText,
  processExtractedUnits,
  toBookParagraph,
} = require('../../.turbo/lector-rag-tests/paragraph.js')

function makeParagraph(overrides = {}) {
  const paragraphIndex = overrides.paragraphIndex ?? 0
  return {
    id: `p-${paragraphIndex}`,
    bookId: 'book',
    bookVersionId: 'revision-a',
    sectionIndex: 0,
    paragraphIndex,
    text: `Text ${paragraphIndex}.`,
    kind: 'paragraph',
    href: 'text/chapter-0.xhtml',
    ...overrides,
  }
}

class FixtureNode {
  constructor(nodeType, nodeValue = null) {
    this.nodeType = nodeType
    this.nodeValue = nodeValue
    this.childNodes = []
    this.parentElement = null
  }

  append(...children) {
    for (const child of children) {
      const node = typeof child === 'string' ? new FixtureNode(3, child) : child
      node.parentElement = this.nodeType === 1 ? this : null
      this.childNodes.push(node)
    }
    return this
  }
}

class FixtureElement extends FixtureNode {
  constructor(tagName) {
    super(1)
    this.tagName = tagName.toUpperCase()
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

  closest(selector) {
    const tags = selector.split(',').map((tag) => tag.trim().toUpperCase())
    let element = this
    while (element) {
      if (tags.includes(element.tagName)) return element
      element = element.parentElement
    }
    return null
  }
}

class FixtureRange {
  setStart(node, offset) {
    this.startContainer = node
    this.startOffset = offset
  }

  setEnd(node, offset) {
    this.endContainer = node
    this.endOffset = offset
  }
}

class FixtureDocument {
  constructor(body) {
    this.body = body
    this.documentElement = body
  }

  createRange() {
    return new FixtureRange()
  }
}

function makeSection(index, body, options = {}) {
  return {
    index,
    href: options.href ?? `text/chapter-${index}.xhtml`,
    navitem: options.title ? { label: options.title } : undefined,
    document: body ? new FixtureDocument(body) : undefined,
    cfiFromRange:
      options.cfiFromRange ??
      ((range) => {
        assert.ok(
          range.startContainer && range.endContainer,
          'CFI generation receives a range over readable text nodes',
        )
        return `epubcfi(/6/${index}!/${range.startContainer.parentElement.tagName.toLowerCase()})`
      }),
  }
}

test('extracts headings, paragraphs, and list items once in document order', () => {
  const body = new FixtureElement('body').append(
    new FixtureElement('h1').append('Chapter\n One'),
    new FixtureElement('div').append(
      new FixtureElement('p').append(
        'A ',
        new FixtureElement('em').append('meaningful'),
        ' phrase — intact.',
        new FixtureElement('script').append('not book text'),
      ),
      new FixtureElement('p').append('Second paragraph.'),
      new FixtureElement('p').append(
        'Line one',
        new FixtureElement('br'),
        'Line two.',
      ),
    ),
    new FixtureElement('ol').append(
      new FixtureElement('li').append(
        new FixtureElement('p').append('First item.'),
      ),
      new FixtureElement('li').append('Second item.'),
    ),
  )

  const units = extractSectionUnits(
    makeSection(2, body, { title: 'Nav title' }),
  )

  assert.deepEqual(
    units.map(({ text, kind }) => [normalizeParagraphText(text), kind]),
    [
      ['Chapter One', 'heading'],
      ['A meaningful phrase — intact.', 'paragraph'],
      ['Second paragraph.', 'paragraph'],
      ['Line one Line two.', 'paragraph'],
      ['First item.', 'list-item'],
      ['Second item.', 'list-item'],
    ],
  )
  assert.deepEqual(
    units.map((unit) => unit.paragraphIndex),
    [0, 1, 2, 3, 4, 5],
  )
  assert.equal(units[0].heading, 'Chapter\n One')
  assert.equal(units[1].heading, 'Chapter\n One')
  assert.equal(units[3].cfi, 'epubcfi(/6/2!/p)')
})

test('handles empty sections, missing documents, and empty elements safely', () => {
  const emptyBody = new FixtureElement('body').append(
    new FixtureElement('p').append('  \n  '),
    new FixtureElement('div'),
  )

  assert.deepEqual(extractSectionUnits(makeSection(0, emptyBody)), [])
  assert.deepEqual(extractSectionUnits(makeSection(1, null)), [])
  assert.deepEqual(
    extractBookSections('book', 'version-a', [makeSection(1, null)]),
    [
      {
        bookId: 'book',
        bookVersionId: 'version-a',
        sectionIndex: 1,
        href: 'text/chapter-1.xhtml',
        title: undefined,
        paragraphs: [],
      },
    ],
  )
})

test('preserves direct container text around nested block elements', () => {
  const body = new FixtureElement('body').append(
    new FixtureElement('div').append(
      'Opening text. ',
      new FixtureElement('p').append('Nested paragraph.'),
      ' Closing text.',
    ),
  )

  const units = extractSectionUnits(makeSection(0, body))

  assert.deepEqual(
    units.map(({ text, kind }) => [normalizeParagraphText(text), kind]),
    [
      ['Opening text.', 'other'],
      ['Nested paragraph.', 'paragraph'],
      ['Closing text.', 'other'],
    ],
  )
})

test('normalizes layout whitespace without changing words or punctuation', () => {
  assert.equal(
    normalizeParagraphText('  A\tshort\n line — with punctuation!  '),
    'A short line — with punctuation!',
  )
  assert.equal(normalizeParagraphText('word\u00a0word'), 'word word')
  assert.equal(normalizeParagraphText('inter' + 'national'), 'international')
})

test('preserves whitespace in preformatted text units', () => {
  const source = '  const value = 1;\n    return value\n'
  const paragraph = toBookParagraph(
    { bookId: 'book', bookVersionId: 'revision' },
    {
      sectionIndex: 0,
      paragraphIndex: 0,
      text: source,
      kind: 'preformatted',
    },
  )

  assert.equal(paragraph.text, source)
})

test('paragraph IDs are deterministic and scoped to book content identity', () => {
  const unit = { sectionIndex: 3, paragraphIndex: 8 }
  const identity = { bookId: 'book / 1', bookVersionId: 'revision-a' }
  const id = createParagraphId(identity, unit)

  assert.equal(id, createParagraphId(identity, unit))
  assert.notEqual(
    id,
    createParagraphId({ ...identity, bookId: 'book-2' }, unit),
  )
  assert.notEqual(
    id,
    createParagraphId({ ...identity, bookVersionId: 'revision-b' }, unit),
  )
})

test('preserves metadata and sorts units and sections by source order', () => {
  const paragraphs = processExtractedUnits(
    { bookId: 'book', bookVersionId: 'revision' },
    [
      {
        sectionIndex: 4,
        paragraphIndex: 2,
        text: 'Later section.',
        kind: 'paragraph',
        href: 'chapter-4.xhtml',
        cfi: 'epubcfi(later)',
        heading: 'Later chapter',
      },
      {
        sectionIndex: 1,
        paragraphIndex: 1,
        text: 'Second.',
        kind: 'list-item',
        href: 'chapter-1.xhtml',
        cfi: 'epubcfi(second)',
        heading: 'First chapter',
      },
      {
        sectionIndex: 1,
        paragraphIndex: 0,
        text: 'First.',
        kind: 'heading',
        href: 'chapter-1.xhtml',
      },
    ],
  )

  assert.deepEqual(
    paragraphs.map(({ sectionIndex, paragraphIndex, text }) => [
      sectionIndex,
      paragraphIndex,
      text,
    ]),
    [
      [1, 0, 'First.'],
      [1, 1, 'Second.'],
      [4, 2, 'Later section.'],
    ],
  )
  assert.equal(paragraphs[1].kind, 'list-item')
  assert.equal(paragraphs[1].href, 'chapter-1.xhtml')
  assert.equal(paragraphs[1].cfi, 'epubcfi(second)')
  assert.equal(paragraphs[1].heading, 'First chapter')

  const sections = extractBookSections('book', 'revision', [
    makeSection(
      4,
      new FixtureElement('body').append(new FixtureElement('p').append('Four')),
    ),
    makeSection(
      1,
      new FixtureElement('body').append(new FixtureElement('p').append('One')),
    ),
  ])
  assert.deepEqual(
    sections.map((section) => section.sectionIndex),
    [1, 4],
  )
})

test('missing or failed CFI generation does not discard extracted text', () => {
  const body = new FixtureElement('body').append(
    new FixtureElement('p').append('Keep this content.'),
  )
  const missingCfi = makeSection(0, body)
  delete missingCfi.cfiFromRange

  const failedCfi = makeSection(1, body, {
    cfiFromRange() {
      throw new Error('unrepresentable range')
    },
  })

  assert.equal(extractSectionUnits(missingCfi)[0].text, 'Keep this content.')
  assert.equal(extractSectionUnits(missingCfi)[0].cfi, undefined)
  assert.equal(extractSectionUnits(failedCfi)[0].text, 'Keep this content.')
  assert.equal(extractSectionUnits(failedCfi)[0].cfi, undefined)
})

test('chunking handles empty input and ignores empty text units', () => {
  assert.deepEqual(chunkParagraphs([]), [])
  assert.deepEqual(chunkParagraphs([makeParagraph({ text: '' })]), [])
  assert.equal(
    chunkParagraphs(
      [
        makeParagraph({ text: '' }),
        makeParagraph({ paragraphIndex: 1, text: 'Keep.' }),
      ],
      { maxChunkSize: 20, overlap: 0 },
    )[0].text,
    'Keep.',
  )
  assert.equal(
    chunkParagraphs([makeParagraph({ text: '  \n', kind: 'preformatted' })], {
      maxChunkSize: 10,
    })[0].text,
    '  \n',
  )
})

test('combines units within the budget and splits at the character limit', () => {
  const units = [
    makeParagraph({ paragraphIndex: 0, text: 'One.' }),
    makeParagraph({ paragraphIndex: 1, text: 'Two.' }),
    makeParagraph({ paragraphIndex: 2, text: 'Three.' }),
  ]
  const chunks = chunkParagraphs(units, { maxChunkSize: 12, overlap: 0 })

  assert.deepEqual(
    chunks.map(({ text }) => text),
    ['One.\n\nTwo.', 'Three.'],
  )
  assert.ok(chunks.every(({ text }) => text.length <= 12))
})

test('chunk IDs are deterministic and include book and content-version identity', () => {
  const units = [makeParagraph({ paragraphIndex: 2, text: 'Stable.' })]
  const options = { maxChunkSize: 30, overlap: 0 }
  const chunk = chunkParagraphs(units, options)[0]

  assert.equal(chunk.id, chunkParagraphs(units, options)[0].id)
  assert.notEqual(
    chunk.id,
    chunkParagraphs(
      [makeParagraph({ ...units[0], bookId: 'other-book' })],
      options,
    )[0].id,
  )
  assert.notEqual(
    chunk.id,
    chunkParagraphs(
      [makeParagraph({ ...units[0], bookVersionId: 'revision-b' })],
      options,
    )[0].id,
  )
  assert.notEqual(
    chunk.id,
    chunkParagraphs(units, { maxChunkSize: 31, overlap: 0 })[0].id,
  )
})

test('preserves source order and provenance without mutating source metadata', () => {
  const units = [
    makeParagraph({
      id: 'later',
      paragraphIndex: 3,
      text: 'Later.',
      cfi: 'epubcfi(/6/4!/4/2)',
      heading: 'Chapter',
    }),
    makeParagraph({
      id: 'earlier',
      paragraphIndex: 1,
      text: 'Earlier.',
      cfi: 'epubcfi(/6/4!/4/1)',
      kind: 'list-item',
    }),
  ]
  const before = structuredClone(units)
  const chunks = chunkParagraphs(units, { maxChunkSize: 50, overlap: 0 })

  assert.deepEqual(units, before)
  assert.deepEqual(chunks[0].sourceUnitIds, ['earlier', 'later'])
  assert.deepEqual(
    chunks[0].sourcePositions.map(({ paragraphIndex, kind }) => [
      paragraphIndex,
      kind,
    ]),
    [
      [1, 'list-item'],
      [3, 'paragraph'],
    ],
  )
  assert.equal(chunks[0].bookId, 'book')
  assert.equal(chunks[0].bookVersionId, 'revision-a')
  assert.equal(chunks[0].sectionIndex, 0)
  assert.equal(chunks[0].startParagraphIndex, 1)
  assert.equal(chunks[0].endParagraphIndex, 3)
  assert.equal(chunks[0].startCfi, 'epubcfi(/6/4!/4/1)')
  assert.equal(chunks[0].endCfi, 'epubcfi(/6/4!/4/2)')
  assert.equal(chunks[0].cfiStatus, 'complete')
})

test('never combines paragraphs across section, book, or content-version boundaries', () => {
  const chunks = chunkParagraphs(
    [
      makeParagraph({ id: 'section-2', sectionIndex: 2, paragraphIndex: 0 }),
      makeParagraph({ id: 'section-1', sectionIndex: 1, paragraphIndex: 1 }),
      makeParagraph({ id: 'other-book', bookId: 'other', sectionIndex: 1 }),
      makeParagraph({
        id: 'other-version',
        bookVersionId: 'revision-b',
        sectionIndex: 1,
      }),
    ],
    { maxChunkSize: 100, overlap: 0 },
  )

  assert.deepEqual(
    chunks.map(({ bookId, bookVersionId, sectionIndex, sourceUnitIds }) => [
      bookId,
      bookVersionId,
      sectionIndex,
      sourceUnitIds,
    ]),
    [
      ['book', 'revision-a', 1, ['section-1']],
      ['book', 'revision-a', 2, ['section-2']],
      ['book', 'revision-b', 1, ['other-version']],
      ['other', 'revision-a', 1, ['other-book']],
    ],
  )
})

test('overlaps whole trailing units only when configured', () => {
  const units = [
    makeParagraph({ paragraphIndex: 0, text: 'one' }),
    makeParagraph({ paragraphIndex: 1, text: 'two' }),
    makeParagraph({ paragraphIndex: 2, text: 'four' }),
  ]
  const withoutOverlap = chunkParagraphs(units, {
    maxChunkSize: 10,
    overlap: 0,
  })
  const withOverlap = chunkParagraphs(units, { maxChunkSize: 10, overlap: 6 })

  assert.deepEqual(
    withoutOverlap.map(({ sourceUnitIds }) => sourceUnitIds),
    [['p-0', 'p-1'], ['p-2']],
  )
  assert.deepEqual(
    withOverlap.map(({ sourceUnitIds }) => sourceUnitIds),
    [
      ['p-0', 'p-1'],
      ['p-1', 'p-2'],
    ],
  )
  assert.equal(withOverlap[1].text, 'two\n\nfour')
})

test('keeps oversized units intact in their own chunks', () => {
  const chunks = chunkParagraphs(
    [
      makeParagraph({ paragraphIndex: 0, text: 'small' }),
      makeParagraph({
        paragraphIndex: 1,
        text: 'This entire unit is too large.',
      }),
      makeParagraph({ paragraphIndex: 2, text: 'next' }),
    ],
    { maxChunkSize: 10, overlap: 4 },
  )

  assert.deepEqual(
    chunks.map(({ text }) => text),
    ['small', 'This entire unit is too large.', 'next'],
  )
  assert.ok(chunks.every(({ sourceUnitIds }) => sourceUnitIds.length === 1))
})

test('keeps headings with following content when practical and preserves list items', () => {
  const chunks = chunkParagraphs(
    [
      makeParagraph({ paragraphIndex: 0, text: 'Chapter 1', kind: 'heading' }),
      makeParagraph({
        paragraphIndex: 1,
        text: 'A paragraph.',
        kind: 'paragraph',
      }),
      makeParagraph({
        paragraphIndex: 2,
        text: 'A list entry.',
        kind: 'list-item',
      }),
      makeParagraph({ paragraphIndex: 3, text: 'Chapter 2', kind: 'heading' }),
      makeParagraph({
        paragraphIndex: 4,
        text: 'Next paragraph.',
        kind: 'paragraph',
      }),
    ],
    { maxChunkSize: 28, overlap: 0 },
  )

  assert.deepEqual(
    chunks.map(({ text }) => text),
    [
      'Chapter 1\n\nA paragraph.',
      'A list entry.',
      'Chapter 2\n\nNext paragraph.',
    ],
  )
  assert.equal(chunks[0].sourcePositions[0].kind, 'heading')
  assert.equal(chunks[1].sourcePositions[0].kind, 'list-item')
})

test('reports missing and malformed CFI ranges without inventing references', () => {
  const unavailable = chunkParagraphs(
    [makeParagraph({ text: 'No CFI.', cfi: undefined })],
    { maxChunkSize: 20, overlap: 0 },
  )[0]
  const partial = chunkParagraphs(
    [
      makeParagraph({ paragraphIndex: 0, cfi: 'epubcfi(/6/2!/4/1)' }),
      makeParagraph({ paragraphIndex: 1, cfi: undefined }),
    ],
    { maxChunkSize: 50, overlap: 0 },
  )[0]
  const unreliable = chunkParagraphs([makeParagraph({ cfi: 'not-a-cfi' })], {
    maxChunkSize: 20,
    overlap: 0,
  })[0]

  assert.equal(unavailable.cfiStatus, 'unavailable')
  assert.equal(unavailable.startCfi, undefined)
  assert.equal(unavailable.endCfi, undefined)
  assert.equal(partial.cfiStatus, 'partial')
  assert.equal(partial.startCfi, 'epubcfi(/6/2!/4/1)')
  assert.equal(partial.endCfi, undefined)
  assert.equal(unreliable.cfiStatus, 'unreliable')
  assert.equal(unreliable.startCfi, undefined)
  assert.equal(unreliable.endCfi, undefined)
  assert.equal(unreliable.sourcePositions[0].cfi, 'not-a-cfi')
})

test('rejects invalid chunk-size and overlap options', () => {
  const unit = [makeParagraph()]
  for (const maxChunkSize of [
    0,
    -1,
    1.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
  ]) {
    assert.throws(() => chunkParagraphs(unit, { maxChunkSize }), RangeError)
  }
  for (const overlap of [-1, 1.5, Number.NaN, 10]) {
    assert.throws(
      () => chunkParagraphs(unit, { maxChunkSize: 10, overlap }),
      RangeError,
    )
  }
})
