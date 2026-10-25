import assert from 'assert'

import ePub from '../../../../packages/epubjs/src/epub'
import EpubCFI from '../../../../packages/epubjs/src/epubcfi'

const {
  extractSectionUnits,
} = require('../../.turbo/lector-rag-tests/extract.js')
const {
  normalizeParagraphText,
} = require('../../.turbo/lector-rag-tests/paragraph.js')

describe('RAG source mapping with epub.js', function () {
  it('creates a CFI range that resolves to the extracted heading', async function () {
    const book = ePub('/fixtures/alice/OPS/package.opf')

    try {
      await book.ready
      const section = book.section('chapter_001.xhtml')
      assert.ok(section, 'fixture chapter is present in the EPUB spine')
      await section.load(book.load.bind(book))

      const units = extractSectionUnits({
        index: section.index,
        href: section.href,
        document: section.document,
        cfiFromRange: section.cfiFromRange.bind(section),
      })
      const heading = units.find((unit) => unit.kind === 'heading')

      assert.ok(heading, 'the chapter heading is extracted')
      assert.equal(heading.text.trim(), 'Down The Rabbit-Hole')
      assert.ok(heading.cfi, 'epub.js generates a source CFI range')
      assert.ok(
        units.indexOf(heading) <
          units.findIndex((unit) => unit.text.includes('Alice was beginning')),
        'the real section heading precedes its paragraph text',
      )

      const parsedCfi = new EpubCFI(heading.cfi)
      assert.equal(parsedCfi.range, true, 'the CFI encodes a range')
      assert.equal(
        parsedCfi.spinePos,
        section.index,
        'the CFI identifies the source spine item',
      )

      const resolvedRange = await book.getRange(heading.cfi)
      assert.ok(
        resolvedRange,
        'epub.js resolves the CFI against the loaded book',
      )
      const resolvedText = normalizeParagraphText(resolvedRange.toString())
      assert.equal(
        resolvedText,
        normalizeParagraphText(heading.text),
        `the resolved range selects the original heading text (CFI: ${heading.cfi}; resolved: ${resolvedText})`,
      )
    } finally {
      book.destroy()
    }
  })
})
