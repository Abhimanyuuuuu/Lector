const assert = require('node:assert/strict')
const { test } = require('node:test')
const {
  downloadSavedEpub,
  getSavedBookLibrary,
  saveEpubToBackend,
  getBookPreparation,
  prepareBookSourceUnits,
  deleteSavedBook,
  RagApiError,
} = require('../../.turbo/lector-rag-tests/api.js')

const book = {
  book_id: 'book-1',
  title: 'Novel',
  original_filename: 'novel.epub',
  content_version_id: `sha256:${'a'.repeat(64)}`,
  file_size: 42,
  indexing_status: 'indexed',
  preparation_indexing_status: 'complete',
  indexing_progress: 1,
  indexing_completed: 10,
  indexing_total: 10,
  embedding_status: 'complete',
  embedding_progress: 1,
  embedding_completed: 5,
  embedding_total: 5,
  graph_rag_enabled: false,
  graph_status: 'not_enabled',
  graph_progress: 0,
  graph_completed: 0,
  graph_total: 0,
  graph_error: null,
  file_available: true,
  updated_at: '2026-10-17 00:00:00',
}

test('saved library response is validated and EPUB can be downloaded', async () => {
  const calls = []
  const library = await getSavedBookLibrary(async (url) => {
    calls.push(url)
    return { ok: true, status: 200, json: async () => ({ books: [book] }) }
  })
  assert.deepEqual(library, [book])
  assert.match(calls[0], /\/api\/v1\/books$/)
  const blob = new Blob(['epub bytes'])
  const downloaded = await downloadSavedEpub('book-1', async (url) => {
    assert.match(url, /books\/book-1\/epub$/)
    return { ok: true, status: 200, blob: async () => blob }
  })
  assert.equal(await downloaded.text(), 'epub bytes')
})

test('preparation client validates persisted job state and schedules source units', async () => {
  const state = {
    book_id: 'book-1',
    content_version_id: book.content_version_id,
    indexing_status: 'complete',
    embedding_status: 'processing',
    indexing_completed: 10,
    indexing_total: 10,
    embedding_completed: 2,
    embedding_total: 5,
    indexing_error: null,
    embedding_error: null,
    graph_rag_enabled: false,
    graph_status: 'not_enabled',
    graph_completed: 0,
    graph_total: 0,
    graph_error: null,
    updated_at: '2026-10-17',
  }
  const current = await getBookPreparation(
    'book-1',
    book.content_version_id,
    async (url) => {
      assert.match(url, /preparation\?content_version_id=/)
      return { ok: true, status: 200, json: async () => state }
    },
  )
  assert.deepEqual(current, state)
  let call
  const scheduled = await prepareBookSourceUnits(
    'book-1',
    book.content_version_id,
    [
      {
        id: 'u1',
        bookId: 'book-1',
        contentVersionId: book.content_version_id,
        sectionIndex: 0,
        paragraphIndex: 0,
        kind: 'paragraph',
        text: 'Text',
        href: 'c.xhtml',
      },
    ],
    async (url, init) => {
      call = { url, init }
      return { ok: true, status: 202, json: async () => state }
    },
    true,
  )
  assert.equal(scheduled.embedding_status, 'processing')
  assert.match(call.url, /books\/book-1\/prepare$/)
  assert.equal(JSON.parse(call.init.body).source_units[0].source_unit_id, 'u1')
  assert.equal(JSON.parse(call.init.body).graph_rag_enabled, true)
})

test('explicit book deletion validates the backend confirmation', async () => {
  await deleteSavedBook('book-1', async (url, init) => {
    assert.match(url, /\/api\/v1\/books\/book-1$/)
    assert.equal(init.method, 'DELETE')
    return {
      ok: true,
      status: 200,
      json: async () => ({ book_id: 'book-1', deleted: true }),
    }
  })
  await assert.rejects(
    deleteSavedBook('book-1', async () => ({
      ok: true,
      status: 200,
      json: async () => ({ book_id: 'other', deleted: true }),
    })),
    (error) =>
      error instanceof RagApiError && error.code === 'invalid_response',
  )
})

test('EPUB upload sends file bytes and version-scoped metadata', async () => {
  const file = new File(['epub bytes'], 'novel.epub', {
    type: 'application/epub+zip',
  })
  let call
  await saveEpubToBackend(
    {
      bookId: 'book-1',
      contentVersionId: book.content_version_id,
      title: 'Novel',
      originalFilename: 'novel.epub',
      file,
    },
    async (url, init) => {
      call = { url, init }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          book_id: 'book-1',
          content_version_id: book.content_version_id,
        }),
      }
    },
  )
  assert.match(call.url, /content_version_id=sha256%3A/)
  assert.equal(call.init.method, 'PUT')
  assert.equal(call.init.body, file)
  assert.equal(call.init.headers['Content-Type'], 'application/epub+zip')
})

test('library rejects invalid and unavailable backend responses clearly', async () => {
  await assert.rejects(
    getSavedBookLibrary(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ books: [{}] }),
    })),
    (error) =>
      error instanceof RagApiError && error.code === 'invalid_response',
  )
  await assert.rejects(
    getSavedBookLibrary(async () => ({
      ok: false,
      status: 503,
      json: async () => ({
        error: { code: 'offline', message: 'Unavailable' },
      }),
    })),
    (error) => error instanceof RagApiError && error.code === 'offline',
  )
})
