const assert = require('node:assert/strict')
const { test } = require('node:test')
const {
  extractStoryGraph,
  retryFullBookGraph,
  getStoryGraphStatus,
  getCharacterGraphView,
  resolveGraphContentVersionId,
  RagApiError,
} = require('../../.turbo/lector-rag-tests/api.js')

const boundary = {
  confirmed: true,
  contentVersionId: 'v1',
  sectionIndex: 2,
  unitIndex: 4,
  sourceUnitId: 'unit-4',
}

function job(status = 'queued') {
  return {
    job_id: 'job-1',
    book_id: 'book',
    content_version_id: 'v1',
    boundary_section: 2,
    boundary_unit: 4,
    boundary_source_unit_id: 'unit-4',
    total_chunks: 9,
    processed_chunks: 3,
    status,
    attempts: 0,
    error: null,
    provider: 'ollama',
    provider_version: 'ollama:fixture',
    job_scope: 'confirmed_boundary',
    updated_at: '2026-10-17 00:00:00',
  }
}

test('graph viewer reuses its saved content version when the local EPUB is unavailable', async () => {
  let fingerprintCalls = 0
  const version = await resolveGraphContentVersionId(
    'sha256:saved-version',
    async () => {
      fingerprintCalls += 1
      throw new Error('The local EPUB file is unavailable.')
    },
  )
  assert.equal(version, 'sha256:saved-version')
  assert.equal(fingerprintCalls, 0)
})

test('graph viewer fingerprints the EPUB for legacy records without a saved version', async () => {
  let fingerprintCalls = 0
  const version = await resolveGraphContentVersionId(undefined, async () => {
    fingerprintCalls += 1
    return 'sha256:computed-version'
  })
  assert.equal(version, 'sha256:computed-version')
  assert.equal(fingerprintCalls, 1)
})

test('character graph view sends the saved version and confirmed boundary and accepts an empty graph', async () => {
  let call
  const response = {
    book_id: 'book',
    content_version_id: 'v1',
    view_mode: 'till_now',
    boundary: { section_index: 2, unit_index: 4, source_unit_id: 'unit-4' },
    nodes: [],
    edges: [],
  }
  const result = await getCharacterGraphView(
    'book',
    'v1',
    boundary,
    async (url, init) => {
      call = { url, init }
      return { ok: true, status: 200, json: async () => response }
    },
  )
  assert.match(call.url, /\/books\/book\/graph\/view$/)
  assert.deepEqual(JSON.parse(call.init.body), {
    content_version_id: 'v1',
    view_mode: 'till_now',
    section_index: 2,
    unit_index: 4,
    source_unit_id: 'unit-4',
    confirmed: true,
  })
  assert.deepEqual(result.nodes, [])
  assert.deepEqual(result.edges, [])
})

test('full-book graph view does not require a boundary and identifies spoiler-bearing data', async () => {
  let call
  const response = {
    book_id: 'book',
    content_version_id: 'v1',
    view_mode: 'full_book',
    boundary: null,
    nodes: [],
    edges: [],
  }
  const result = await getCharacterGraphView(
    'book',
    'v1',
    undefined,
    async (url, init) => {
      call = { url, init }
      return { ok: true, status: 200, json: async () => response }
    },
    'full_book',
  )
  assert.deepEqual(JSON.parse(call.init.body), {
    content_version_id: 'v1',
    view_mode: 'full_book',
  })
  assert.equal(result.view_mode, 'full_book')
  assert.equal(result.boundary, null)
})

test('graph extraction schedules a durable job and retry uses the retry route', async () => {
  const calls = []
  const fetchImpl = async (url, init) => {
    calls.push({ url, init })
    return { ok: true, status: 202, json: async () => job() }
  }
  const result = await extractStoryGraph('book', boundary, fetchImpl)
  assert.equal(result.job_id, 'job-1')
  assert.match(calls[0].url, /\/books\/book\/graph\/extract$/)
  assert.equal(JSON.parse(calls[0].init.body).source_unit_id, 'unit-4')
  await extractStoryGraph('book', boundary, fetchImpl, true)
  assert.match(calls[1].url, /\/graph\/extract\/retry$/)
  assert.equal(JSON.parse(calls[1].init.body).provider, undefined)
})

test('full-book graph retry is scoped to the content version and local job', async () => {
  let call
  await retryFullBookGraph('book', 'v1', async (url, init) => {
    call = { url, init }
    return {
      ok: true,
      status: 202,
      json: async () => ({ ...job(), job_scope: 'full_book' }),
    }
  })
  assert.match(call.url, /preparation\/graph\/retry\?content_version_id=v1$/)
  assert.equal(JSON.parse(call.init.body).provider, undefined)
})

test('status validates book/version and returns durable progress', async () => {
  let requestedUrl = ''
  const result = await getStoryGraphStatus('book', 'v1', async (url) => {
    requestedUrl = url
    return { ok: true, status: 200, json: async () => job('processing') }
  })
  assert.match(requestedUrl, /content_version_id=v1$/)
  assert.equal(result.processed_chunks, 3)
  assert.equal(result.status, 'processing')
})

test('status rejects mismatched content versions and reports backend errors', async () => {
  await assert.rejects(
    getStoryGraphStatus('book', 'v1', async () => ({
      ok: true,
      status: 200,
      json: async () => ({ ...job(), content_version_id: 'old-version' }),
    })),
    (error) =>
      error instanceof RagApiError && error.code === 'invalid_response',
  )
  await assert.rejects(
    getStoryGraphStatus('book', 'v1', async () => ({
      ok: false,
      status: 503,
      json: async () => ({
        error: {
          code: 'story_extraction_unavailable',
          message: 'No local extractor.',
        },
      }),
    })),
    (error) =>
      error instanceof RagApiError &&
      error.code === 'story_extraction_unavailable',
  )
})
