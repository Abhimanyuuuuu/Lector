const assert = require('node:assert/strict')
const {
  mergeBookProgress,
} = require('../../.turbo/lector-rag-tests/progress.js')

const local = {
  id: 'book-a',
  name: 'Novel.epub',
  cfi: 'epubcfi(/6/12!/4/2)',
  percentage: 0.42,
  updatedAt: 200,
}

// Backend availability is independent of browser progress: without a remote
// snapshot, the locally saved position remains the source of truth.
assert.equal(mergeBookProgress(local, undefined), local)

// A stale Dropbox snapshot at the opening page cannot clobber newer local
// IndexedDB progress during app bootstrap after a backend reconnect/restart.
const staleRemote = {
  ...local,
  cfi: 'epubcfi(/6/2!/4/2)',
  percentage: 0,
  updatedAt: 100,
}
const restored = mergeBookProgress(local, staleRemote)
assert.equal(restored.cfi, local.cfi)
assert.equal(restored.percentage, local.percentage)

// The newer side can still restore progress from Dropbox on another device.
const newerRemote = { ...staleRemote, updatedAt: 300 }
assert.equal(mergeBookProgress(local, newerRemote).cfi, newerRemote.cfi)

// A snapshot without a location must not erase a known position.
assert.equal(
  mergeBookProgress(local, { ...local, cfi: undefined, percentage: undefined })
    .cfi,
  local.cfi,
)

// Progress is never copied across different book identities.
const otherBook = { ...local, id: 'book-b', cfi: 'remote-cfi', percentage: 0.1 }
assert.equal(mergeBookProgress(local, otherBook), otherBook)

console.log('progress restore tests passed')
