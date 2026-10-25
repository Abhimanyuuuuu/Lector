/**
 * Merge a remote book snapshot without letting stale progress replace the
 * reader's locally persisted EPUB location. `updatedAt` is the best revision
 * marker available in the existing book record; on ties, local progress wins.
 */
export interface BookProgressSnapshot {
  id: string
  cfi?: string
  percentage?: number
  updatedAt?: number
}

function hasProgress(book: BookProgressSnapshot): boolean {
  return typeof book.cfi === 'string' || typeof book.percentage === 'number'
}

export function mergeBookProgress<T extends BookProgressSnapshot>(
  local: T | undefined,
  remote: T,
): T
export function mergeBookProgress<T extends BookProgressSnapshot>(
  local: T | undefined,
  remote: undefined,
): T | undefined
export function mergeBookProgress<T extends BookProgressSnapshot>(
  local: T | undefined,
  remote: T | undefined,
): T | undefined {
  if (!remote) return local
  if (!local || local.id !== remote.id) return remote

  const localHasProgress = hasProgress(local)
  const remoteHasProgress = hasProgress(remote)
  let progressSource: T

  if (!localHasProgress) progressSource = remote
  else if (!remoteHasProgress) progressSource = local
  else if (
    typeof remote.updatedAt === 'number' &&
    typeof local.updatedAt === 'number' &&
    remote.updatedAt > local.updatedAt
  ) {
    progressSource = remote
  } else {
    // Equal or unavailable timestamps favor local IndexedDB state.
    progressSource = local
  }

  return {
    ...local,
    ...remote,
    cfi: progressSource.cfi,
    percentage: progressSource.percentage,
    updatedAt:
      local.updatedAt === undefined
        ? remote.updatedAt
        : remote.updatedAt === undefined
        ? local.updatedAt
        : Math.max(local.updatedAt, remote.updatedAt),
  }
}
