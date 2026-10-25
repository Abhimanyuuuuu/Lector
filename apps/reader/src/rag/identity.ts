/** Hash the exact EPUB bytes to scope all source IDs and citations to a revision. */
export async function fingerprintEpub(
  file: Blob,
  cryptoApi: Crypto = crypto,
): Promise<string> {
  if (!cryptoApi?.subtle) {
    throw new Error('This browser does not support secure EPUB fingerprinting.')
  }
  const digest = await cryptoApi.subtle.digest(
    'SHA-256',
    await file.arrayBuffer(),
  )
  const hex = Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, '0'),
  ).join('')
  return `sha256:${hex}`
}
