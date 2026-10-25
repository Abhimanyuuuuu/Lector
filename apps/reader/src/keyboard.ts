const READER_NAVIGATION_KEYS = new Set([
  'ArrowLeft',
  'ArrowUp',
  'ArrowRight',
  'ArrowDown',
])

const TEXT_ENTRY_SELECTOR =
  'input, textarea, select, [role="textbox"], [contenteditable]:not([contenteditable="false"])'

export function shouldNavigateReaderWithKey(
  code: string,
  target: EventTarget | null,
) {
  if (!READER_NAVIGATION_KEYS.has(code)) return false

  const element = target as HTMLElement | null
  if (!element || typeof element.closest !== 'function') return false
  if (element.isContentEditable) return false

  return !element.closest(TEXT_ENTRY_SELECTOR)
}
