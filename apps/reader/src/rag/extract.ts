import { processExtractedUnits } from './paragraph'
import {
  BookSection,
  ExtractedTextUnit,
  LoadedEpubSection,
  ParagraphKind,
} from './types'

const BLOCK_SELECTOR =
  'p, h1, h2, h3, h4, h5, h6, li, blockquote, pre, dt, dd, div'
const IGNORED_TAGS = new Set([
  'SCRIPT',
  'STYLE',
  'NOSCRIPT',
  'TEMPLATE',
  'SVG',
  'CANVAS',
  'AUDIO',
  'VIDEO',
])
const CONTAINER_TAGS = new Set(['div', 'li', 'blockquote'])

function kindFor(element: Element): ParagraphKind {
  const tag = element.tagName.toLowerCase()
  if (/^h[1-6]$/.test(tag)) return 'heading'
  if (tag === 'li') return 'list-item'
  if (tag === 'blockquote') return 'blockquote'
  if (tag === 'pre') return 'preformatted'
  if (tag === 'p') {
    return element.closest('li') ? 'list-item' : 'paragraph'
  }
  return 'other'
}

function isHeading(element: Element) {
  return /^h[1-6]$/i.test(element.tagName)
}

function hasBlockDescendant(element: Element) {
  return !!element.querySelector(BLOCK_SELECTOR)
}

function readContent(node: Node) {
  const nodes: Text[] = []
  const pieces: string[] = []
  const visit = (current: Node) => {
    if (current.nodeType === 1) {
      const element = current as Element
      if (IGNORED_TAGS.has(element.tagName)) return
      if (element.tagName === 'BR') {
        pieces.push(' ')
        return
      }
    }
    if (current.nodeType === 3) {
      nodes.push(current as Text)
      pieces.push(current.nodeValue ?? '')
      return
    }
    current.childNodes.forEach(visit)
  }
  visit(node)
  return { nodes, text: pieces.join('') }
}

function sourceCfi(section: LoadedEpubSection, textNodes: Text[]) {
  if (!section.cfiFromRange || !section.document) return undefined

  try {
    const firstTextNode = textNodes[0]
    const lastTextNode = textNodes[textNodes.length - 1]
    if (!firstTextNode || !lastTextNode) return undefined

    const range = section.document.createRange()
    range.setStart(firstTextNode, 0)
    range.setEnd(lastTextNode, lastTextNode.nodeValue?.length ?? 0)
    return section.cfiFromRange(range) || undefined
  } catch {
    // Some malformed EPUB DOMs cannot be represented as a CFI range.
    return undefined
  }
}

/** Extract readable, ordered text units from an already-loaded epub.js section. */
export function extractSectionUnits(
  section: LoadedEpubSection,
): ExtractedTextUnit[] {
  const document = section.document
  const root = document?.body ?? document?.documentElement
  if (!root) return []

  const candidates = new Set(Array.from(root.querySelectorAll(BLOCK_SELECTOR)))

  const units: ExtractedTextUnit[] = []
  let heading = section.navitem?.label?.trim() || undefined

  const emit = (text: string, textNodes: Text[], kind: ParagraphKind) => {
    if (!text.trim()) return
    units.push({
      sectionIndex: section.index,
      paragraphIndex: units.length,
      text,
      kind,
      href: section.href,
      cfi: sourceCfi(section, textNodes),
      heading,
    })
  }

  const emitElement = (element: Element) => {
    const content = readContent(element)
    if (!content.text.trim()) return
    if (isHeading(element)) heading = content.text
    emit(content.text, content.nodes, kindFor(element))
  }

  const visitChildren = (
    parent: Element,
    fallbackKind: ParagraphKind,
    pending: { nodes: Text[]; pieces: string[] },
    flushAtEnd: boolean,
  ) => {
    const flush = () => {
      emit(pending.pieces.join(''), pending.nodes, fallbackKind)
      pending.nodes = []
      pending.pieces = []
    }

    for (const child of Array.from(parent.childNodes)) {
      if (child.nodeType === 3) {
        pending.nodes.push(child as Text)
        pending.pieces.push(child.nodeValue ?? '')
        continue
      }
      if (child.nodeType !== 1) continue

      const element = child as Element
      if (IGNORED_TAGS.has(element.tagName)) continue
      if (element.tagName === 'BR') {
        pending.pieces.push(' ')
        continue
      }
      if (candidates.has(element)) {
        flush()
        const tag = element.tagName.toLowerCase()
        if (CONTAINER_TAGS.has(tag) && hasBlockDescendant(element)) {
          visitChildren(
            element,
            kindFor(element),
            { nodes: [], pieces: [] },
            true,
          )
        } else {
          emitElement(element)
        }
        continue
      }

      if (hasBlockDescendant(element)) {
        visitChildren(element, fallbackKind, pending, false)
      } else {
        const content = readContent(element)
        pending.nodes.push(...content.nodes)
        pending.pieces.push(content.text)
      }
    }

    if (flushAtEnd) flush()
  }

  visitChildren(root, 'other', { nodes: [], pieces: [] }, true)

  return units
}

/** Convert the loaded Flow spine sections into source-mapped paragraph records. */
export function extractBookSections(
  bookId: string,
  bookVersionId: string,
  sections: LoadedEpubSection[],
): BookSection[] {
  return sections
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((section) => ({
      bookId,
      bookVersionId,
      sectionIndex: section.index,
      href: section.href,
      title: section.navitem?.label,
      paragraphs: processExtractedUnits(
        { bookId, bookVersionId },
        extractSectionUnits(section),
      ),
    }))
}
