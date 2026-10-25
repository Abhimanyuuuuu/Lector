import { v4 as uuidv4 } from 'uuid'

import ePub, { Book } from '@flow/epubjs'

import { BookRecord, db } from './db'
import { mapExtToMimes } from './mime'
import { unpack } from './sync'
import { fingerprintEpub } from './rag/identity'

export async function fileToEpub(file: File) {
  const data = await file.arrayBuffer()
  return ePub(data)
}

export async function handleFiles(files: Iterable<File>) {
  const books = (await db?.books.toArray()) ?? []
  const newBooks = []

  for (const file of files) {
    console.log(file)

    if (mapExtToMimes['.zip'].includes(file.type)) {
      unpack(file)
      continue
    }

    if (!mapExtToMimes['.epub'].includes(file.type)) {
      console.error(`Unsupported file type: ${file.type}`)
      continue
    }

    const versionId = await fingerprintEpub(file)
    let book: BookRecord | undefined
    for (const candidate of books.filter((item) => item.name === file.name)) {
      let existingVersion = candidate.contentVersionId
      if (!existingVersion) {
        const savedFile = await db?.files.get(candidate.id)
        if (savedFile) {
          existingVersion = await fingerprintEpub(savedFile.file)
          await db?.books.update(candidate.id, {
            contentVersionId: existingVersion,
          })
        }
      }
      if (existingVersion === versionId) {
        book = candidate
        break
      }
    }

    if (!book) {
      const graphRagEnabled = askGraphRagPreference(file.name)
      book = await addBook(file, versionId, undefined, graphRagEnabled)
      books.push(book)
    }

    newBooks.push(book)
  }

  return newBooks
}

export async function addBook(
  file: File,
  versionId?: string,
  stableId?: string,
  graphRagEnabled?: boolean,
) {
  const epub = await fileToEpub(file)
  const metadata = await epub.loaded.metadata
  versionId ??= await fingerprintEpub(file)

  const book: BookRecord = {
    id: stableId ?? uuidv4(),
    contentVersionId: versionId,
    graphRagEnabled,
    name: file.name || `${metadata.title}.epub`,
    size: file.size,
    metadata,
    createdAt: Date.now(),
    definitions: [],
    annotations: [],
  }
  await db?.books.put(book)
  await addFile(book.id, file, epub)
  return book
}

export async function addFile(id: string, file: File, epub?: Book) {
  await db?.files.put({ id, file })

  if (!epub) {
    epub = await fileToEpub(file)
  }

  const url = await epub.coverUrl()
  const cover = url && (await toDataUrl(url))
  await db?.covers.put({ id, cover })
}

export function readBlob(fn: (reader: FileReader) => void) {
  return new Promise<string>((resolve) => {
    const reader = new FileReader()
    reader.addEventListener('load', () => {
      resolve(reader.result as string)
    })
    fn(reader)
  })
}

async function toDataUrl(url: string) {
  const res = await fetch(url)
  const buffer = await res.blob()
  return readBlob((r) => r.readAsDataURL(buffer))
}

export async function fetchBook(url: string) {
  const filename = decodeURIComponent(/\/([^/]*\.epub)$/i.exec(url)?.[1] ?? '')
  const books = await db?.books.toArray()
  const book = books?.find((b) => b.name === filename)

  return (
    book ??
    fetch(url)
      .then((res) => res.blob())
      .then((blob) =>
        addBook(
          new File([blob], filename),
          undefined,
          undefined,
          askGraphRagPreference(filename),
        ),
      )
  )
}

function askGraphRagPreference(filename: string): boolean {
  if (typeof window === 'undefined') return false
  return window.confirm(
    `Enable Graph RAG for “${filename}”?\n\nChoose OK to build the story graph locally with Lector, or Cancel to prepare this book without Graph RAG.`,
  )
}
