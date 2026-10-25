import { useBoolean } from '@literal-ui/hooks'
import clsx from 'clsx'
import { useLiveQuery } from 'dexie-react-hooks'
import Head from 'next/head'
import { useRouter } from 'next/router'
import React, { useEffect, useState } from 'react'
import {
  MdCheckBox,
  MdCheckBoxOutlineBlank,
  MdCheckCircle,
  MdOutlineFileDownload,
  MdOutlineShare,
} from 'react-icons/md'
import { useSet } from 'react-use'
import { usePrevious } from 'react-use'

import { ReaderGridView, Button, TextField, DropZone } from '../components'
import { BookRecord, CoverRecord, db } from '../db'
import { addBook, addFile, fetchBook, handleFiles } from '../file'
import {
  deleteSavedBook,
  downloadSavedEpub,
  getSavedBookLibrary,
} from '../rag/api'
import { fingerprintEpub } from '../rag/identity'
import { mergeBookProgress } from '../progress'
import {
  useDisablePinchZooming,
  useLibrary,
  useMobile,
  useRemoteBooks,
  useRemoteFiles,
  useTranslation,
} from '../hooks'
import { reader, useReaderSnapshot } from '../models'
import { lock } from '../styles'
import { dbx, pack, uploadData } from '../sync'
import { copy } from '../utils'

const placeholder = `data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><rect fill="gray" fill-opacity="0" width="1" height="1"/></svg>`

const SOURCE = 'src'

export default function Index() {
  const { focusedTab } = useReaderSnapshot()
  const router = useRouter()
  const src = new URL(window.location.href).searchParams.get(SOURCE)
  const [loading, setLoading] = useState(!!src)

  useDisablePinchZooming()

  useEffect(() => {
    let src = router.query[SOURCE]
    if (!src) return
    if (!Array.isArray(src)) src = [src]

    Promise.all(
      src.map((s) =>
        fetchBook(s).then((b) => {
          reader.addTab(b)
        }),
      ),
    ).finally(() => setLoading(false))
  }, [router.query])

  useEffect(() => {
    if ('launchQueue' in window && 'LaunchParams' in window) {
      window.launchQueue.setConsumer((params) => {
        console.log('launchQueue', params)
        if (params.files.length) {
          Promise.all(params.files.map((f) => f.getFile()))
            .then((files) => handleFiles(files))
            .then((books) => books.forEach((b) => reader.addTab(b)))
        }
      })
    }
  }, [])

  useEffect(() => {
    router.beforePopState(({ url }) => {
      if (url === '/') {
        reader.clear()
      }
      return true
    })
  }, [router])

  return (
    <>
      <Head>
        {/* https://github.com/microsoft/vscode/blob/36fdf6b697cba431beb6e391b5a8c5f3606975a1/src/vs/code/browser/workbench/workbench.html#L16 */}
        {/* Disable pinch zooming */}
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no"
        />
        <title>
          {focusedTab?.title ? `${focusedTab.title} — Lector` : 'Lector'}
        </title>
      </Head>
      <ReaderGridView />
      {loading || <Library />}
    </>
  )
}

const Library: React.FC = () => {
  const books = useLibrary()
  const covers = useLiveQuery(() => db?.covers.toArray() ?? [])
  const t = useTranslation('home')

  const { data: remoteBooks, mutate: mutateRemoteBooks } = useRemoteBooks()
  const { data: remoteFiles, mutate: mutateRemoteFiles } = useRemoteFiles()
  const previousRemoteBooks = usePrevious(remoteBooks)
  const previousRemoteFiles = usePrevious(remoteFiles)

  const [select, toggleSelect] = useBoolean(false)
  const [selectedBookIds, { add, has, toggle, reset }] = useSet<string>()

  const [loading, setLoading] = useState<string | undefined>()
  const [readyToSync, setReadyToSync] = useState(false)
  const [recoveryError, setRecoveryError] = useState('')

  const { groups } = useReaderSnapshot()

  useEffect(() => {
    if (!books) return
    let active = true
    const recover = async () => {
      try {
        const savedBooks = await getSavedBookLibrary()
        const missingFiles: string[] = []
        for (const saved of savedBooks) {
          if (!active) continue
          const localBook = await db?.books.get(saved.book_id)
          const localFile = await db?.files.get(saved.book_id)
          if (localBook)
            await db?.books.update(saved.book_id, {
              graphRagEnabled: saved.graph_rag_enabled,
            })
          if (localFile) continue
          if (!saved.file_available) {
            missingFiles.push(saved.original_filename)
            continue
          }
          const blob = await downloadSavedEpub(saved.book_id)
          const file = new File([blob], saved.original_filename, {
            type: 'application/epub+zip',
          })
          if ((await fingerprintEpub(file)) !== saved.content_version_id) {
            throw new Error(
              `Saved EPUB fingerprint mismatch for ${saved.original_filename}.`,
            )
          }
          if (localBook) {
            await addFile(saved.book_id, file)
            if (
              saved.indexing_status === 'indexed' &&
              localBook.contentVersionId === saved.content_version_id
            )
              await db?.books.update(saved.book_id, {
                ragIndexedContentVersionId: saved.content_version_id,
              })
          } else {
            const restored = await addBook(
              file,
              saved.content_version_id,
              saved.book_id,
              saved.graph_rag_enabled,
            )
            if (saved.indexing_status === 'indexed')
              await db?.books.update(restored.id, {
                ragIndexedContentVersionId: saved.content_version_id,
              })
          }
        }
        if (active)
          setRecoveryError(
            missingFiles.length
              ? `Saved EPUB missing: ${missingFiles.join(
                  ', ',
                )}. Its indexed text remains available.`
              : '',
          )
      } catch (reason) {
        if (active)
          setRecoveryError(
            reason instanceof Error
              ? reason.message
              : 'Saved-book recovery failed.',
          )
      }
    }
    void recover()
    return () => {
      active = false
    }
  }, [books?.length])

  useEffect(() => {
    if (previousRemoteFiles && remoteFiles) {
      // to remove effect dependency `books`
      db?.books.toArray().then((books) => {
        if (books.length === 0) return

        const newRemoteBooks = remoteFiles.map((f) =>
          books.find((b) => b.name === f.name),
        ) as BookRecord[]

        uploadData(newRemoteBooks)
        mutateRemoteBooks(newRemoteBooks, { revalidate: false })
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mutateRemoteBooks, remoteFiles])

  useEffect(() => {
    if (!previousRemoteBooks && remoteBooks) {
      db?.books.toArray().then((localBooks) => {
        const localById = new Map<string, BookRecord>(
          localBooks.map((book) => [book.id, book]),
        )
        const mergedBooks = remoteBooks.map((remoteBook) =>
          mergeBookProgress(localById.get(remoteBook.id), remoteBook),
        )
        return db?.books.bulkPut(mergedBooks).then(() => setReadyToSync(true))
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remoteBooks])

  useEffect(() => {
    if (!remoteFiles || !readyToSync) return

    db?.books.toArray().then(async (books) => {
      for (const remoteFile of remoteFiles) {
        const book = books.find((b) => b.name === remoteFile.name)
        if (!book) continue

        const file = await db?.files.get(book.id)
        if (file) continue

        setLoading(book.id)
        await dbx
          .filesDownload({ path: `/files/${remoteFile.name}` })
          .then((d) => {
            const blob: Blob = (d.result as any).fileBlob
            return addFile(book.id, new File([blob], book.name))
          })
        setLoading(undefined)
      }
    })
  }, [readyToSync, remoteFiles])

  useEffect(() => {
    if (!select) reset()
  }, [reset, select])

  if (groups.length) return null
  if (!books) return null

  const selectedBooks = [...selectedBookIds].map(
    (id) => books.find((b) => b.id === id)!,
  )
  const allSelected = selectedBookIds.size === books.length

  return (
    <DropZone
      className="scroll-parent h-full p-4"
      onDrop={(e) => {
        const bookId = e.dataTransfer.getData('text/plain')
        const book = books.find((b) => b.id === bookId)
        if (book) reader.addTab(book)

        handleFiles(e.dataTransfer.files)
      }}
    >
      <div className="mb-4 space-y-2.5">
        {recoveryError && (
          <p role="status" className="text-sm">
            {recoveryError}
          </p>
        )}
        <div>
          <TextField
            name={SOURCE}
            placeholder="https://link.to/remote.epub"
            type="url"
            hideLabel
            actions={[
              {
                title: t('share'),
                Icon: MdOutlineShare,
                onClick(el) {
                  if (el?.reportValidity()) {
                    copy(`${window.location.origin}/?${SOURCE}=${el.value}`)
                  }
                },
              },
              {
                title: t('download'),
                Icon: MdOutlineFileDownload,
                onClick(el) {
                  if (el?.reportValidity()) fetchBook(el.value)
                },
              },
            ]}
          />
        </div>
        <div className="flex items-center justify-between gap-4">
          <div className="space-x-2">
            {books.length ? (
              <Button variant="secondary" onClick={toggleSelect}>
                {t(select ? 'cancel' : 'select')}
              </Button>
            ) : (
              <Button
                variant="secondary"
                disabled={!books}
                onClick={() => {
                  fetchBook(
                    'https://epubtest.org/books/Fundamental-Accessibility-Tests-Basic-Functionality-v1.0.0.epub',
                  )
                }}
              >
                {t('download_sample_book')}
              </Button>
            )}
            {select &&
              (allSelected ? (
                <Button variant="secondary" onClick={reset}>
                  {t('deselect_all')}
                </Button>
              ) : (
                <Button
                  variant="secondary"
                  onClick={() => books.forEach((b) => add(b.id))}
                >
                  {t('select_all')}
                </Button>
              ))}
          </div>

          <div className="space-x-2">
            {select ? (
              <>
                <Button
                  onClick={async () => {
                    toggleSelect()

                    for (const book of selectedBooks) {
                      const remoteFile = remoteFiles?.find(
                        (f) => f.name === book.name,
                      )
                      if (remoteFile) continue

                      const file = await db?.files.get(book.id)
                      if (!file) continue

                      setLoading(book.id)
                      await dbx.filesUpload({
                        path: `/files/${book.name}`,
                        contents: file.file,
                      })
                      setLoading(undefined)

                      mutateRemoteFiles()
                    }
                  }}
                >
                  {t('upload')}
                </Button>
                <Button
                  onClick={async () => {
                    const bookIds = [...selectedBookIds]
                    try {
                      const saved = await getSavedBookLibrary()
                      for (const id of bookIds) {
                        if (saved.some((item) => item.book_id === id))
                          await deleteSavedBook(id)
                      }
                      await db?.books.bulkDelete(bookIds)
                      await db?.covers.bulkDelete(bookIds)
                      await db?.files.bulkDelete(bookIds)
                      toggleSelect()
                      setRecoveryError('')
                    } catch (reason) {
                      setRecoveryError(
                        reason instanceof Error
                          ? reason.message
                          : 'Lector could not delete the selected book data.',
                      )
                      return
                    }

                    // folder data is not updated after `filesDeleteBatch`
                    mutateRemoteFiles(
                      async (data) => {
                        await dbx.filesDeleteBatch({
                          entries: selectedBooks.map((b) => ({
                            path: `/files/${b.name}`,
                          })),
                        })
                        return data?.filter(
                          (f) => !selectedBooks.find((b) => b.name === f.name),
                        )
                      },
                      { revalidate: false },
                    )
                  }}
                >
                  {t('delete')}
                </Button>
              </>
            ) : (
              <>
                <Button
                  variant="secondary"
                  disabled={!books.length}
                  onClick={pack}
                >
                  {t('export')}
                </Button>
                <Button className="relative">
                  <input
                    type="file"
                    accept="application/epub+zip,application/epub,application/zip"
                    className="absolute inset-0 cursor-pointer opacity-0"
                    onChange={(e) => {
                      const files = e.target.files
                      if (files) handleFiles(files)
                    }}
                    multiple
                  />
                  {t('import')}
                </Button>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="scroll h-full">
        <ul
          className="grid"
          style={{
            gridTemplateColumns: `repeat(auto-fill, minmax(calc(80px + 3vw), 1fr))`,
            columnGap: lock(16, 32),
            rowGap: lock(24, 40),
          }}
        >
          {books.map((book) => (
            <Book
              key={book.id}
              book={book}
              covers={covers}
              select={select}
              selected={has(book.id)}
              loading={loading === book.id}
              toggle={toggle}
            />
          ))}
        </ul>
      </div>
    </DropZone>
  )
}

interface BookProps {
  book: BookRecord
  covers?: CoverRecord[]
  select?: boolean
  selected?: boolean
  loading?: boolean
  toggle: (id: string) => void
}
const Book: React.FC<BookProps> = ({
  book,
  covers,
  select,
  selected,
  loading,
  toggle,
}) => {
  const remoteFiles = useRemoteFiles()

  const router = useRouter()
  const mobile = useMobile()

  const cover = covers?.find((c) => c.id === book.id)?.cover
  const remoteFile = remoteFiles.data?.find((f) => f.name === book.name)

  const Icon = selected ? MdCheckBox : MdCheckBoxOutlineBlank

  return (
    <div className="relative flex flex-col">
      <div
        role="button"
        className="border-inverse-on-surface relative border"
        onClick={async () => {
          if (select) {
            toggle(book.id)
          } else {
            if (mobile) await router.push('/_')
            reader.addTab(book)
          }
        }}
      >
        <div
          className={clsx(
            'absolute bottom-0 h-1 bg-blue-500',
            loading && 'progress-bit w-[5%]',
          )}
        />
        {book.percentage !== undefined && (
          <div className="typescale-body-large absolute right-0 bg-gray-500/60 px-2 text-gray-100">
            {(book.percentage * 100).toFixed()}%
          </div>
        )}
        <img
          src={cover ?? placeholder}
          alt="Cover"
          className="mx-auto aspect-[9/12] object-cover"
          draggable={false}
        />
        {select && (
          <div className="absolute bottom-1 right-1">
            <Icon
              size={24}
              className={clsx(
                '-m-1',
                selected ? 'text-tertiary' : 'text-outline',
              )}
            />
          </div>
        )}
      </div>

      <div
        className="line-clamp-2 text-on-surface-variant typescale-body-small lg:typescale-body-medium mt-2 w-full"
        title={book.name}
      >
        <MdCheckCircle
          className={clsx(
            'mr-1 mb-0.5 inline',
            remoteFile ? 'text-tertiary' : 'text-surface-variant',
          )}
          size={16}
        />
        {book.name}
      </div>
    </div>
  )
}
