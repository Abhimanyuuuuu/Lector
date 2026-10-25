import React, { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { MdArrowUpward } from 'react-icons/md'
import { useSnapshot } from 'valtio'

import { useTranslation } from '@flow/reader/hooks'
import { reader, useReaderSnapshot } from '@flow/reader/models'
import { BookTab } from '@flow/reader/models/reader'
import { db } from '@flow/reader/db'
import { Select } from '../Form'
import { PaneView, PaneViewProps } from '../base'
import {
  askBook,
  prepareBookSourceUnits,
  getBookPreparation,
  getSavedBookLibrary,
  retryFullBookGraph,
  saveEpubToBackend,
  RagApiError,
} from '../../rag/api'
import type {
  ConfirmedBoundary,
  GroundedAnswer,
  RetrievalMode,
  BookPreparation,
} from '../../rag/api'
import {
  createBoundaryForLocation,
  extractLoadedBook,
  navigateToCitation,
} from '../../rag/flow'
import { RagPreparationProgress } from './RagPreparationProgress'

type WorkState = 'idle' | 'extracting' | 'uploading' | 'indexed' | 'error'

export const RagView: React.FC<PaneViewProps> = (props) => {
  const { focusedBookTab } = useReaderSnapshot()
  const t = useTranslation('lector')
  if (!focusedBookTab) {
    return (
      <PaneView {...props}>
        <p className="px-5">{t('open_book')}</p>
      </PaneView>
    )
  }
  return (
    <BookRagView
      key={focusedBookTab.id}
      tab={reader.focusedBookTab!}
      {...props}
    />
  )
}

const BookRagView: React.FC<PaneViewProps & { tab: BookTab }> = ({
  tab,
  ...props
}) => {
  const snapshot = useSnapshot(tab)
  const t = useTranslation('lector')
  const [versionId, setVersionId] = useState<string>()
  const [work, setWork] = useState<WorkState>('idle')
  const [progress, setProgress] = useState('')
  const [error, setError] = useState('')
  const [boundaryError, setBoundaryError] = useState('')
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('keyword')
  const [asking, setAsking] = useState(false)
  const [answer, setAnswer] = useState<GroundedAnswer>()
  const [preparation, setPreparation] = useState<BookPreparation>()
  const [boundaryConfirmed, setBoundaryConfirmed] = useState(false)
  const [savingBoundary, setSavingBoundary] = useState(false)
  const isMounted = useRef(true)
  const boundarySaveLock = useRef(false)
  const confirmationTimer = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      if (confirmationTimer.current) clearTimeout(confirmationTimer.current)
    }
  }, [])

  useEffect(() => {
    let active = true
    tab
      .getContentVersionId()
      .then(async (id) => {
        if (!active) return
        setVersionId(id)
        if (tab.book.ragIndexedContentVersionId !== id) return
        const saved = await getSavedBookLibrary()
        if (
          saved.some(
            (item) =>
              item.book_id === tab.book.id &&
              item.content_version_id === id &&
              item.file_available,
          )
        )
          return
        const localFile = await db?.files.get(tab.book.id)
        if (!localFile)
          throw new Error(
            'The indexed EPUB file is unavailable locally and in the backend library.',
          )
        setWork('uploading')
        setProgress('Saving the EPUB locally…')
        await saveEpubToBackend({
          bookId: tab.book.id,
          contentVersionId: id,
          title: String(tab.book.metadata.title || tab.book.name),
          originalFilename: tab.book.name,
          file: localFile.file,
        })
        if (active) {
          setProgress('Saved EPUB is available for recovery.')
          setWork('indexed')
        }
      })
      .catch((reason) => {
        if (active) {
          setWork('error')
          setError(errorMessage(reason))
        }
      })
    return () => {
      active = false
    }
  }, [tab, snapshot.book.contentVersionId])

  useEffect(() => {
    if (!versionId) return
    let active = true
    const refresh = () =>
      getBookPreparation(tab.book.id, versionId)
        .then(async (status) => {
          if (!active) return
          setPreparation(status)
          if (tab.book.graphRagEnabled !== status.graph_rag_enabled)
            await tab.updateBook({ graphRagEnabled: status.graph_rag_enabled })
          if (
            status.indexing_status === 'complete' &&
            tab.book.ragIndexedContentVersionId !== versionId
          )
            await tab.updateBook({ ragIndexedContentVersionId: versionId })
        })
        .catch((reason) => {
          if (
            active &&
            !(
              reason instanceof RagApiError &&
              reason.code === 'preparation_not_found'
            )
          )
            setError(errorMessage(reason))
        })
    void refresh()
    const timer = setInterval(refresh, 5000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [tab, tab.book.id, versionId])

  const indexed =
    !!versionId && snapshot.book.ragIndexedContentVersionId === versionId
  const savedBoundary = snapshot.book.confirmedReadingBoundary
  const boundary = useMemo<ConfirmedBoundary | undefined>(() => {
    if (
      !versionId ||
      !savedBoundary ||
      savedBoundary.contentVersionId !== versionId
    )
      return undefined
    return {
      confirmed: true,
      contentVersionId: versionId,
      sectionIndex: savedBoundary.sectionIndex,
      unitIndex: savedBoundary.unitIndex,
      sourceUnitId: savedBoundary.sourceUnitId,
      cfi: savedBoundary.cfi,
    }
  }, [versionId, savedBoundary])

  const doIndex = async () => {
    if (!versionId || !tab.sections || tab.sectionLoadStatus !== 'ready') return
    setError('')
    setWork('extracting')
    setProgress('')
    try {
      const units = await extractLoadedBook(
        tab.book.id,
        versionId,
        tab.sections,
        (item) => {
          setProgress(
            t('indexing')
              .replace('{current}', String(item.completedSections))
              .replace('{total}', String(item.totalSections)),
          )
        },
      )
      tab.ragParagraphs = units
      const saved = await getSavedBookLibrary().catch(() => [])
      if (
        !saved.some(
          (item) =>
            item.book_id === tab.book.id &&
            item.content_version_id === versionId &&
            item.file_available,
        )
      ) {
        const localFile = await db?.files.get(tab.book.id)
        if (!localFile) throw new Error('The local EPUB file is unavailable.')
        setProgress('Saving the EPUB locally…')
        await saveEpubToBackend({
          bookId: tab.book.id,
          contentVersionId: versionId,
          title: String(tab.book.metadata.title || tab.book.name),
          originalFilename: tab.book.name,
          file: localFile.file,
        })
      }
      setWork('uploading')
      setProgress(t('uploading'))
      const result = await prepareBookSourceUnits(
        tab.book.id,
        versionId,
        units,
        fetch,
        tab.book.graphRagEnabled,
      )
      setPreparation(result)
      setProgress('Preparation queued. Keyword search will be ready first.')
      setWork('indexed')
    } catch (reason) {
      setWork('error')
      setProgress('')
      setError(errorMessage(reason))
    }
  }

  const confirmBoundary = async () => {
    if (boundarySaveLock.current) return
    if (confirmationTimer.current) clearTimeout(confirmationTimer.current)
    confirmationTimer.current = undefined
    setBoundaryConfirmed(false)
    setBoundaryError('')
    if (!versionId) return
    if (!indexed) {
      setBoundaryError(t('index_first'))
      return
    }
    boundarySaveLock.current = true
    setSavingBoundary(true)
    try {
      const units =
        tab.ragParagraphs ??
        (await extractLoadedBook(tab.book.id, versionId, tab.sections ?? []))
      const cfi = snapshot.location?.start.cfi
      const cfiApi = snapshot.rendition?.epubcfi
      if (!cfi || !cfiApi) throw new Error(t('boundary_unavailable'))
      const mapped = createBoundaryForLocation(units, cfi, cfiApi)
      if (!mapped) throw new Error(t('boundary_unavailable'))
      const confirmedBoundary = { contentVersionId: versionId, ...mapped }
      tab.ragParagraphs = units
      await tab.updateBook({ confirmedReadingBoundary: confirmedBoundary })
      if (!isMounted.current) return
      setBoundaryError('')
      setBoundaryConfirmed(true)
      confirmationTimer.current = setTimeout(() => {
        if (!isMounted.current) return
        setBoundaryConfirmed(false)
        confirmationTimer.current = undefined
      }, 1800)
    } catch (reason) {
      if (isMounted.current) setBoundaryError(errorMessage(reason))
    } finally {
      boundarySaveLock.current = false
      if (isMounted.current) setSavingBoundary(false)
    }
  }

  const submitQuestion = async (event?: FormEvent) => {
    event?.preventDefault()
    if (!boundary || !versionId || !question.trim()) return
    if (mode === 'semantic' && preparation?.embedding_status !== 'complete') {
      setError(
        'Semantic retrieval is unavailable until compatible embeddings are complete.',
      )
      return
    }
    if (
      mode === 'hybrid' &&
      (preparation?.indexing_status !== 'complete' ||
        preparation.embedding_status !== 'complete')
    ) {
      setError(
        'Hybrid retrieval needs the keyword index and compatible embeddings to be complete.',
      )
      return
    }
    if (mode === 'hybrid_graph') {
      if (preparation?.graph_rag_enabled !== true) {
        setError('Hybrid + GraphRAG is not enabled for this book.')
        return
      }
      if (preparation.graph_status !== 'complete') {
        setError(
          'BookNLP story graph preparation is not complete yet. You can use Keyword, Semantic, or Hybrid while it runs.',
        )
        return
      }
    }
    if (
      mode === 'keyword' &&
      preparation?.indexing_status !== 'complete' &&
      !indexed
    ) {
      setError('Keyword retrieval is unavailable until indexing is complete.')
      return
    }
    setAsking(true)
    setError('')
    setAnswer(undefined)
    try {
      const result = await askBook({
        bookId: tab.book.id,
        contentVersionId: versionId,
        query: question.trim(),
        retrievalMode: mode,
        boundary,
      })
      if (
        result.citations.some(
          (citation) =>
            citation.book_id !== tab.book.id ||
            citation.content_version_id !== versionId,
        )
      )
        throw new RagApiError(
          'A citation referred to another book revision.',
          'invalid_response',
        )
      setAnswer(result)
    } catch (reason) {
      if (
        reason instanceof RagApiError &&
        (reason.code === 'book_not_indexed' ||
          reason.code === 'content_version_mismatch')
      ) {
        tab.updateBook({ ragIndexedContentVersionId: undefined })
      }
      setError(errorMessage(reason))
    } finally {
      setAsking(false)
    }
  }

  const currentDisplay = progressLabel(tab, snapshot, indexed)
  const retryGraph = async () => {
    if (!versionId) return
    try {
      await retryFullBookGraph(tab.book.id, versionId)
      const status = await getBookPreparation(tab.book.id, versionId)
      setPreparation(status)
    } catch (reason) {
      setBoundaryError(errorMessage(reason))
    }
  }

  return (
    <PaneView {...props}>
      <div className="scroll-parent rag-panel-scroll space-y-2 px-5 pb-5">
        <RagPreparationProgress
          title={<p className="min-w-0 truncate py-1">{tab.book.name}</p>}
          book={
            preparation
              ? progressBook(tab, versionId, preparation)
              : fallbackProgressBook(tab, versionId, indexed)
          }
        />
        {indexed && work === 'uploading' && (
          <p role="status" className="text-sm">
            {progress || 'Saving the EPUB locally…'}
          </p>
        )}
        {!indexed ? (
          <section className="space-y-1.5">
            {snapshot.sectionLoadStatus !== 'ready' && (
              <p>{t('index_not_ready')}</p>
            )}
            <button
              className="bg-primary text-on-primary rounded px-3 py-1"
              disabled={
                work === 'extracting' ||
                work === 'uploading' ||
                preparation?.indexing_status === 'queued' ||
                preparation?.indexing_status === 'processing' ||
                snapshot.sectionLoadStatus !== 'ready'
              }
              onClick={doIndex}
            >
              {work === 'extracting' ||
              work === 'uploading' ||
              preparation?.indexing_status === 'queued' ||
              preparation?.indexing_status === 'processing'
                ? progress || t('loading')
                : t('index')}
            </button>
          </section>
        ) : (
          <>
            <section className="space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p
                  tabIndex={0}
                  title={savedProgressTitle(savedBoundary, versionId)}
                  aria-label={savedProgressTitle(savedBoundary, versionId)}
                  className="text-outline min-w-0 flex-1 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
                >
                  {currentDisplay}
                </p>
                <button
                  className="bg-secondary text-on-secondary rounded px-3 py-1"
                  disabled={
                    !versionId ||
                    !snapshot.location?.start.cfi ||
                    savingBoundary
                  }
                  onClick={() => void confirmBoundary()}
                >
                  {savingBoundary
                    ? 'Saving…'
                    : boundaryConfirmed
                    ? 'Confirmed'
                    : 'Set progress'}
                </button>
              </div>
              {versionId &&
                savedBoundary &&
                savedBoundary.contentVersionId !== versionId && (
                  <p role="status">{t('boundary_invalidated')}</p>
                )}
            </section>

            <section className="border-outline-variant space-y-1.5 border-t pt-2">
              <form
                onSubmit={submitQuestion}
                className="border-outline-variant bg-surface-variant/40 focus-within:border-primary rounded-xl border p-2 shadow-sm"
              >
                <textarea
                  id="lector-question"
                  name="lector-question"
                  value={question}
                  onChange={(event: React.ChangeEvent<HTMLTextAreaElement>) =>
                    setQuestion(event.target.value)
                  }
                  placeholder={t('question')}
                  aria-label={t('question')}
                  onKeyDown={(event) => {
                    if (
                      event.key === 'Enter' &&
                      (event.ctrlKey || event.metaKey)
                    )
                      void submitQuestion()
                  }}
                  className="text-on-surface placeholder:text-outline min-h-20 w-full resize-y bg-transparent px-2 py-1 text-sm outline-none"
                />
                <div className="border-outline-variant/60 flex items-center justify-between gap-2 border-t pt-2">
                  <label className="text-outline flex min-w-0 items-center gap-2 text-xs">
                    <span className="shrink-0">{t('mode')}</span>
                    <Select
                      value={mode}
                      aria-label={t('mode')}
                      onChange={(event) =>
                        setMode(event.target.value as RetrievalMode)
                      }
                      className="min-w-0"
                    >
                      {(
                        [
                          'keyword',
                          'semantic',
                          'hybrid',
                          'hybrid_graph',
                        ] as RetrievalMode[]
                      ).map((item) => (
                        <option
                          key={item}
                          value={item}
                          disabled={
                            (item === 'hybrid_graph' &&
                              preparation?.graph_rag_enabled !== true) ||
                            (item === 'semantic' &&
                              preparation?.embedding_status !== 'complete') ||
                            ((item === 'hybrid' || item === 'hybrid_graph') &&
                              (preparation?.embedding_status !== 'complete' ||
                                (preparation?.indexing_status !== 'complete' &&
                                  !indexed))) ||
                            (item === 'keyword' &&
                              preparation?.indexing_status !== 'complete' &&
                              !indexed)
                          }
                        >
                          {item === 'hybrid_graph'
                            ? 'Hybrid + GraphRAG'
                            : item === 'hybrid'
                            ? 'Hybrid'
                            : item === 'semantic'
                            ? 'Semantic'
                            : 'Keyword'}
                        </option>
                      ))}
                    </Select>
                  </label>
                  <button
                    type="submit"
                    className="bg-primary text-on-primary disabled:bg-disabled disabled:text-on-disabled flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition-opacity hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
                    aria-label={asking ? t('loading') : t('send')}
                    title={asking ? t('loading') : t('send')}
                    disabled={
                      !boundary ||
                      asking ||
                      !question.trim() ||
                      (mode === 'semantic' &&
                        preparation?.embedding_status !== 'complete') ||
                      ((mode === 'hybrid' || mode === 'hybrid_graph') &&
                        (preparation?.embedding_status !== 'complete' ||
                          (preparation?.indexing_status !== 'complete' &&
                            !indexed))) ||
                      (mode === 'hybrid_graph' &&
                        preparation?.graph_status !== 'complete')
                    }
                  >
                    <MdArrowUpward size={21} aria-hidden="true" />
                  </button>
                </div>
              </form>
              {mode === 'hybrid_graph' &&
                preparation?.graph_status !== 'complete' && (
                  <p role="status" className="text-outline text-xs">
                    {preparation?.graph_rag_enabled
                      ? ['failed', 'unavailable'].includes(
                          preparation.graph_status,
                        )
                        ? preparation.graph_error ||
                          'BookNLP graph preparation needs attention. Retry it from the messages below.'
                        : 'BookNLP is building the graph in the background. Other retrieval modes remain available.'
                      : 'GraphRAG is not enabled for this book.'}
                  </p>
                )}
              {answer && <AnswerView answer={answer} tab={tab} t={t} />}
            </section>
          </>
        )}

        {(error ||
          boundaryError ||
          snapshot.sectionLoadError ||
          work === 'error' ||
          snapshot.book.ragPreparationError ||
          preparation?.indexing_error ||
          preparation?.embedding_error ||
          preparation?.graph_error) && (
          <section
            className="border-outline-variant space-y-1 border-t pt-2"
            aria-label="Errors"
          >
            <h3>Messages</h3>
            {error && <p role="alert">{error}</p>}
            {boundaryError && <p role="alert">{boundaryError}</p>}
            {snapshot.sectionLoadError && (
              <p role="alert">{snapshot.sectionLoadError}</p>
            )}
            {snapshot.book.ragPreparationError && (
              <p role="alert">{snapshot.book.ragPreparationError}</p>
            )}
            {preparation?.indexing_error && (
              <p role="alert">{preparation.indexing_error}</p>
            )}
            {preparation?.embedding_error && (
              <p role="alert">{preparation.embedding_error}</p>
            )}
            {preparation?.graph_error && (
              <p role="alert">{preparation.graph_error}</p>
            )}
            {preparation?.graph_rag_enabled &&
              ['failed', 'unavailable'].includes(preparation.graph_status) && (
                <button className="underline" onClick={() => void retryGraph()}>
                  Retry graph preparation
                </button>
              )}
            {work === 'error' && <p>{t('partial')}</p>}
            {(error ||
              work === 'error' ||
              preparation?.indexing_status === 'failed' ||
              preparation?.embedding_status === 'failed' ||
              (indexed && preparation?.embedding_status === 'unavailable')) && (
              <button
                className="underline"
                onClick={() =>
                  void (question.trim() && boundary
                    ? submitQuestion()
                    : doIndex())
                }
              >
                {t('retry')}
              </button>
            )}
          </section>
        )}
      </div>
    </PaneView>
  )
}

function progressBook(
  tab: BookTab,
  versionId: string | undefined,
  preparation: BookPreparation,
) {
  const indexTotal = preparation.indexing_total
  const embeddingTotal = preparation.embedding_total
  return {
    book_id: preparation.book_id,
    title: String(tab.book.metadata.title || tab.book.name),
    original_filename: tab.book.name,
    content_version_id: versionId || preparation.content_version_id,
    file_size: tab.book.size,
    indexing_status:
      preparation.indexing_status === 'complete'
        ? ('indexed' as const)
        : ('uploaded' as const),
    preparation_indexing_status: preparation.indexing_status,
    indexing_progress:
      preparation.indexing_status === 'complete'
        ? 1
        : indexTotal
        ? preparation.indexing_completed / indexTotal
        : 0,
    indexing_completed: preparation.indexing_completed,
    indexing_total: indexTotal,
    indexing_error: preparation.indexing_error,
    embedding_status: preparation.embedding_status,
    embedding_progress:
      preparation.embedding_status === 'complete'
        ? 1
        : embeddingTotal
        ? preparation.embedding_completed / embeddingTotal
        : 0,
    embedding_completed: preparation.embedding_completed,
    embedding_total: embeddingTotal,
    embedding_error: preparation.embedding_error,
    graph_rag_enabled: preparation.graph_rag_enabled,
    graph_status: preparation.graph_status,
    graph_progress:
      preparation.graph_status === 'complete'
        ? 1
        : preparation.graph_total
        ? preparation.graph_completed / preparation.graph_total
        : 0,
    graph_completed: preparation.graph_completed,
    graph_total: preparation.graph_total,
    graph_error: preparation.graph_error,
    file_available: true,
    updated_at: preparation.updated_at,
  }
}

function fallbackProgressBook(
  tab: BookTab,
  versionId: string | undefined,
  indexed: boolean,
) {
  return {
    book_id: tab.book.id,
    title: String(tab.book.metadata.title || tab.book.name),
    original_filename: tab.book.name,
    content_version_id: versionId || '',
    file_size: tab.book.size,
    indexing_status: indexed ? ('indexed' as const) : ('uploaded' as const),
    preparation_indexing_status: indexed
      ? ('complete' as const)
      : ('pending' as const),
    indexing_progress: indexed ? 1 : 0,
    indexing_completed: 0,
    indexing_total: 0,
    embedding_status: 'unavailable' as const,
    embedding_progress: 0,
    embedding_completed: 0,
    embedding_total: 0,
    graph_rag_enabled: tab.book.graphRagEnabled ?? false,
    graph_status: tab.book.graphRagEnabled
      ? ('pending' as const)
      : ('not_enabled' as const),
    graph_progress: 0,
    graph_completed: 0,
    graph_total: 0,
    graph_error: undefined,
    file_available: true,
    updated_at: '',
  }
}

function AnswerView({
  answer,
  tab,
  t,
}: {
  answer: GroundedAnswer
  tab: BookTab
  t: (key: string) => string
}) {
  const [navigationMessage, setNavigationMessage] = useState('')
  return (
    <section className="border-outline-variant space-y-3 border-t pt-3">
      <p>
        {answer.status === 'insufficient_evidence'
          ? t('insufficient')
          : answer.answer}
      </p>
      {answer.uncertainty && (
        <p>
          <strong>{t('uncertainty')}:</strong> {answer.uncertainty}
        </p>
      )}
      <h3>{t('citations')}</h3>
      {answer.citations.map((citation) => {
        const sourceCfi = citation.source_positions.find(
          (position) => position.cfi,
        )?.cfi
        const candidateCfi =
          citation.cfi_status === 'unreliable' ||
          citation.cfi_status === 'unavailable'
            ? undefined
            : sourceCfi || citation.start_cfi || undefined
        const exactAvailable =
          !!candidateCfi && !!tab.rendition?.epubcfi.isCfiString(candidateCfi)
        const sectionHref =
          citation.href ?? citation.source_positions[0]?.href ?? ''
        const sectionAvailable = tab.sections?.some(
          (section) =>
            section.index === citation.section_index && !!section.href,
        )
        return (
          <article
            key={citation.passage_id}
            className="border-outline space-y-1 border-l-2 pl-2"
          >
            <p>{citation.excerpt}</p>
            <button
              className="underline"
              onClick={async () => {
                const result = await navigateToCitation(
                  citation,
                  tab.sections ?? [],
                  (cfi) => !!tab.rendition?.epubcfi.isCfiString(cfi),
                  async (target) => {
                    if (!tab.rendition) throw new Error('Reader is not ready')
                    tab.locationToReturn = tab.location
                    await tab.rendition.display(target)
                  },
                )
                setNavigationMessage(
                  result === 'exact'
                    ? t('navigate_exact')
                    : result === 'section'
                    ? t('navigate_section')
                    : t('navigation_unavailable'),
                )
              }}
            >
              {exactAvailable
                ? t('navigate_exact')
                : sectionAvailable
                ? t('navigate_section')
                : t('navigation_unavailable')}
            </button>
            <p className="text-outline">
              {tab.mapSectionToNavItem(sectionHref)?.label ?? sectionHref} ·{' '}
              {citation.cfi_status}
            </p>
          </article>
        )
      })}
      {navigationMessage && <p role="status">{navigationMessage}</p>}
      {answer.graph_fact_ids.length > 0 && (
        <details>
          <summary>{t('graph_facts')}</summary>
          <p>{answer.graph_fact_ids.join(', ')}</p>
        </details>
      )}
    </section>
  )
}

function errorMessage(error: unknown) {
  if (error instanceof RagApiError) {
    const friendlyMessages: Record<string, string> = {
      generation_not_configured:
        'Lector answer generation is not configured. Check the backend answer-service settings.',
      generation_authentication_failed:
        'Lector could not authenticate with its configured answer service. Check the backend credentials.',
      generation_rate_limited:
        'Lector is temporarily busy answering questions. Please try again shortly.',
      generation_timeout:
        'Lector took too long to prepare an answer. Please try again.',
      generation_structured_output_failed:
        'Lector could not complete a structured answer from the selected passages. Try again or simplify the question.',
      generation_malformed_response:
        'Lector received an answer in an unexpected format. Please try again.',
      generation_unavailable:
        'Lector could not reach its configured answer service. Check that the backend is running.',
    }
    return friendlyMessages[error.code] || brandProviderError(error.message)
  }
  return error instanceof Error
    ? brandProviderError(error.message)
    : 'The Lector request failed.'
}

function brandProviderError(message: string) {
  return message.replace(/\bGroq\b/gi, 'Lector')
}

function progressLabel(
  tab: BookTab,
  snapshot: {
    location?: {
      start?: {
        cfi?: string
        href?: string
        displayed?: { page?: number }
      }
    }
    rendition?: { epubcfi?: Parameters<typeof createBoundaryForLocation>[2] }
  },
  indexed: boolean,
) {
  const start = snapshot.location?.start
  const page = start?.displayed?.page
  const sectionLabel = start?.href
    ? tab.mapSectionToNavItem(start.href)?.label
    : undefined
  let paragraph: number | undefined
  const units = tab.ragParagraphs
  const cfiApi = snapshot.rendition?.epubcfi
  if (indexed && start?.cfi && cfiApi && units) {
    try {
      paragraph = createBoundaryForLocation(units, start.cfi, cfiApi)?.unitIndex
    } catch {
      paragraph = undefined
    }
  }
  const locationLabel =
    typeof page === 'number' && Number.isFinite(page) && page > 0
      ? 'Page ' + page
      : sectionLabel || 'Current section'
  return (
    locationLabel +
    (paragraph === undefined ? '' : ', paragraph ' + (paragraph + 1))
  )
}

function savedProgressTitle(
  savedBoundary:
    | { sectionIndex: number; unitIndex: number; contentVersionId: string }
    | undefined,
  versionId: string | undefined,
) {
  if (!savedBoundary) return 'No saved progress'
  const versionNote =
    versionId && savedBoundary.contentVersionId !== versionId
      ? ' (for a different book version)'
      : ''
  return `Last saved progress: section ${
    savedBoundary.sectionIndex + 1
  }, paragraph ${savedBoundary.unitIndex + 1}${versionNote}`
}
