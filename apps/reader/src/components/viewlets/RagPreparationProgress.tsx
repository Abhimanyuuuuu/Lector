import React, { useState } from 'react'
import { VscChevronRight } from 'react-icons/vsc'

import type { SavedBook } from '../../rag/api'

/** Compact, persisted indexing, embedding, and graph state shared by reader/library. */
export const RagPreparationProgress: React.FC<{
  book: SavedBook
  title: React.ReactNode
}> = ({ book, title }) => {
  const [expanded, setExpanded] = useState(true)

  return (
    <section className="pb-2 text-xs">
      <div className="min-h-6 flex items-center gap-1">
        {title}
        <button
          type="button"
          className="text-outline hover:text-on-surface-variant focus-visible:outline-primary inline-flex h-6 w-5 shrink-0 items-center justify-center focus-visible:outline focus-visible:outline-2"
          aria-label={`${
            expanded ? 'Collapse' : 'Expand'
          } preparation details for ${book.title}`}
          aria-expanded={expanded}
          onClick={(event) => {
            event.stopPropagation()
            setExpanded((value) => !value)
          }}
        >
          <VscChevronRight
            aria-hidden="true"
            className={`shrink-0 transition-transform duration-150 ${
              expanded ? 'rotate-90' : ''
            }`}
            size={16}
          />
        </button>
      </div>
      {expanded && (
        <div className="space-y-1 px-5 pb-1 text-xs">
          <PreparationBar
            label="Indexing"
            status={book.preparation_indexing_status}
            progress={book.indexing_progress}
            completed={book.indexing_completed}
            total={book.indexing_total}
            error={book.indexing_error}
            color="bg-primary"
          />
          <PreparationBar
            label="Embeddings"
            status={book.embedding_status}
            progress={book.embedding_progress}
            completed={book.embedding_completed}
            total={book.embedding_total}
            error={book.embedding_error}
            color="bg-secondary"
          />
          <PreparationBar
            label="Story graph"
            status={book.graph_status}
            progress={book.graph_progress}
            completed={book.graph_completed}
            total={book.graph_total}
            error={book.graph_error}
            color="bg-emerald-600"
          />
        </div>
      )}
    </section>
  )
}

function PreparationBar({
  label,
  status,
  progress,
  completed,
  total,
  error,
  color,
}: {
  label: string
  status: string
  progress: number
  completed: number
  total: number
  error?: string | null
  color: string
}) {
  const complete =
    status === 'complete' || status === 'indexed' || status === 'not_applicable'
  const failed = status === 'failed' || status === 'stale'
  const unavailable = status === 'unavailable'
  const terminal = failed || unavailable
  const known = total > 0
  const analyzingBook =
    label === 'Story graph' && status === 'processing' && completed === 0
  const value = complete
    ? 100
    : terminal
    ? 0
    : Math.max(0, Math.min(100, progress * 100))
  const labelStatus =
    status === 'not_enabled'
      ? 'Graph RAG disabled'
      : status === 'not_applicable'
      ? 'not applicable'
      : unavailable
      ? 'unavailable'
      : failed
      ? 'failed'
      : complete
      ? 'complete'
      : status
  return (
    <div className="flex items-center gap-2">
      <span className="w-[4.5rem] shrink-0">{label}</span>
      <div
        className="bg-surface-variant h-1.5 min-w-0 flex-1 overflow-hidden rounded"
        role="progressbar"
        aria-label={`${label}: ${labelStatus}`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={
          known || complete ? (analyzingBook ? undefined : value) : undefined
        }
        aria-valuetext={
          analyzingBook
            ? 'BookNLP full-book analysis in progress'
            : known && !terminal
            ? `${completed} of ${total}, ${labelStatus}`
            : labelStatus
        }
        title={
          error ||
          (analyzingBook
            ? 'BookNLP is analyzing the full book locally'
            : known && !terminal
            ? `${completed}/${total} · ${labelStatus}`
            : labelStatus)
        }
      >
        <div
          className={`${color} h-full rounded ${terminal ? 'opacity-40' : ''} ${
            (analyzingBook || !known) &&
            !complete &&
            !terminal &&
            status !== 'not_enabled' &&
            ['queued', 'pending', 'processing'].includes(status)
              ? 'w-1/3 animate-pulse'
              : ''
          }`}
          style={
            (known && !analyzingBook) ||
            complete ||
            terminal ||
            status === 'not_enabled'
              ? { width: `${value}%` }
              : undefined
          }
        />
      </div>
      <span className="w-24 shrink-0 text-right capitalize">
        {analyzingBook
          ? 'analyzing book'
          : known && !complete && !terminal
          ? `${completed}/${total}`
          : labelStatus}
      </span>
    </div>
  )
}
