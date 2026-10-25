import { StateLayer } from '@literal-ui/core'
import { useEffect, useMemo, useState } from 'react'
import { VscCollapseAll, VscExpandAll } from 'react-icons/vsc'

import {
  useLibrary,
  useList,
  useMobile,
  useTranslation,
} from '@flow/reader/hooks'
import {
  compareHref,
  dfs,
  flatTree,
  INavItem,
  reader,
  useReaderSnapshot,
} from '@flow/reader/models'

import { Row } from '../Row'
import { PaneViewProps, PaneView, Pane } from '../base'
import { getSavedBookLibrary, type SavedBook } from '../../rag/api'
import { RagPreparationProgress } from './RagPreparationProgress'

export const TocView: React.FC<PaneViewProps> = (props) => {
  const mobile = useMobile()
  return (
    <PaneView {...props}>
      {mobile || <LibraryPane />}
      <TocPane />
    </PaneView>
  )
}

const LibraryPane: React.FC = () => {
  const books = useLibrary()
  const t = useTranslation('toc')
  const [statuses, setStatuses] = useState<SavedBook[]>([])
  useEffect(() => {
    let active = true
    const refresh = () =>
      getSavedBookLibrary()
        .then((items) => active && setStatuses(items))
        .catch(() => undefined)
    void refresh()
    const timer = setInterval(refresh, 8000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])
  return (
    <Pane headline={t('library')} preferredSize={240}>
      {books?.map((book) => (
        <div key={book.id}>
          <RagPreparationProgress
            title={
              <button
                className="relative min-w-0 max-w-[calc(100%-1.5rem)] truncate py-1 pl-5 pr-1 text-left"
                title={book.name}
                draggable
                onClick={() => reader.addTab(book)}
                onDragStart={(e) => {
                  e.dataTransfer.setData('text/plain', book.id)
                }}
              >
                <StateLayer />
                {book.name}
              </button>
            }
            book={
              statuses.find((item) => item.book_id === book.id) ??
              fallbackStatus(book)
            }
          />
        </div>
      ))}
    </Pane>
  )
}

function fallbackStatus(book: {
  id: string
  name: string
  contentVersionId?: string
  graphRagEnabled?: boolean
}): SavedBook {
  return {
    book_id: book.id,
    title: book.name,
    original_filename: book.name,
    content_version_id: book.contentVersionId ?? '',
    file_size: 0,
    indexing_status: 'uploaded',
    preparation_indexing_status: 'pending',
    indexing_progress: 0,
    indexing_completed: 0,
    indexing_total: 0,
    embedding_status: 'pending',
    embedding_progress: 0,
    embedding_completed: 0,
    embedding_total: 0,
    graph_rag_enabled: book.graphRagEnabled ?? false,
    graph_status: book.graphRagEnabled ? 'pending' : 'not_enabled',
    graph_progress: 0,
    graph_completed: 0,
    graph_total: 0,
    file_available: false,
    updated_at: '',
  }
}

const TocPane: React.FC = () => {
  const t = useTranslation()
  const { focusedBookTab } = useReaderSnapshot()
  const toc = focusedBookTab?.nav?.toc as INavItem[] | undefined
  const rows = useMemo(() => toc?.flatMap((i) => flatTree(i)), [toc])
  const expanded = toc?.some((r) => r.expanded)
  const currentNavItem = focusedBookTab?.currentNavItem

  const { outerRef, innerRef, items, scrollToItem } = useList(rows)

  return (
    <Pane
      headline={t('toc.contents')}
      ref={outerRef}
      actions={[
        {
          id: expanded ? 'collapse-all' : 'expand-all',
          title: t(expanded ? 'action.collapse_all' : 'action.expand_all'),
          Icon: expanded ? VscCollapseAll : VscExpandAll,
          handle() {
            reader.focusedBookTab?.nav?.toc?.forEach((r) =>
              dfs(r as INavItem, (i) => (i.expanded = !expanded)),
            )
          },
        },
      ]}
    >
      {rows && (
        <div ref={innerRef}>
          {items.map(({ index }) => (
            <TocRow
              key={index}
              currentNavItem={currentNavItem as INavItem}
              item={rows[index]}
              onActivate={() => scrollToItem(index)}
            />
          ))}
        </div>
      )}
    </Pane>
  )
}

interface TocRowProps {
  currentNavItem?: INavItem
  item?: INavItem
  onActivate: () => void
}
const TocRow: React.FC<TocRowProps> = ({
  currentNavItem,
  item,
  onActivate,
}) => {
  if (!item) return null
  const { label, subitems, depth, expanded, id, href } = item
  const tab = reader.focusedBookTab

  return (
    <Row
      title={label.trim()}
      depth={depth}
      active={href === currentNavItem?.href}
      expanded={expanded}
      subitems={subitems}
      onClick={() => {
        const [, id] = href.split('#')
        const section = tab?.sections?.find((s) => compareHref(s.href, href))

        if (!section) return

        if (id) {
          tab?.displayFromSelector(`#${id}`, section, false)
        } else {
          tab?.display(section.href, false)
        }
      }}
      // `tab` can not be proxy here
      toggle={() => tab?.toggle(id)}
      onActivate={onActivate}
    />
  )
}
