import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  MdClose,
  MdZoomIn,
  MdZoomOut,
  MdCenterFocusStrong,
} from 'react-icons/md'
import { useSnapshot } from 'valtio'

import { useSetAction } from '@flow/reader/hooks'
import { BookTab } from '@flow/reader/models/reader'
import { reader, useReaderSnapshot } from '@flow/reader/models'
import {
  CHARACTER_GRAPH_LIMITS,
  DEFAULT_CHARACTER_GRAPH_LIMIT,
  expandGraphHorizontally,
  GRAPH_HORIZONTAL_SPREAD,
  GRAPH_LAYOUT_ITERATIONS,
  edgeStrokeOpacity,
  edgeStrokeWidth,
  fitGraphToViewport,
  layoutCharacterGraph,
  nodeLabelSize,
  nodeRadius,
  selectCharacterGraph,
} from '../../rag/graph-layout'

import {
  BookPreparation,
  CharacterGraphEdge,
  CharacterGraphNode,
  CharacterGraphView as CharacterGraphData,
  ConfirmedBoundary,
  getBookPreparation,
  getCharacterGraphView,
  RagApiError,
  resolveGraphContentVersionId,
  retryFullBookGraph,
} from '../../rag/api'

interface Point {
  x: number
  y: number
}

type Hovered =
  | { kind: 'node'; id: string }
  | { kind: 'edge'; id: string }
  | undefined

export const CharacterGraphView: React.FC<{ className?: string }> = ({
  className,
}) => {
  return className?.includes('!hidden') ? null : <CharacterGraphOverlay />
}

const CharacterGraphOverlay: React.FC = () => {
  const { focusedBookTab } = useReaderSnapshot()
  if (!focusedBookTab)
    return <MessageOverlay message="Open a book to view its character graph." />
  return (
    <BookCharacterGraph key={focusedBookTab.id} tab={reader.focusedBookTab!} />
  )
}

function BookCharacterGraph({ tab }: { tab: BookTab }) {
  const snapshot = useSnapshot(tab)
  const setAction = useSetAction()
  const boundaryRecord = snapshot.book.confirmedReadingBoundary
  const [versionId, setVersionId] = useState<string>()
  const [preparation, setPreparation] = useState<BookPreparation>()
  const [graph, setGraph] = useState<CharacterGraphData>()
  const [viewMode, setViewMode] = useState<'full_book' | 'till_now'>('till_now')
  const [status, setStatus] = useState('Checking this book’s graph status…')
  const [error, setError] = useState('')
  const [hovered, setHovered] = useState<Hovered>()
  const [pinnedNode, setPinnedNode] = useState<string>()
  const [characterLimit, setCharacterLimit] = useState(
    DEFAULT_CHARACTER_GRAPH_LIMIT,
  )
  const [labelsVisible, setLabelsVisible] = useState(false)
  const [viewport, setViewport] = useState({ width: 1200, height: 800 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 })
  const closeRef = useRef<HTMLButtonElement>(null)
  const dragStart = useRef<Point>()

  const close = useCallback(() => setAction(undefined), [setAction])

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      previousFocus?.focus?.()
    }
  }, [close])

  useEffect(() => {
    const resize = () =>
      setViewport({ width: window.innerWidth, height: window.innerHeight })
    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])

  useEffect(() => {
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    setGraph(undefined)
    setError('')
    setPreparation(undefined)
    setStatus('Checking this book’s graph status…')

    const refresh = async () => {
      try {
        const version = await resolveGraphContentVersionId(
          tab.book.contentVersionId ?? boundaryRecord?.contentVersionId,
          () => tab.getContentVersionId(),
        )
        if (!active) return
        setVersionId(version)
        if (
          viewMode === 'till_now' &&
          (!boundaryRecord || boundaryRecord.contentVersionId !== version)
        ) {
          setStatus(
            'Confirm a reading position in Lector before viewing the graph.',
          )
          return
        }
        const confirmed: ConfirmedBoundary | undefined =
          boundaryRecord && boundaryRecord.contentVersionId === version
            ? {
                confirmed: true,
                contentVersionId: version,
                sectionIndex: boundaryRecord.sectionIndex,
                unitIndex: boundaryRecord.unitIndex,
                sourceUnitId: boundaryRecord.sourceUnitId,
                cfi: boundaryRecord.cfi,
              }
            : undefined
        const prep = await getBookPreparation(tab.book.id, version)
        if (!active) return
        setPreparation(prep)
        if (!prep.graph_rag_enabled) {
          setStatus(
            'GraphRAG is not enabled for this book. It remains optional per book.',
          )
          return
        }
        if (
          prep.graph_status === 'failed' ||
          prep.graph_status === 'unavailable'
        ) {
          setStatus('BookNLP graph preparation needs attention.')
          if (prep.graph_error) setError(prep.graph_error)
          return
        }
        if (prep.graph_status !== 'complete') {
          setStatus(
            prep.graph_status === 'processing' && prep.graph_completed === 0
              ? 'BookNLP is analyzing the full book locally. Chunk-by-chunk graph progress begins after this analysis.'
              : prep.graph_total > 0
              ? `Saving full-book graph evidence: ${prep.graph_completed} of ${prep.graph_total} chunks.`
              : 'BookNLP graph preparation is queued. Reading and text retrieval remain available.',
          )
          timer = setTimeout(() => void refresh(), 4000)
          return
        }
        const result = await getCharacterGraphView(
          tab.book.id,
          version,
          confirmed,
          fetch,
          viewMode,
        )
        if (!active) return
        setGraph(result)
        setStatus(
          result.nodes.length
            ? viewMode === 'full_book'
              ? `Full-book character data is ready. This view may contain spoilers.`
              : `Character data is limited to evidence through your confirmed reading boundary.`
            : viewMode === 'full_book'
            ? 'The full-book graph has no stored character profiles or relationships.'
            : 'No characters or relationships are eligible at this reading position yet.',
        )
      } catch (reason) {
        if (!active) return
        if (
          reason instanceof RagApiError &&
          reason.code === 'story_graph_unavailable'
        ) {
          setStatus('The character graph is not ready yet.')
          timer = setTimeout(() => void refresh(), 4000)
          return
        }
        setError(messageFor(reason))
        setStatus('Lector could not load the character graph.')
      }
    }
    void refresh()
    return () => {
      active = false
      if (timer) clearTimeout(timer)
    }
  }, [
    tab,
    tab.book.id,
    boundaryRecord?.contentVersionId,
    boundaryRecord?.sectionIndex,
    boundaryRecord?.unitIndex,
    boundaryRecord?.sourceUnitId,
    viewMode,
  ])

  const selectedGraph = useMemo(
    () => selectCharacterGraph(graph, characterLimit),
    [graph, characterLimit],
  )
  const nodesById = useMemo(
    () => new Map(selectedGraph.nodes.map((node) => [node.id, node])),
    [selectedGraph],
  )
  const springPositions = useMemo(
    () =>
      layoutCharacterGraph(
        selectedGraph.nodes,
        selectedGraph.edges,
        42,
        GRAPH_LAYOUT_ITERATIONS,
      ),
    [selectedGraph],
  )
  const spreadPositions = useMemo(
    () => expandGraphHorizontally(springPositions, GRAPH_HORIZONTAL_SPREAD),
    [springPositions],
  )
  const maxNodeRadius = useMemo(
    () =>
      Math.max(
        8,
        ...selectedGraph.nodes.map((node) => nodeRadius(node.mention_count)),
      ),
    [selectedGraph],
  )
  const points = useMemo(
    () =>
      fitGraphToViewport(
        spreadPositions,
        viewport.width,
        viewport.height,
        maxNodeRadius,
        selectedGraph.nodes,
        selectedGraph.edges,
      ),
    [
      spreadPositions,
      viewport,
      maxNodeRadius,
      selectedGraph.nodes,
      selectedGraph.edges,
    ],
  )

  const activeNodeId = hovered?.kind === 'node' ? hovered.id : pinnedNode
  const activeEdgeId = hovered?.kind === 'edge' ? hovered.id : undefined
  const maxEdgeWeight = useMemo(
    () => Math.max(0, ...selectedGraph.edges.map((edge) => edge.weight)),
    [selectedGraph],
  )
  const activeEdge = selectedGraph.edges.find(
    (edge) => edge.id === activeEdgeId,
  )
  const activeNode = nodesById.get(activeNodeId || '')
  const activeNeighbors = useMemo(() => {
    const ids = new Set<string>()
    const edgeId = hovered?.kind === 'edge' ? hovered.id : undefined
    const nodeId = hovered?.kind === 'node' ? hovered.id : pinnedNode
    for (const edge of selectedGraph.edges) {
      if (
        edge.id === edgeId ||
        edge.source === nodeId ||
        edge.target === nodeId
      ) {
        ids.add(edge.source)
        ids.add(edge.target)
      }
    }
    return ids
  }, [selectedGraph, hovered, pinnedNode])

  const retry = async () => {
    if (!versionId) return
    setError('')
    try {
      await retryFullBookGraph(tab.book.id, versionId)
      setStatus('BookNLP graph preparation has been queued to retry.')
      setPreparation(await getBookPreparation(tab.book.id, versionId))
    } catch (reason) {
      setError(messageFor(reason))
    }
  }

  const panStart = (event: React.PointerEvent<SVGSVGElement>) => {
    if ((event.target as Element).closest('[data-graph-item]')) return
    dragStart.current = { x: event.clientX - pan.x, y: event.clientY - pan.y }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const panMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!dragStart.current) return
    setPan({
      x: event.clientX - dragStart.current.x,
      y: event.clientY - dragStart.current.y,
    })
  }

  const graphAvailable = !!selectedGraph.nodes.length
  const transform = `translate(${pan.x} ${pan.y}) translate(${
    viewport.width / 2
  } ${viewport.height / 2}) scale(${zoom}) translate(${-viewport.width / 2} ${
    -viewport.height / 2
  })`

  return (
    <div
      className="fixed inset-0 z-[100] overflow-hidden bg-[#090d1a] text-white"
      role="dialog"
      aria-modal="true"
      aria-label="Lector character graph"
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between p-4 sm:p-6">
        <div className="bg-slate-950/75 pointer-events-auto max-w-[min(70vw,38rem)] rounded-xl border border-white/10 px-4 py-3 shadow-xl backdrop-blur">
          <p className="text-sm font-medium">{tab.book.name}</p>
          <p role="status" className="mt-1 text-xs text-slate-300">
            {status}
          </p>
          <div
            className="mt-2 inline-flex rounded-lg border border-white/10 bg-black/20 p-0.5"
            role="group"
            aria-label="Graph view mode"
          >
            {(['full_book', 'till_now'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                aria-pressed={viewMode === mode}
                onClick={() => setViewMode(mode)}
                className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
                  viewMode === mode
                    ? 'bg-white/15 text-white'
                    : 'text-slate-300 hover:bg-white/5'
                }`}
              >
                {mode === 'full_book' ? 'Full book · spoilers' : 'Till now'}
              </button>
            ))}
          </div>
          {!!graph?.nodes.length && (
            <div className="mt-2 flex items-center gap-2 text-xs text-slate-300">
              <label htmlFor="lector-character-limit">Characters</label>
              <select
                id="lector-character-limit"
                value={characterLimit}
                onChange={(event) => {
                  setCharacterLimit(Number(event.target.value))
                  setHovered(undefined)
                  setPinnedNode(undefined)
                }}
                className="rounded-md border border-white/10 bg-slate-900 px-2 py-1 text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
              >
                {CHARACTER_GRAPH_LIMITS.map((limit) => (
                  <option key={limit} value={limit}>
                    Top {limit}
                  </option>
                ))}
              </select>
            </div>
          )}
          {error && (
            <p role="alert" className="mt-2 max-w-xl text-xs text-rose-300">
              {error}
            </p>
          )}
          {preparation?.graph_rag_enabled &&
            ['failed', 'unavailable'].includes(preparation.graph_status) && (
              <button
                className="mt-2 text-xs underline"
                onClick={() => void retry()}
              >
                Retry BookNLP graph preparation
              </button>
            )}
        </div>
        <div className="bg-slate-950/75 pointer-events-auto flex items-center gap-2 rounded-xl border border-white/10 p-1.5 shadow-xl backdrop-blur">
          <GraphControl
            label="Zoom out"
            onClick={() => setZoom((value) => Math.max(0.4, value / 1.2))}
            Icon={MdZoomOut}
          />
          <GraphControl
            label="Zoom in"
            onClick={() => setZoom((value) => Math.min(3.5, value * 1.2))}
            Icon={MdZoomIn}
          />
          <GraphControl
            label="Reset view"
            onClick={() => {
              setZoom(1)
              setPan({ x: 0, y: 0 })
            }}
            Icon={MdCenterFocusStrong}
          />
          <button
            type="button"
            aria-pressed={labelsVisible}
            aria-label={`Character labels ${labelsVisible ? 'on' : 'off'}`}
            onClick={() => setLabelsVisible((visible) => !visible)}
            className={`rounded-lg px-3 py-2 text-xs transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300 ${
              labelsVisible
                ? 'bg-cyan-400/20 text-cyan-100'
                : 'text-slate-300 hover:bg-white/10'
            }`}
          >
            Labels {labelsVisible ? 'On' : 'Off'}
          </button>
          <GraphControl
            label="Close character graph"
            onClick={close}
            Icon={MdClose}
            buttonRef={closeRef}
          />
        </div>
      </div>

      {graphAvailable && (
        <svg
          className="h-full w-full touch-none"
          viewBox={`0 0 ${viewport.width} ${viewport.height}`}
          role="img"
          aria-label={
            viewMode === 'full_book'
              ? 'Full-book character relationship graph that may contain spoilers. Drag to pan and use the wheel to zoom.'
              : 'Character relationship graph filtered through the confirmed reading boundary. Drag to pan and use the wheel to zoom.'
          }
          onPointerDown={panStart}
          onPointerMove={panMove}
          onPointerUp={() => {
            dragStart.current = undefined
          }}
          onPointerCancel={() => {
            dragStart.current = undefined
          }}
          onWheel={(event) => {
            event.preventDefault()
            setZoom((value) =>
              Math.max(
                0.4,
                Math.min(3.5, value * (event.deltaY < 0 ? 1.1 : 0.9)),
              ),
            )
          }}
          onClick={(event) => {
            if (!(event.target as Element).closest('[data-graph-item]'))
              setPinnedNode(undefined)
          }}
        >
          <defs>
            <radialGradient id="lector-graph-bg">
              <stop offset="0%" stopColor="#15213b" />
              <stop offset="100%" stopColor="#090d1a" />
            </radialGradient>
            <filter
              id="lector-node-glow"
              x="-100%"
              y="-100%"
              width="300%"
              height="300%"
            >
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <rect width="100%" height="100%" fill="url(#lector-graph-bg)" />
          <g transform={transform}>
            {selectedGraph.edges.map((edge) => {
              const start = points.get(edge.source)
              const end = points.get(edge.target)
              if (!start || !end) return null
              const incidentToFocusedNode =
                !!activeNodeId &&
                (edge.source === activeNodeId || edge.target === activeNodeId)
              const active = edge.id === activeEdgeId || incidentToFocusedNode
              const dim = !!(activeNodeId || activeEdgeId) && !active
              return (
                <g
                  key={edge.id}
                  data-graph-item="edge"
                  className="cursor-pointer"
                  onPointerEnter={() =>
                    setHovered({ kind: 'edge', id: edge.id })
                  }
                  onPointerLeave={() => setHovered(undefined)}
                  onClick={(event) => event.stopPropagation()}
                >
                  <line
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke={categoryColor(edge.candidate_category)}
                    strokeWidth={edgeStrokeWidth(
                      edge.weight,
                      maxEdgeWeight,
                      active,
                    )}
                    strokeOpacity={edgeStrokeOpacity(
                      edge.weight,
                      maxEdgeWeight,
                      active,
                      dim,
                    )}
                    strokeLinecap="round"
                    className="pointer-events-none transition-[stroke-opacity,stroke-width]"
                  />
                  <line
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke="transparent"
                    strokeWidth="18"
                  />
                </g>
              )
            })}
            {selectedGraph.nodes.map((node, index) => {
              const point = points.get(node.id)
              if (!point) return null
              const radius = nodeRadius(node.mention_count)
              const active =
                node.id === activeNodeId || activeNeighbors.has(node.id)
              const dim = !!(activeNodeId || activeEdgeId) && !active
              const color = nodeColor(index)
              return (
                <g
                  key={node.id}
                  data-graph-item="node"
                  className="cursor-pointer"
                  onPointerEnter={() =>
                    setHovered({ kind: 'node', id: node.id })
                  }
                  onPointerLeave={() => setHovered(undefined)}
                  onClick={(event) => {
                    event.stopPropagation()
                    setPinnedNode((current) =>
                      current === node.id ? undefined : node.id,
                    )
                  }}
                >
                  {active && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={radius + 8}
                      fill={color}
                      opacity="0.2"
                      filter="url(#lector-node-glow)"
                    />
                  )}
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={radius}
                    fill={color}
                    stroke="white"
                    strokeOpacity={active ? 0.95 : 0.55}
                    strokeWidth={active ? 2.5 : 1}
                    opacity={dim ? 0.2 : 0.95}
                  />
                  {labelsVisible && (
                    <text
                      x={point.x}
                      y={
                        point.y + radius + nodeLabelSize(node.mention_count) + 3
                      }
                      textAnchor="middle"
                      fontSize={nodeLabelSize(node.mention_count)}
                      fontWeight={active ? 600 : 400}
                      className="pointer-events-none"
                      opacity={dim ? 0.16 : 0.92}
                      fill="#f8fafc"
                      stroke="#090d1a"
                      strokeWidth="3"
                      strokeLinejoin="round"
                      paintOrder="stroke"
                    >
                      {node.name.length > 24
                        ? `${node.name.slice(0, 22)}…`
                        : node.name}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        </svg>
      )}

      {activeNode && <NodeCard node={activeNode} viewMode={viewMode} />}
      {activeEdge && (
        <EdgeCard edge={activeEdge} nodes={nodesById} viewMode={viewMode} />
      )}
      {!graphAvailable && !error && (
        <div className="absolute inset-0 flex items-center justify-center px-6 pt-24 text-center">
          <p className="bg-slate-950/60 max-w-lg rounded-2xl border border-white/10 px-6 py-5 text-sm text-slate-200 shadow-xl backdrop-blur">
            {graph?.nodes.length
              ? `No characters with retained, evidence-backed edges are in the selected Top ${characterLimit}. Increase the character limit to explore more of this ${
                  viewMode === 'till_now' ? 'boundary-eligible ' : ''
                }graph.`
              : status}
          </p>
        </div>
      )}
    </div>
  )
}

function GraphControl({
  label,
  onClick,
  Icon,
  buttonRef,
}: {
  label: string
  onClick: () => void
  Icon: React.ComponentType<{ size?: number }>
  buttonRef?: React.RefObject<HTMLButtonElement>
}) {
  return (
    <button
      ref={buttonRef}
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex h-9 w-9 items-center justify-center rounded-lg text-slate-200 transition hover:bg-white/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-300"
    >
      <Icon size={20} />
    </button>
  )
}

function NodeCard({
  node,
  viewMode,
}: {
  node: CharacterGraphNode
  viewMode: 'full_book' | 'till_now'
}) {
  const profileTerms =
    viewMode === 'full_book' ? fullProfileTerms(node.profile_summary) : []
  return (
    <aside className="bg-slate-950/85 absolute bottom-5 left-5 z-10 max-w-sm rounded-2xl border border-white/10 p-4 shadow-2xl backdrop-blur sm:bottom-7 sm:left-7">
      <p className="text-base font-semibold">{node.name}</p>
      <p className="mt-1 text-xs text-slate-300">
        {node.mention_count} mentions
      </p>
      {!!node.aliases.length && (
        <p className="mt-2 text-xs text-slate-300">
          <strong className="text-slate-100">Aliases:</strong>{' '}
          {node.aliases.join(', ')}
        </p>
      )}
      {!!node.pronouns.length && (
        <p className="mt-1 text-xs text-slate-300">
          <strong className="text-slate-100">Pronouns:</strong>{' '}
          {node.pronouns.join(', ')}
        </p>
      )}
      {!!profileTerms.length && (
        <p className="line-clamp-4 mt-2 text-xs text-slate-300">
          <strong className="text-slate-100">Full-book profile:</strong>{' '}
          {profileTerms.join(' · ')}
        </p>
      )}
      <p className="mt-2 text-[11px] text-slate-400">
        {viewMode === 'full_book'
          ? 'Profile may include later-book information.'
          : 'Names and aliases use eligible mentions only.'}
      </p>
    </aside>
  )
}

function fullProfileTerms(summary?: Record<string, unknown>): string[] {
  if (!summary) return []
  const terms: string[] = []
  for (const key of ['agent', 'patient', 'poss', 'mod']) {
    const values = summary[key]
    if (!Array.isArray(values)) continue
    for (const value of values) {
      if (
        value &&
        typeof value === 'object' &&
        'n' in value &&
        typeof value.n === 'string'
      )
        terms.push(value.n)
    }
  }
  return [...new Set(terms)].slice(0, 8)
}

function EdgeCard({
  edge,
  nodes,
  viewMode,
}: {
  edge: CharacterGraphEdge
  nodes: Map<string, CharacterGraphNode>
  viewMode: 'full_book' | 'till_now'
}) {
  const excerpt = edge.evidence[0]
  return (
    <aside className="bg-slate-950/90 absolute bottom-5 right-5 z-10 max-w-sm rounded-2xl border border-white/10 p-4 shadow-2xl backdrop-blur sm:bottom-7 sm:right-7">
      <p className="text-sm font-semibold">
        {nodes.get(edge.source)?.name} · {nodes.get(edge.target)?.name}
      </p>
      <p className="mt-1 text-xs text-cyan-200">
        {edge.candidate_category.replace(/_/g, ' ')}
      </p>
      <p className="mt-1 text-xs text-slate-300">
        {edge.evidence_count} {viewMode === 'full_book' ? 'stored' : 'eligible'}{' '}
        co-mention
        {edge.evidence_count === 1 ? '' : 's'}
      </p>
      {!!edge.candidate_verbs.length && (
        <p className="mt-1 text-xs text-slate-300">
          Candidate verbs: {edge.candidate_verbs.join(', ')}
        </p>
      )}
      {excerpt?.sentence && (
        <blockquote className="mt-3 border-l-2 border-cyan-400/70 pl-3 text-xs leading-relaxed text-slate-200">
          {excerpt.sentence}
        </blockquote>
      )}
      <p className="mt-2 text-[11px] text-amber-200/90">
        Co-mentions and verbs are heuristic evidence; they do not prove a direct
        relationship.
      </p>
    </aside>
  )
}

function MessageOverlay({ message }: { message: string }) {
  const setAction = useSetAction()
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[#090d1a] p-6 text-center text-white"
      role="dialog"
      aria-modal="true"
    >
      <div className="bg-slate-950/70 rounded-2xl border border-white/10 p-6">
        <p>{message}</p>
        <button
          className="mt-4 rounded-lg bg-white/10 px-4 py-2 text-sm"
          onClick={() => setAction(undefined)}
        >
          Close
        </button>
      </div>
    </div>
  )
}

function nodeColor(index: number) {
  return `hsl(${(index * 137.508 + 185) % 360} 76% 62%)`
}

function categoryColor(category: string) {
  const colors: Record<string, string> = {
    POSSIBLE_CONFLICT: '#fb7185',
    POSSIBLE_HELP: '#4ade80',
    POSSIBLE_TRANSFER: '#fbbf24',
    POSSIBLE_DIALOGUE: '#60a5fa',
    POSSIBLE_MOVEMENT: '#c084fc',
    ACTION_UNVERIFIED: '#2dd4bf',
    CO_MENTION: '#94a3b8',
  }
  return colors[category] || '#94a3b8'
}

function messageFor(error: unknown) {
  if (error instanceof RagApiError) {
    if (error.code === 'graph_rag_disabled')
      return 'GraphRAG is not enabled for this book.'
    if (error.code === 'story_graph_unavailable')
      return 'BookNLP graph preparation is not complete yet.'
    if (error.code === 'invalid_reading_boundary')
      return 'Confirm a valid reading position for this book before opening its graph.'
  }
  return error instanceof Error
    ? error.message
    : 'The Lector graph request failed.'
}
