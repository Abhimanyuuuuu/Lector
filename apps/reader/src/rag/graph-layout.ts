import type {
  CharacterGraphEdge,
  CharacterGraphNode,
  CharacterGraphView,
} from './api'

export const CHARACTER_GRAPH_LIMITS = [20, 30, 50, 75] as const
export const DEFAULT_CHARACTER_GRAPH_LIMIT = 30
export const GRAPH_IDEAL_DISTANCE = 1.2
export const GRAPH_LAYOUT_ITERATIONS = 75
export const GRAPH_HORIZONTAL_SPREAD = 1.5
const MAX_RELATIONSHIP_ATTRACTION = 3
const CORE_COHESION_STRENGTH = 0.12
const CORE_COHESION_DISTANCE = 2.25

export interface SelectedCharacterGraph {
  nodes: CharacterGraphNode[]
  edges: CharacterGraphEdge[]
}

export interface GraphPoint {
  x: number
  y: number
}

export interface GraphLayoutDiagnostics {
  iterationsRun: number
  finalNormalizedMovement: number
  stoppedByTolerance: boolean
}

/**
 * Selects prominent characters using the already mode-filtered graph payload.
 * Mention frequency is the primary signal; eligible weighted degree only
 * breaks mention-count ties. Names and IDs provide stable final tie-breaks.
 */
export function selectCharacterGraph(
  graph: Pick<CharacterGraphView, 'nodes' | 'edges'> | undefined,
  limit = DEFAULT_CHARACTER_GRAPH_LIMIT,
): SelectedCharacterGraph {
  if (!Number.isInteger(limit) || limit < 1)
    throw new RangeError('Character graph limit must be a positive integer.')
  if (!graph) return { nodes: [], edges: [] }

  const degree = new Map<string, number>()
  const graphNodeIds = new Set(graph.nodes.map((node) => node.id))
  for (const edge of graph.edges) {
    if (
      !graphNodeIds.has(edge.source) ||
      !graphNodeIds.has(edge.target) ||
      edge.evidence_count <= 0 ||
      !Number.isFinite(edge.weight) ||
      edge.weight <= 0
    )
      continue
    const weight = Number.isFinite(edge.weight) ? Math.max(0, edge.weight) : 0
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + weight)
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + weight)
  }

  const rankedNodes = [...graph.nodes]
    .filter((node) => node.id && node.name.trim())
    .sort(
      (left, right) =>
        safeCount(right.mention_count) - safeCount(left.mention_count) ||
        (degree.get(right.id) ?? 0) - (degree.get(left.id) ?? 0) ||
        left.name.localeCompare(right.name) ||
        left.id.localeCompare(right.id),
    )
    .slice(0, limit)

  const selectedIds = new Set(rankedNodes.map((node) => node.id))
  const candidateEdges = graph.edges
    .filter(
      (edge) =>
        selectedIds.has(edge.source) &&
        selectedIds.has(edge.target) &&
        edge.evidence_count > 0 &&
        Number.isFinite(edge.weight) &&
        edge.weight > 0,
    )
    .sort((left, right) => left.id.localeCompare(right.id))
  const connectedIds = new Set<string>()
  for (const edge of candidateEdges) {
    connectedIds.add(edge.source)
    connectedIds.add(edge.target)
  }
  const nodes = rankedNodes.filter((node) => connectedIds.has(node.id))
  const connectedNodeIds = new Set(nodes.map((node) => node.id))
  const edges = candidateEdges.filter(
    (edge) =>
      connectedNodeIds.has(edge.source) && connectedNodeIds.has(edge.target),
  )

  return { nodes, edges }
}

/**
 * Deterministic Fruchterman-Reingold spring layout, using the NetworkX force
 * equations on normalized coordinates. Nodes start from a seeded random
 * distribution, all pairs repel, and evidence-backed edges attract according
 * to their bounded evidence weight. Bounded extra cohesion is limited to
 * pairs sharing a real neighbor, so structural importance does not pull
 * unrelated hubs into one global center. Output is centered and normalized
 * independently of viewport dimensions, so resizing only requires
 * fitGraphToViewport().
 */
export function layoutCharacterGraph(
  nodes: readonly CharacterGraphNode[],
  edges: readonly CharacterGraphEdge[],
  seed = 42,
  iterations = GRAPH_LAYOUT_ITERATIONS,
  idealDistance = GRAPH_IDEAL_DISTANCE,
  diagnostics?: GraphLayoutDiagnostics,
): Map<string, GraphPoint> {
  if (!nodes.length) {
    if (diagnostics)
      Object.assign(diagnostics, {
        iterationsRun: 0,
        finalNormalizedMovement: 0,
        stoppedByTolerance: true,
      })
    return new Map()
  }
  if (!Number.isInteger(iterations) || iterations < 1)
    throw new RangeError('Spring layout iterations must be a positive integer.')
  if (!Number.isFinite(idealDistance) || idealDistance <= 0)
    throw new RangeError('Spring layout ideal distance must be positive.')

  // Preserve selectedGraph insertion order, matching NetworkX's graph-node
  // order so seed 42 assigns the same initial random coordinates.
  const orderedNodes = [...nodes]
  const random = numpyRandomState(seed)
  const positions = orderedNodes.map(() => ({ x: random(), y: random() }))
  const indexById = new Map(orderedNodes.map((node, index) => [node.id, index]))
  const weights = Array.from({ length: nodes.length }, () =>
    Array<number>(nodes.length).fill(0),
  )
  for (const edge of [...edges].sort((left, right) =>
    left.id.localeCompare(right.id),
  )) {
    const source = indexById.get(edge.source)
    const target = indexById.get(edge.target)
    if (
      source === undefined ||
      target === undefined ||
      source === target ||
      !Number.isFinite(edge.weight) ||
      edge.weight <= 0
    )
      continue
    // The Colab graph omitted an edge `weight` and therefore used unit springs.
    // Lector's API weight is the eligible source-evidence count; the bounded
    // transform uses that existing signal without letting prolific pairs
    // dominate the force field.
    weights[source]![target] = relationshipAttraction(edge.weight)
    weights[target]![source] = relationshipAttraction(edge.weight)
  }
  const weightedDegrees = weights.map((row) =>
    row.reduce((sum, attraction) => sum + attraction, 0),
  )
  const minimumDegree = Math.min(...weightedDegrees)
  const maximumDegree = Math.max(...weightedDegrees)
  const degreeRange = maximumDegree - minimumDegree
  const centrality = weightedDegrees.map((degree) =>
    degreeRange > 0 ? (degree - minimumDegree) / degreeRange : 0,
  )
  const maxMentions = Math.max(
    ...orderedNodes.map((node) => safeCount(node.mention_count)),
    1,
  )
  const importance = orderedNodes.map(
    (node) =>
      Math.log1p(safeCount(node.mention_count)) / Math.log1p(maxMentions),
  )
  // Build only two-hop pairs from real evidence-backed edges. This adds no
  // visible relationship; it gently gathers characters who share a strongly
  // connected, important neighbor into a readable core.
  const commonNeighborPairs = new Map<string, number>()
  for (let center = 0; center < weights.length; center += 1) {
    const neighbors = weights[center]!.map((weight, index) => ({
      index,
      weight,
    })).filter((item) => item.weight > 0)
    for (let left = 0; left < neighbors.length; left += 1) {
      for (let right = left + 1; right < neighbors.length; right += 1) {
        const first = neighbors[left]!
        const second = neighbors[right]!
        const pairKey = `${Math.min(first.index, second.index)}:${Math.max(
          first.index,
          second.index,
        )}`
        const sharedStrength =
          Math.min(first.weight, second.weight) / MAX_RELATIONSHIP_ATTRACTION
        const structuralImportance = Math.sqrt(
          centrality[first.index]! *
            importance[first.index]! *
            centrality[second.index]! *
            importance[second.index]!,
        )
        const affinity = sharedStrength * structuralImportance
        if (affinity > (commonNeighborPairs.get(pairKey) ?? 0))
          commonNeighborPairs.set(pairKey, affinity)
      }
    }
  }
  const cohesionPairs = [...commonNeighborPairs]
    .map(([key, affinity]) => {
      const [left, right] = key.split(':').map(Number)
      return { left: left!, right: right!, affinity }
    })
    .filter((pair) => pair.affinity > 0)

  // Keep the existing NetworkX-style cooling schedule. The default 75-step
  // budget is based on the checked 50-step residual movement in the graph
  // fixture; callers may still request a different explicit limit.
  const initialTemperature =
    Math.max(
      Math.max(...positions.map((point) => point.x)) -
        Math.min(...positions.map((point) => point.x)),
      Math.max(...positions.map((point) => point.y)) -
        Math.min(...positions.map((point) => point.y)),
    ) * 0.1
  const temperatureStep = initialTemperature / (iterations + 1)
  let iterationsRun = 0
  let finalNormalizedMovement = Number.POSITIVE_INFINITY
  let stoppedByTolerance = false
  for (let step = 0; step < iterations; step += 1) {
    const displacements = positions.map(() => ({ x: 0, y: 0 }))
    for (let left = 0; left < positions.length; left += 1) {
      for (let right = left + 1; right < positions.length; right += 1) {
        const a = positions[left]!
        const b = positions[right]!
        let deltaX = a.x - b.x
        let deltaY = a.y - b.y
        let distance = Math.hypot(deltaX, deltaY)
        if (distance < 1e-4) {
          deltaX = 1e-4
          deltaY = 0
          distance = 1e-4
        }
        const weight = weights[left]![right]!
        const springForce =
          (idealDistance * idealDistance) / (distance * distance) -
          (weight * distance) / idealDistance
        displacements[left]!.x += deltaX * springForce
        displacements[left]!.y += deltaY * springForce
        displacements[right]!.x -= deltaX * springForce
        displacements[right]!.y -= deltaY * springForce
      }
    }

    for (const pair of cohesionPairs) {
      const a = positions[pair.left]!
      const b = positions[pair.right]!
      const deltaX = a.x - b.x
      const deltaY = a.y - b.y
      const distance = Math.max(Math.hypot(deltaX, deltaY), 1e-4)
      const excess = Math.max(
        0,
        distance - CORE_COHESION_DISTANCE * idealDistance,
      )
      if (!excess) continue
      const force =
        (-CORE_COHESION_STRENGTH * pair.affinity * excess) / distance
      displacements[pair.left]!.x += deltaX * force
      displacements[pair.left]!.y += deltaY * force
      displacements[pair.right]!.x -= deltaX * force
      displacements[pair.right]!.y -= deltaY * force
    }

    const temperature = initialTemperature - temperatureStep * step
    let movementSquared = 0
    for (let index = 0; index < positions.length; index += 1) {
      const displacement = displacements[index]!
      const magnitude = Math.hypot(displacement.x, displacement.y)
      const movementScale = temperature / (magnitude < 0.01 ? 0.1 : magnitude)
      positions[index]!.x += displacement.x * movementScale
      positions[index]!.y += displacement.y * movementScale
      movementSquared +=
        (displacement.x * movementScale) ** 2 +
        (displacement.y * movementScale) ** 2
    }
    iterationsRun = step + 1
    finalNormalizedMovement = Math.sqrt(movementSquared) / positions.length
    if (finalNormalizedMovement < 1e-4) {
      stoppedByTolerance = true
      break
    }
  }
  if (diagnostics)
    Object.assign(diagnostics, {
      iterationsRun,
      finalNormalizedMovement,
      stoppedByTolerance,
    })

  const centerX =
    positions.reduce((sum, point) => sum + point.x, 0) / positions.length
  const centerY =
    positions.reduce((sum, point) => sum + point.y, 0) / positions.length
  const centered = positions.map((point) => ({
    x: point.x - centerX,
    y: point.y - centerY,
  }))
  // Orient the naturally widest axis horizontally. Rotation preserves every
  // pairwise distance and does not impose a shape or distort circular nodes.
  const covarianceXX = centered.reduce(
    (sum, point) => sum + point.x * point.x,
    0,
  )
  const covarianceYY = centered.reduce(
    (sum, point) => sum + point.y * point.y,
    0,
  )
  const covarianceXY = centered.reduce(
    (sum, point) => sum + point.x * point.y,
    0,
  )
  const angle = 0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY)
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  const horizontallyOriented = centered.map((point) => ({
    x: point.x * cosine + point.y * sine,
    y: -point.x * sine + point.y * cosine,
  }))
  // NetworkX rescale_layout uses the largest absolute coordinate, not the
  // Euclidean radius. This preserves the reference graph's aspect ratio.
  const extent = Math.max(
    ...horizontallyOriented.flatMap((point) => [
      Math.abs(point.x),
      Math.abs(point.y),
    ]),
    1e-6,
  )
  return new Map(
    orderedNodes.map((node, index) => {
      const point = horizontallyOriented[index]!
      return [node.id, { x: point.x / extent, y: point.y / extent }]
    }),
  )
}

export function nodeRadius(mentionCount: number): number {
  // Colab's node_size is a Matplotlib area in pt². Convert its area rule to
  // an SVG pixel radius, with a cap for high-mention characters.
  const area = Math.max(100, safeCount(mentionCount) * 4)
  return Math.min(16, Math.sqrt(area / Math.PI) * (96 / 72))
}

/**
 * API edge weights count eligible evidence records. Rational saturation keeps
 * attraction strictly increasing for every finite positive weight while
 * approaching a ceiling of three times the one-evidence baseline.
 */
export function relationshipAttraction(weight: number): number {
  if (!Number.isFinite(weight) || weight <= 0) return 0
  return weight <= 2
    ? (MAX_RELATIONSHIP_ATTRACTION * weight) / (weight + 2)
    : MAX_RELATIONSHIP_ATTRACTION / (1 + 2 / weight)
}

export function edgeStrokeWidth(
  weight: number,
  maxWeight: number,
  highlighted: boolean,
): number {
  const maximum = Math.max(0, Number.isFinite(maxWeight) ? maxWeight : 0)
  const value = Math.max(0, Number.isFinite(weight) ? weight : 0)
  const relative = maximum > 0 ? Math.sqrt(Math.min(1, value / maximum)) : 0
  return highlighted ? 1.7 + relative * 2.8 : 0.5 + relative
}

export function edgeStrokeOpacity(
  weight: number,
  maxWeight: number,
  highlighted: boolean,
  dimmed: boolean,
): number {
  if (dimmed) return 0.1
  const maximum = Math.max(0, Number.isFinite(maxWeight) ? maxWeight : 0)
  const value = Math.max(0, Number.isFinite(weight) ? weight : 0)
  const relative = maximum > 0 ? Math.sqrt(Math.min(1, value / maximum)) : 0
  // Keep every edge visible, while letting weak evidence recede behind the
  // stronger relationships that define the graph's central web.
  return highlighted ? 0.68 + relative * 0.28 : 0.12 + relative * 0.13
}

export function nodeLabelSize(mentionCount: number): number {
  return 9 + ((nodeRadius(mentionCount) - 7) / 16) * 7
}

/** Widens the existing layout around its horizontal center without moving y. */
export function expandGraphHorizontally(
  positions: ReadonlyMap<string, GraphPoint>,
  factor = GRAPH_HORIZONTAL_SPREAD,
): Map<string, GraphPoint> {
  if (!Number.isFinite(factor) || factor <= 0)
    throw new RangeError('Horizontal graph spread must be a positive number.')
  if (!positions.size) return new Map()
  const xValues = [...positions.values()].map((point) => point.x)
  const centerX = (Math.min(...xValues) + Math.max(...xValues)) / 2
  return new Map(
    [...positions].map(([id, point]) => [
      id,
      { x: centerX + (point.x - centerX) * factor, y: point.y },
    ]),
  )
}

/** Uniformly fits spring positions and corrects only actual circle overlap. */
export function fitGraphToViewport(
  positions: ReadonlyMap<string, GraphPoint>,
  width: number,
  height: number,
  maxNodeRadius: number,
  nodes: readonly CharacterGraphNode[] = [],
  edges: readonly CharacterGraphEdge[] = [],
): Map<string, GraphPoint> {
  if (!positions.size) return new Map()
  const points = [...positions.values()]
  const minX = Math.min(...points.map((point) => point.x))
  const maxX = Math.max(...points.map((point) => point.x))
  const minY = Math.min(...points.map((point) => point.y))
  const maxY = Math.max(...points.map((point) => point.y))
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  const spanX = Math.max(maxX - minX, 1e-3)
  const spanY = Math.max(maxY - minY, 1e-3)
  const padding = Math.max(36, maxNodeRadius + 22)
  const availableWidth = Math.max(1, width - padding * 2)
  const availableHeight = Math.max(1, height - padding * 2)
  const scale = Math.min(availableWidth / spanX, availableHeight / spanY)
  const fitted = new Map(
    [...positions].map(([id, point]) => [
      id,
      {
        x: width / 2 + (point.x - centerX) * scale,
        y: height / 2 + (point.y - centerY) * scale,
      },
    ]),
  )
  return nodes.length
    ? resolveCircleOverlaps(fitted, nodes, width, height, edges)
    : fitted
}

/** Separates circles only when their rendered bounds overlap. */
function resolveCircleOverlaps(
  input: ReadonlyMap<string, GraphPoint>,
  nodes: readonly CharacterGraphNode[],
  width: number,
  height: number,
  edges: readonly CharacterGraphEdge[],
): Map<string, GraphPoint> {
  const positions = new Map(input)
  const ordered = [...nodes].filter((node) => positions.has(node.id))
  const margin = 2
  const rigidity = new Map(ordered.map((node) => [node.id, 0]))
  for (const edge of edges) {
    if (
      !rigidity.has(edge.source) ||
      !rigidity.has(edge.target) ||
      edge.source === edge.target
    )
      continue
    const attraction = relationshipAttraction(edge.weight)
    rigidity.set(edge.source, rigidity.get(edge.source)! + attraction)
    rigidity.set(edge.target, rigidity.get(edge.target)! + attraction)
  }
  const clamp = (value: number, radius: number, extent: number) => {
    const min = radius + margin
    const max = extent - radius - margin
    return min <= max ? Math.max(min, Math.min(max, value)) : extent / 2
  }
  for (let pass = 0; pass < 12; pass += 1) {
    let moved = false
    for (let left = 0; left < ordered.length; left += 1) {
      for (let right = left + 1; right < ordered.length; right += 1) {
        const firstNode = ordered[left]!
        const secondNode = ordered[right]!
        const first = positions.get(firstNode.id)!
        const second = positions.get(secondNode.id)!
        let dx = second.x - first.x
        let dy = second.y - first.y
        let distance = Math.hypot(dx, dy)
        const required =
          nodeRadius(firstNode.mention_count) +
          nodeRadius(secondNode.mention_count) +
          1
        if (distance >= required) continue
        if (distance < 1e-6) {
          dx = 1
          dy = 0
          distance = 1
        }
        const shift = required - distance
        const unitX = dx / distance
        const unitY = dy / distance
        // Preserve the spring layout's connected core: higher weighted-degree
        // nodes have less collision mobility, while peripheral nodes absorb
        // more of the correction. With no edge metadata this remains an even
        // split, as before.
        const firstMobility = 1 / (1 + rigidity.get(firstNode.id)!)
        const secondMobility = 1 / (1 + rigidity.get(secondNode.id)!)
        const mobilityTotal = firstMobility + secondMobility
        const firstShift = shift * (firstMobility / mobilityTotal)
        const secondShift = shift * (secondMobility / mobilityTotal)
        positions.set(firstNode.id, {
          x: clamp(
            first.x - unitX * firstShift,
            nodeRadius(firstNode.mention_count),
            width,
          ),
          y: clamp(
            first.y - unitY * firstShift,
            nodeRadius(firstNode.mention_count),
            height,
          ),
        })
        positions.set(secondNode.id, {
          x: clamp(
            second.x + unitX * secondShift,
            nodeRadius(secondNode.mention_count),
            width,
          ),
          y: clamp(
            second.y + unitY * secondShift,
            nodeRadius(secondNode.mention_count),
            height,
          ),
        })
        moved = true
      }
    }
    if (!moved) break
  }
  return positions
}

function safeCount(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0
}

/** NumPy legacy RandomState-compatible MT19937 doubles used by NetworkX. */
function numpyRandomState(seed: number): () => number {
  const state = new Uint32Array(624)
  state[0] = seed >>> 0
  for (let index = 1; index < state.length; index += 1) {
    const previous = state[index - 1]!
    state[index] =
      (Math.imul(1812433253, previous ^ (previous >>> 30)) + index) >>> 0
  }
  let position = 624
  const nextUInt32 = () => {
    if (position >= 624) {
      for (let index = 0; index < 624; index += 1) {
        const y =
          (state[index]! & 0x80000000) |
          (state[(index + 1) % 624]! & 0x7fffffff)
        state[index] =
          state[(index + 397) % 624]! ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)
      }
      position = 0
    }
    let value = state[position++]!
    value ^= value >>> 11
    value ^= (value << 7) & 0x9d2c5680
    value ^= (value << 15) & 0xefc60000
    value ^= value >>> 18
    return value >>> 0
  }
  return () => {
    const high = nextUInt32() >>> 5
    const low = nextUInt32() >>> 6
    return (high * 67108864 + low) / 9007199254740992
  }
}
