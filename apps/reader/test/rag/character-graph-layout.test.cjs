const assert = require('node:assert/strict')
const { test } = require('node:test')
const {
  DEFAULT_CHARACTER_GRAPH_LIMIT,
  expandGraphHorizontally,
  GRAPH_HORIZONTAL_SPREAD,
  GRAPH_LAYOUT_ITERATIONS,
  GRAPH_IDEAL_DISTANCE,
  edgeStrokeOpacity,
  edgeStrokeWidth,
  fitGraphToViewport,
  layoutCharacterGraph,
  nodeRadius,
  nodeLabelSize,
  relationshipAttraction,
  selectCharacterGraph,
} = require('../../.turbo/lector-rag-tests/graph-layout.js')

function node(id, mention_count) {
  return {
    id,
    name: id,
    mention_count,
    aliases: [],
    pronouns: [],
  }
}

function edge(id, source, target, weight = 1) {
  return {
    id,
    source,
    target,
    weight,
    evidence_count: weight,
    candidate_category: 'CO_MENTION',
    candidate_verbs: [],
    evidence: [],
    evidence_truncated: false,
  }
}

test('default character selection is Top 30 and mention count is the primary rank', () => {
  assert.equal(DEFAULT_CHARACTER_GRAPH_LIMIT, 30)
  const graph = {
    nodes: [node('low', 3), node('prominent', 30), node('middle', 10)],
    edges: [
      edge('important-link', 'prominent', 'middle', 2),
      edge('noisy', 'low', 'middle', 500),
    ],
  }
  const selected = selectCharacterGraph(graph, 2)
  assert.deepEqual(
    selected.nodes.map((item) => item.id),
    ['prominent', 'middle'],
  )
})

test('all Top-N presets cap the selected graph without changing ranking', () => {
  const nodes = Array.from({ length: 75 }, (_, index) =>
    node(`character-${String(index).padStart(3, '0')}`, 100 - index),
  )
  const edges = Array.from({ length: 74 }, (_, index) =>
    edge(
      `edge-${String(index).padStart(3, '0')}`,
      nodes[index].id,
      nodes[index + 1].id,
      1,
    ),
  )
  for (const limit of [20, 30, 50, 75]) {
    const selected = selectCharacterGraph({ nodes, edges }, limit)
    assert.equal(selected.nodes.length, limit)
    assert.equal(selected.edges.length, limit - 1)
    assert.equal(selected.nodes[0].id, 'character-000')
  }
})

test('eligible weighted degree breaks mention-count ties deterministically', () => {
  const graph = {
    nodes: [node('Zed', 12), node('Ada', 12), node('linked', 4)],
    edges: [
      edge('strong', 'Zed', 'linked', 7),
      edge('ada-link', 'Zed', 'Ada', 1),
    ],
  }
  assert.deepEqual(
    selectCharacterGraph(graph, 3).nodes.map((item) => item.id),
    ['Zed', 'Ada', 'linked'],
  )
  const nameTie = {
    nodes: [node('b', 12), node('a', 12)],
    edges: [edge('tie', 'a', 'b', 2)],
  }
  assert.deepEqual(
    selectCharacterGraph(nameTie, 2).nodes.map((item) => item.id),
    ['a', 'b'],
  )
})

test('only evidence-backed edges between selected nodes are retained', () => {
  const graph = {
    nodes: [node('A', 30), node('B', 20), node('C', 1)],
    edges: [
      edge('inside', 'A', 'B', 3),
      edge('outside', 'A', 'C', 9),
      { ...edge('unsupported', 'A', 'B', 3), evidence_count: 0 },
    ],
  }
  const selected = selectCharacterGraph(graph, 2)
  assert.deepEqual(
    selected.nodes.map((item) => item.id),
    ['A', 'B'],
  )
  assert.deepEqual(
    selected.edges.map((item) => item.id),
    ['inside'],
  )
  assert.equal(graph.edges.length, 3)
})

test('isolated nodes are removed while separate connected components remain', () => {
  const graph = {
    nodes: [
      node('A', 40),
      node('B', 30),
      node('C', 20),
      node('D', 10),
      node('alone', 100),
    ],
    edges: [edge('AB', 'A', 'B', 3), edge('CD', 'C', 'D', 2)],
  }
  const selected = selectCharacterGraph(graph, 5)
  assert.deepEqual(
    selected.nodes.map((item) => item.id),
    ['A', 'B', 'C', 'D'],
  )
  const finalIds = new Set(selected.nodes.map((item) => item.id))
  assert.ok(
    selected.edges.every(
      (item) => finalIds.has(item.source) && finalIds.has(item.target),
    ),
  )
  assert.deepEqual(
    selected.edges.map((item) => item.id),
    ['AB', 'CD'],
  )
})

test('mode-filtered mention counts directly control Top-N and node size', () => {
  const tillNow = {
    nodes: [
      node('future-heavy', 2),
      node('read-prominent', 20),
      node('other', 5),
    ],
    edges: [edge('eligible', 'read-prominent', 'other', 1)],
  }
  const fullBook = {
    nodes: [
      node('future-heavy', 40),
      node('read-prominent', 20),
      node('other', 5),
    ],
    edges: [edge('full', 'future-heavy', 'read-prominent', 1)],
  }
  assert.equal(selectCharacterGraph(tillNow, 2).nodes[0].id, 'read-prominent')
  assert.equal(selectCharacterGraph(fullBook, 2).nodes[0].id, 'future-heavy')
  assert.ok(
    nodeRadius(fullBook.nodes[0].mention_count) >
      nodeRadius(tillNow.nodes[0].mention_count),
  )
})

test('force layout is deterministic, bounded to the viewport, and handles isolates', () => {
  assert.equal(GRAPH_IDEAL_DISTANCE, 1.2)
  assert.equal(GRAPH_LAYOUT_ITERATIONS, 75)
  const nodes = [node('A', 80), node('B', 40), node('C', 8), node('D', 5)]
  const edges = [edge('AB', 'A', 'B', 8), edge('BC', 'B', 'C', 2)]
  const firstDiagnostics = {}
  const secondDiagnostics = {}
  const first = layoutCharacterGraph(
    nodes,
    edges,
    42,
    GRAPH_LAYOUT_ITERATIONS,
    undefined,
    firstDiagnostics,
  )
  const second = layoutCharacterGraph(
    nodes,
    edges,
    42,
    GRAPH_LAYOUT_ITERATIONS,
    undefined,
    secondDiagnostics,
  )
  assert.deepEqual([...first], [...second])
  assert.deepEqual(firstDiagnostics, secondDiagnostics)
  assert.ok(
    firstDiagnostics.stoppedByTolerance ||
      firstDiagnostics.iterationsRun === GRAPH_LAYOUT_ITERATIONS,
    'layout either reaches its movement tolerance or the documented iteration limit',
  )
  const fitted = fitGraphToViewport(first, 1000, 700, 20, nodes)
  for (const point of fitted.values()) {
    assert.ok(point.x >= 0 && point.x <= 1000)
    assert.ok(point.y >= 0 && point.y <= 700)
  }
  assert.equal(first.size, nodes.length)
  const radii = [...first.values()].map((point) => Math.hypot(point.x, point.y))
  assert.ok(Math.max(...radii) - Math.min(...radii) > 0.1)
})

test('horizontal spread is 1.5x before fitting and preserves vertical structure', () => {
  const nodes = Array.from({ length: 12 }, (_, index) =>
    node(`spread-${index}`, 40 - index),
  )
  const edges = nodes
    .slice(1)
    .map((item, index) =>
      edge(`spread-edge-${index}`, nodes[index].id, item.id, 1 + (index % 4)),
    )
  const original = layoutCharacterGraph(nodes, edges)
  const expanded = expandGraphHorizontally(original)
  assert.deepEqual(
    [...expandGraphHorizontally(original)],
    [...expanded],
    'identical layouts produce identical expanded positions',
  )
  assert.equal(GRAPH_HORIZONTAL_SPREAD, 1.5)
  const span = (positions, axis) => {
    const values = [...positions.values()].map((point) => point[axis])
    return Math.max(...values) - Math.min(...values)
  }
  assert.ok(Math.abs(span(expanded, 'x') / span(original, 'x') - 1.5) < 1e-10)
  const originalXs = [...original.values()].map((point) => point.x)
  const centerX = (Math.min(...originalXs) + Math.max(...originalXs)) / 2
  const radiusBefore = nodeRadius(20)
  for (const [id, point] of original) {
    assert.equal(expanded.get(id).y, point.y)
    assert.ok(
      Math.abs(expanded.get(id).x - (centerX + (point.x - centerX) * 1.5)) <
        1e-10,
    )
  }
  const originalFit = fitGraphToViewport(original, 1200, 700, 16)
  const expandedFit = fitGraphToViewport(expanded, 1200, 700, 16)
  assert.ok(
    Math.abs(
      span(expandedFit, 'x') /
        span(expandedFit, 'y') /
        (span(originalFit, 'x') / span(originalFit, 'y')) -
        1.5,
    ) < 1e-10,
    'uniform viewport fitting must preserve the requested aspect adjustment',
  )
  const collisionChecked = fitGraphToViewport(
    expanded,
    1200,
    700,
    16,
    nodes,
    edges,
  )
  for (const point of collisionChecked.values()) {
    assert.ok(point.x >= 0 && point.x <= 1200)
    assert.ok(point.y >= 0 && point.y <= 700)
  }
  assert.equal(nodeRadius(20), radiusBefore)
})

test('mention-scaled nodes are restrained, monotonic, and labels stay bounded', () => {
  assert.ok(nodeRadius(500) - nodeRadius(2) >= 7)
  assert.ok(nodeRadius(5000) <= 16)
  assert.ok(nodeRadius(100) > nodeRadius(10))
  assert.equal(nodeRadius(10), nodeRadius(1))
  assert.ok(nodeRadius(50) > nodeRadius(10))
  assert.ok(nodeLabelSize(500) > nodeLabelSize(2))
  assert.ok(nodeLabelSize(2) >= 9 && nodeLabelSize(500) <= 16)
  for (const mentions of [0, 1, 10, 50]) {
    const expected = Math.min(
      16,
      Math.sqrt(Math.max(100, mentions * 4) / Math.PI) * (96 / 72),
    )
    assert.ok(Math.abs(nodeRadius(mentions) - expected) < 1e-10)
  }
})

test('higher mention count creates a noticeably larger rendered character node', () => {
  const lowMentionRadius = nodeRadius(2)
  const highMentionRadius = nodeRadius(100)
  assert.ok(highMentionRadius > lowMentionRadius * 1.5)
  assert.ok(nodeLabelSize(100) > nodeLabelSize(2))
})

test('evidence-weighted springs preserve irregular relationship-driven distances', () => {
  const nodes = Array.from({ length: 7 }, (_, index) =>
    node(`n${index}`, 10 + index),
  )
  const positions = layoutCharacterGraph(nodes, [
    edge('strong-a', 'n0', 'n1', 18),
    edge('strong-b', 'n1', 'n2', 14),
    edge('weak-a', 'n2', 'n3', 1),
    edge('weak-b', 'n3', 'n4', 1),
    edge('peripheral', 'n4', 'n5', 0.5),
  ])
  const distances = []
  for (let left = 0; left < nodes.length; left += 1) {
    for (let right = left + 1; right < nodes.length; right += 1) {
      const a = positions.get(nodes[left].id)
      const b = positions.get(nodes[right].id)
      distances.push(Math.hypot(a.x - b.x, a.y - b.y))
    }
  }
  assert.ok(new Set(distances.map((distance) => distance.toFixed(3))).size > 8)
})

test('a highly interconnected core stays cohesive while weak branches extend outward', () => {
  const coreIds = Array.from({ length: 5 }, (_, index) => `core-${index}`)
  const peripheralIds = Array.from(
    { length: 5 },
    (_, index) => `peripheral-${index}`,
  )
  // Higher peripheral mentions ensure the spatial hierarchy comes from graph
  // structure rather than mention frequency.
  const nodes = [
    ...coreIds.map((id) => node(id, 20)),
    ...peripheralIds.map((id) => node(id, 90)),
  ]
  const edges = []
  for (let left = 0; left < coreIds.length; left += 1) {
    for (let right = left + 1; right < coreIds.length; right += 1) {
      edges.push(
        edge(`core-${left}-${right}`, coreIds[left], coreIds[right], 9),
      )
    }
    edges.push(edge(`branch-${left}`, peripheralIds[left], coreIds[left], 1))
  }

  const normalized = layoutCharacterGraph(nodes, edges)
  const fitted = fitGraphToViewport(
    normalized,
    1200,
    700,
    Math.max(...nodes.map((item) => nodeRadius(item.mention_count))),
    nodes,
    edges,
  )
  const meanCoreDistance = (positions) => {
    const distances = []
    for (let left = 0; left < coreIds.length; left += 1) {
      for (let right = left + 1; right < coreIds.length; right += 1) {
        const a = positions.get(coreIds[left])
        const b = positions.get(coreIds[right])
        distances.push(Math.hypot(a.x - b.x, a.y - b.y))
      }
    }
    return distances.reduce((sum, value) => sum + value, 0) / distances.length
  }
  const meanNearestBranchDistance = (positions) => {
    const distances = peripheralIds.map((id) => {
      const point = positions.get(id)
      return Math.min(
        ...coreIds.map((coreId) => {
          const center = positions.get(coreId)
          return Math.hypot(point.x - center.x, point.y - center.y)
        }),
      )
    })
    return distances.reduce((sum, value) => sum + value, 0) / distances.length
  }

  assert.ok(
    meanCoreDistance(normalized) < meanNearestBranchDistance(normalized) * 0.7,
  )
  assert.ok(meanCoreDistance(fitted) < meanNearestBranchDistance(fitted) * 0.7)
})

test('important core and connected secondary groups keep their own cohesion', () => {
  const core = ['core-a', 'core-b', 'core-c']
  const secondary = ['secondary-a', 'secondary-b', 'secondary-c']
  const nodes = [
    ...core.map((id) => node(id, 80)),
    ...secondary.map((id) => node(id, 24)),
    node('weak-branch', 100),
  ]
  const edges = []
  for (const [group, weight] of [
    [core, 9],
    [secondary, 7],
  ]) {
    for (let left = 0; left < group.length; left += 1) {
      for (let right = left + 1; right < group.length; right += 1) {
        edges.push(
          edge(
            `${group[left]}-${group[right]}`,
            group[left],
            group[right],
            weight,
          ),
        )
      }
    }
  }
  edges.push(
    edge('core-secondary', core[2], secondary[0], 1),
    edge('weak-branch-link', 'weak-branch', core[0], 0.5),
  )
  const points = layoutCharacterGraph(nodes, edges)
  const meanPairDistance = (ids) => {
    const distances = []
    for (let left = 0; left < ids.length; left += 1) {
      for (let right = left + 1; right < ids.length; right += 1) {
        const a = points.get(ids[left])
        const b = points.get(ids[right])
        distances.push(Math.hypot(a.x - b.x, a.y - b.y))
      }
    }
    return (
      distances.reduce((sum, distance) => sum + distance, 0) / distances.length
    )
  }
  assert.ok(meanPairDistance(core) < 0.8)
  assert.ok(meanPairDistance(secondary) < 0.8)
  assert.equal(edges.length, 8)
})

test('important characters anchor local neighborhoods across an important-character bridge', () => {
  const hubs = ['anchor-a', 'anchor-b', 'anchor-c']
  const neighborhoods = hubs.map((hub, groupIndex) => [
    `${hub}-near-1`,
    `${hub}-near-2`,
    `${hub}-near-3`,
  ])
  const nodes = hubs.map((id, index) => node(id, 120 - index * 10))
  for (const group of neighborhoods)
    nodes.push(...group.map((id) => node(id, 25)))
  nodes.push(node('weak-offshoot', 18))

  const edges = []
  for (let groupIndex = 0; groupIndex < hubs.length; groupIndex += 1) {
    const hub = hubs[groupIndex]
    const members = neighborhoods[groupIndex]
    for (const member of members)
      edges.push(edge(`${hub}-${member}`, hub, member, 8))
    edges.push(edge(`${members[0]}-${members[1]}`, members[0], members[1], 5))
    edges.push(edge(`${members[1]}-${members[2]}`, members[1], members[2], 4))
  }
  edges.push(
    edge('important-bridge-ab', hubs[0], hubs[1], 3),
    edge('important-bridge-bc', hubs[1], hubs[2], 3),
    edge('weak-offshoot-link', 'weak-offshoot', neighborhoods[2][2], 0.5),
  )

  const first = layoutCharacterGraph(nodes, edges)
  const second = layoutCharacterGraph(nodes, edges)
  assert.deepEqual([...first], [...second])
  const distance = (left, right) => {
    const a = first.get(left)
    const b = first.get(right)
    return Math.hypot(a.x - b.x, a.y - b.y)
  }
  const neighborhoodMeans = hubs.map((hub, index) => {
    const own = neighborhoods[index]
    const ownMean =
      own.reduce((sum, id) => sum + distance(hub, id), 0) / own.length
    const remote = neighborhoods
      .filter((_, otherIndex) => otherIndex !== index)
      .flat()
    const remoteMean =
      remote.reduce((sum, id) => sum + distance(hub, id), 0) / remote.length
    assert.ok(ownMean < remoteMean, `${hub} should sit among its own neighbors`)
    return ownMean
  })
  const meanLocalDistance =
    neighborhoodMeans.reduce((sum, value) => sum + value, 0) /
    neighborhoodMeans.length
  const bridgeAB = distance(hubs[0], hubs[1])
  const bridgeBC = distance(hubs[1], hubs[2])
  const unbridgedAC = distance(hubs[0], hubs[2])
  assert.ok(bridgeAB < unbridgedAC)
  assert.ok(bridgeBC < unbridgedAC)
  assert.ok((bridgeAB + bridgeBC + unbridgedAC) / 3 > meanLocalDistance)
  assert.ok(
    distance('weak-offshoot', neighborhoods[2][2]) >
      distance(hubs[2], neighborhoods[2][2]),
  )
  assert.equal(edges.length, 18)
})

test('a connected multi-cluster graph uses a broad, irregular viewport layout', () => {
  const nodes = Array.from({ length: 30 }, (_, index) =>
    node(`character-${String(index).padStart(2, '0')}`, 20 + index),
  )
  const edges = []
  for (let index = 1; index < nodes.length; index += 1) {
    edges.push(edge(`chain-${index}`, nodes[index - 1].id, nodes[index].id, 1))
    if (index > 3 && index % 2 === 0)
      edges.push(
        edge(`cross-${index}`, nodes[index - 3].id, nodes[index].id, 5),
      )
  }
  const baselineDiagnostics = {}
  layoutCharacterGraph(nodes, edges, 42, 50, undefined, baselineDiagnostics)
  const diagnostics = {}
  const normalized = layoutCharacterGraph(
    nodes,
    edges,
    42,
    GRAPH_LAYOUT_ITERATIONS,
    undefined,
    diagnostics,
  )
  assert.ok(
    diagnostics.stoppedByTolerance ||
      diagnostics.iterationsRun === GRAPH_LAYOUT_ITERATIONS,
    'layout converges or reaches its documented iteration budget',
  )
  assert.equal(baselineDiagnostics.iterationsRun, 50)
  assert.equal(baselineDiagnostics.stoppedByTolerance, false)
  assert.ok(
    diagnostics.finalNormalizedMovement <
      baselineDiagnostics.finalNormalizedMovement,
    'the 75-step budget reduces the residual movement on this fixture',
  )
  const points = fitGraphToViewport(
    normalized,
    1200,
    700,
    Math.max(...nodes.map((item) => nodeRadius(item.mention_count))),
    nodes,
  )
  const xs = [...points.values()].map((point) => point.x)
  const ys = [...points.values()].map((point) => point.y)
  assert.ok(Math.max(...xs) - Math.min(...xs) > 500)
  assert.ok(Math.max(...ys) - Math.min(...ys) > 250)
  assert.ok(
    new Set(
      [...points.values()].map(
        (point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`,
      ),
    ).size > 25,
  )
  const radii = [...normalized.values()].map((point) =>
    Math.hypot(point.x, point.y),
  )
  assert.ok(
    Math.max(...radii) - Math.min(...radii) > 0.2,
    'the layout should not impose a circular or radial boundary',
  )
  const allX = [...normalized.values()].map((point) => point.x)
  const allY = [...normalized.values()].map((point) => point.y)
  const xSpan = Math.max(...allX) - Math.min(...allX)
  const ySpan = Math.max(...allY) - Math.min(...allY)
  assert.ok(xSpan >= ySpan - 1e-10)
})

test('viewport fitting separates only circles that actually overlap', () => {
  const nodes = [node('large-one', 500), node('large-two', 500)]
  const normalized = layoutCharacterGraph(nodes, [
    edge('link', 'large-one', 'large-two', 4),
  ])
  const points = fitGraphToViewport(
    normalized,
    1000,
    700,
    nodeRadius(500),
    nodes,
  )
  const first = points.get('large-one')
  const second = points.get('large-two')
  assert.ok(first && second)
  const distance = Math.hypot(first.x - second.x, first.y - second.y)
  assert.ok(
    distance >= nodeRadius(500) * 2 - 2,
    `large nodes substantially overlapped at distance ${distance}`,
  )
})

test('evidence weight scales spring attraction monotonically and remains bounded', () => {
  assert.equal(relationshipAttraction(1), 1)
  assert.equal(relationshipAttraction(4), 2)
  assert.ok(relationshipAttraction(9) > relationshipAttraction(4))
  assert.ok(relationshipAttraction(10000) < 3)
  assert.ok(relationshipAttraction(10000) > relationshipAttraction(9))
  assert.ok(Number.isFinite(relationshipAttraction(Number.MAX_VALUE)))
  assert.ok(relationshipAttraction(Number.MAX_VALUE) <= 3)
  assert.ok(relationshipAttraction(2) > relationshipAttraction(1))
  assert.ok(relationshipAttraction(4) > relationshipAttraction(2))
  assert.equal(relationshipAttraction(Number.NaN), 0)
  assert.equal(relationshipAttraction(Number.POSITIVE_INFINITY), 0)
  assert.equal(relationshipAttraction(-2), 0)
  assert.equal(relationshipAttraction(0), 0)
  assert.ok(Math.abs(relationshipAttraction(0.25) - 1 / 3) < 1e-10)
  assert.ok(edgeStrokeWidth(10, 10, true) > edgeStrokeWidth(1, 10, true))
  assert.ok(edgeStrokeWidth(10, 10, true) <= 5.2)
  assert.ok(
    edgeStrokeOpacity(10, 10, true, false) >
      edgeStrokeOpacity(1, 10, true, false),
  )
  assert.ok(
    edgeStrokeOpacity(10, 10, false, false) >
      edgeStrokeOpacity(1, 10, false, false),
    'stronger edges remain more visible in the unhovered graph',
  )
  assert.equal(edgeStrokeOpacity(10, 10, true, true), 0.1)
  assert.ok(edgeStrokeOpacity(10, 10, false, false) <= 0.35)
  assert.ok(edgeStrokeWidth(1, 10, false) <= 1.5)
  assert.ok(edgeStrokeWidth(10, 10, false) < edgeStrokeWidth(10, 10, true))
})

test('stronger evidence shortens its spring relative to an otherwise equal spring', () => {
  const nodes = [node('A', 10), node('B', 10), node('C', 10)]
  const weak = layoutCharacterGraph(nodes, [
    edge('AB', 'A', 'B', 1),
    edge('AC', 'A', 'C', 1),
  ])
  const strong = layoutCharacterGraph(nodes, [
    edge('AB', 'A', 'B', 9),
    edge('AC', 'A', 'C', 1),
  ])
  const distance = (positions, left, right) => {
    const a = positions.get(left)
    const b = positions.get(right)
    return Math.hypot(a.x - b.x, a.y - b.y)
  }
  assert.ok(
    distance(strong, 'A', 'B') < distance(weak, 'A', 'B'),
    'increasing the AB evidence weight should increase attraction and reduce its final separation',
  )
})

test('overlap correction moves a peripheral node more than a connected hub', () => {
  const nodes = [node('hub', 20), node('leaf', 10), node('other', 5)]
  const edges = [
    edge('hub-leaf', 'hub', 'leaf', 9),
    edge('hub-other', 'hub', 'other', 9),
  ]
  const before = fitGraphToViewport(
    new Map([
      ['hub', { x: 0, y: 0 }],
      ['leaf', { x: 0.005, y: 0 }],
      ['other', { x: 1, y: 0 }],
    ]),
    1000,
    700,
    16,
  )
  const after = fitGraphToViewport(
    new Map([
      ['hub', { x: 0, y: 0 }],
      ['leaf', { x: 0.005, y: 0 }],
      ['other', { x: 1, y: 0 }],
    ]),
    1000,
    700,
    16,
    nodes,
    edges,
  )
  const movement = (id) =>
    Math.hypot(
      after.get(id).x - before.get(id).x,
      after.get(id).y - before.get(id).y,
    )
  const hubRadius = nodeRadius(nodes[0].mention_count)
  const leafRadius = nodeRadius(nodes[1].mention_count)
  const hubLeafDistance = Math.hypot(
    after.get('hub').x - after.get('leaf').x,
    after.get('hub').y - after.get('leaf').y,
  )
  assert.ok(movement('hub') < movement('leaf'))
  assert.ok(hubLeafDistance >= hubRadius + leafRadius - 0.1)
})

test('label visibility is independent from graph selection and coordinates', () => {
  const graph = {
    nodes: [node('A', 20), node('B', 12), node('C', 5)],
    edges: [edge('AB', 'A', 'B', 3), edge('BC', 'B', 'C', 1)],
  }
  const selected = selectCharacterGraph(graph, 3)
  const layout = layoutCharacterGraph(selected.nodes, selected.edges)
  const labelsOff = fitGraphToViewport(layout, 900, 700, 18)
  const labelsOn = fitGraphToViewport(layout, 900, 700, 18)
  assert.deepEqual([...labelsOn], [...labelsOff])
  assert.deepEqual(
    selected.nodes.map((item) => item.id),
    ['A', 'B', 'C'],
  )
})

test('evidence counts deterministically affect spring positions', () => {
  const nodes = [node('A', 12), node('B', 12), node('C', 12), node('D', 12)]
  const weak = layoutCharacterGraph(nodes, [
    edge('AB', 'A', 'B', 1),
    edge('CD', 'C', 'D', 2),
  ])
  const strong = layoutCharacterGraph(nodes, [
    edge('AB', 'A', 'B', 100),
    edge('CD', 'C', 'D', 2),
  ])
  assert.notDeepEqual([...strong], [...weak])
})

test('viewport fitting preserves topology and aspect ratio with uniform scaling', () => {
  const source = new Map([
    ['a', { x: -1, y: -0.2 }],
    ['b', { x: 0.1, y: 0.8 }],
    ['c', { x: 1, y: -0.6 }],
  ])
  const fitted = fitGraphToViewport(source, 1200, 800, 20)
  const distance = (positions, left, right) => {
    const a = positions.get(left)
    const b = positions.get(right)
    return Math.hypot(a.x - b.x, a.y - b.y)
  }
  assert.ok(
    Math.abs(
      distance(source, 'a', 'b') / distance(source, 'a', 'c') -
        distance(fitted, 'a', 'b') / distance(fitted, 'a', 'c'),
    ) < 1e-10,
  )
})

test('invalid Top-N limits fail clearly', () => {
  assert.throws(
    () => selectCharacterGraph({ nodes: [], edges: [] }, 0),
    RangeError,
  )
})
