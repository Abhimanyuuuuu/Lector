const assert = require('node:assert/strict')
const fs = require('node:fs')
const Module = require('node:module')
const path = require('node:path')
const ts = require('typescript')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')

const componentPath = path.resolve(
  __dirname,
  '../../src/components/viewlets/RagPreparationProgress.tsx',
)
const compiled = ts.transpileModule(fs.readFileSync(componentPath, 'utf8'), {
  compilerOptions: {
    esModuleInterop: true,
    jsx: ts.JsxEmit.React,
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2019,
  },
})
const componentModule = new Module(componentPath, module)
componentModule.filename = componentPath
componentModule.paths = Module._nodeModulePaths(path.dirname(componentPath))
componentModule._compile(compiled.outputText, componentPath)

const book = {
  preparation_indexing_status: 'complete',
  indexing_progress: 1,
  indexing_completed: 438,
  indexing_total: 438,
  embedding_status: 'not_applicable',
  embedding_progress: 1,
  embedding_completed: 0,
  embedding_total: 0,
  graph_status: 'unavailable',
  graph_progress: 0,
  graph_completed: 0,
  graph_total: 438,
  graph_error: 'Required local model assets are missing.',
}
const html = renderToStaticMarkup(
  React.createElement(componentModule.exports.RagPreparationProgress, {
    book: { ...book, title: 'Legend.epub' },
    title: React.createElement('span', {}, 'Legend.epub'),
  }),
)

assert.match(html, /aria-label="Story graph: unavailable"/)
assert.match(html, /aria-expanded="true"/)
assert.match(html, />Legend\.epub<\/span>/)
assert.match(html, /aria-label="Collapse preparation details for Legend\.epub"/)
assert.doesNotMatch(html, /Processing status/)
assert.match(html, />Indexing<\/span>/)
assert.match(html, />Embeddings<\/span>/)
assert.match(html, />Story graph<\/span>/)
assert.match(html, /title="Required local model assets are missing\."/)
assert.match(html, />unavailable<\/span>/)
assert.doesNotMatch(html, /0\/438/)
assert.doesNotMatch(html, /animate-pulse/)

const originalUseState = React.useState
let expanded = true
React.useState = () => [
  expanded,
  (update) => {
    expanded = typeof update === 'function' ? update(expanded) : update
  },
]
const props = {
  book: { ...book, title: 'Legend.epub' },
  title: React.createElement('span', {}, 'Legend.epub'),
}
const findButton = (element) => {
  if (!element || typeof element !== 'object') return undefined
  if (element.type === 'button') return element
  const children = element.props?.children
  for (const child of Array.isArray(children) ? children : [children]) {
    const button = findButton(child)
    if (button) return button
  }
  return undefined
}

try {
  let tree = componentModule.exports.RagPreparationProgress(props)
  const disclosure = findButton(tree)
  assert.equal(disclosure.props['aria-expanded'], true)
  let stopped = false
  disclosure.props.onClick({
    stopPropagation: () => {
      stopped = true
    },
  })
  assert.equal(stopped, true)
  assert.equal(expanded, false)

  tree = componentModule.exports.RagPreparationProgress(props)
  assert.equal(findButton(tree).props['aria-expanded'], false)
  assert.equal(tree.props.children[1], false)
} finally {
  React.useState = originalUseState
}

process.stdout.write('RAG preparation progress terminal-state test passed.\n')
