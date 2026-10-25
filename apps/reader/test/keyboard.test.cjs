const assert = require('node:assert/strict')
const {
  shouldNavigateReaderWithKey,
} = require('../.turbo/lector-keyboard-tests/keyboard.js')

const page = {
  isContentEditable: false,
  closest: () => null,
}

const textEntry = (selector) => ({
  isContentEditable: false,
  closest: (query) => (query.split(', ').includes(selector) ? {} : null),
})

assert.equal(shouldNavigateReaderWithKey('ArrowLeft', page), true)
assert.equal(shouldNavigateReaderWithKey('ArrowRight', page), true)
assert.equal(shouldNavigateReaderWithKey('ArrowUp', page), true)
assert.equal(shouldNavigateReaderWithKey('ArrowDown', page), true)

for (const code of ['Space', 'KeyW', 'KeyA', 'KeyS', 'KeyD']) {
  assert.equal(shouldNavigateReaderWithKey(code, page), false, code)
}

for (const code of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown']) {
  for (const selector of [
    'input',
    'textarea',
    'select',
    '[role="textbox"]',
    '[contenteditable]:not([contenteditable="false"])',
  ]) {
    assert.equal(
      shouldNavigateReaderWithKey(code, textEntry(selector)),
      false,
      `${code} in ${selector}`,
    )
  }
}

assert.equal(
  shouldNavigateReaderWithKey('ArrowLeft', {
    isContentEditable: true,
    closest: () => null,
  }),
  false,
)
assert.equal(shouldNavigateReaderWithKey('ArrowLeft', null), false)

console.log('Reader keyboard navigation tests passed')
