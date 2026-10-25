const path = require('path')

const epubjsRoot = path.resolve(__dirname, '../../../../packages/epubjs')
const epubjsModules = path.join(epubjsRoot, 'node_modules')
const webpackConfig = require(path.join(epubjsRoot, 'webpack.config.js'))
const testFile = path.resolve(__dirname, 'extraction.cfi.test.js')

webpackConfig.mode = 'development'
webpackConfig.externals = {}
webpackConfig.module.rules.push({
  test: /\.xhtml$/i,
  use: 'raw-loader',
})

module.exports = function (config) {
  config.set({
    basePath: epubjsRoot,
    frameworks: ['mocha'],
    plugins: [
      require(path.join(epubjsModules, 'karma-chrome-launcher')),
      require(path.join(epubjsModules, 'karma-mocha')),
      require(path.join(epubjsModules, 'karma-mocha-reporter')),
      require(path.join(epubjsModules, 'karma-sourcemap-loader')),
      require(path.join(epubjsModules, 'karma-webpack')),
    ],
    files: [
      { pattern: testFile, watched: false },
      {
        pattern: 'test/fixtures/**/*',
        watched: false,
        included: false,
        served: true,
      },
    ],
    preprocessors: {
      [testFile]: ['webpack', 'sourcemap'],
    },
    webpack: webpackConfig,
    webpackMiddleware: { stats: 'errors-only' },
    reporters: ['mocha'],
    browsers: ['ChromeHeadlessNoSandbox'],
    singleRun: true,
    autoWatch: false,
    proxies: {
      '/fixtures/': '/base/test/fixtures/',
    },
    customLaunchers: {
      ChromeHeadlessNoSandbox: {
        base: 'ChromeHeadless',
        flags: ['--no-sandbox'],
      },
    },
  })
}
