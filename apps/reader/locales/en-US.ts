export default {
  'home.title': 'Home',
  'home.share': 'Share',
  'home.download': 'Download',
  'home.download_sample_book': 'Download sample book',
  'home.select': 'Select',
  'home.cancel': 'Cancel',
  'home.select_all': 'Select all',
  'home.deselect_all': 'Deselect all',
  'home.export': 'Export',
  'home.import': 'Import',
  'home.upload': 'Upload',
  'home.delete': 'Delete',

  'toc.title': 'Table of Contents',
  'toc.contents': 'Contents',
  'toc.library': 'Library',

  'search.title': 'Search',
  'search.files.result': '{n} results in {m} sections',

  'annotation.title': 'Annotation',
  'annotation.definitions': 'Definitions',
  'annotation.annotations': 'Annotations',
  'annotation.copy_as_markdown': 'Copy as Markdown',

  'image.title': 'Image',

  'timeline.title': 'Timeline',

  'typography.title': 'Typography',
  'typography.scope.book': 'Book',
  'typography.scope.global': 'Global',
  'typography.page_view': 'Page View',
  'typography.page_view.single_page': 'Single Page',
  'typography.page_view.double_page': 'Double Page',
  'typography.font_family': 'Font Family',
  'typography.font_size': 'Font Size',
  'typography.font_weight': 'Font Weight',
  'typography.line_height': 'Line Height',
  'typography.zoom': 'Zoom',

  'theme.title': 'Theme',
  'lector.title': 'Lector',
  'lector-info.title': 'Lector guide',
  'lector.open_book': 'Open a book to use Lector.',
  'lector.reading_boundary': 'Confirmed reading boundary',
  'lector.index_first': 'Index this book before confirming a boundary.',
  'lector.boundary_unavailable':
    'The current location could not be mapped to a complete source paragraph. Move within a reliably mapped passage and try again.',
  'lector.navigation_unavailable':
    'This citation has no usable CFI or known section location.',
  'lector.index': 'Index this book',
  'lector.indexing': 'Indexing section {current} of {total}…',
  'lector.uploading': 'Sending source text to the local Lector backend…',
  'lector.indexed': 'Indexed {units} source units into {chunks} chunks.',
  'lector.partial':
    'The book was not fully indexed; no partial result was committed.',
  'lector.confirmed': 'Confirmed boundary: {section}, paragraph {paragraph}.',
  'lector.no_boundary':
    'Confirm a reading boundary before asking questions or extracting story facts.',
  'lector.confirm_description':
    'Confirming covers every source unit up to this location, including earlier sections. If you skipped content, confirm an earlier point instead.',
  'lector.confirm_read': 'I have read all content through this point.',
  'lector.confirm_progress':
    'Confirm through the last completed paragraph before the current location',
  'lector.ask': 'Ask about this book',
  'lector.question': 'Question',
  'lector.send': 'Ask',
  'lector.mode': 'Retrieval mode',
  'lector.loading': 'Searching eligible book evidence…',
  'lector.retry': 'Retry',
  'lector.navigate_exact': 'Open cited passage',
  'lector.navigate_section':
    'Open cited section (exact paragraph location unavailable)',
  'lector.citations': 'Citations',
  'lector.graph_facts': 'Supporting graph fact IDs',
  'lector.uncertainty': 'Uncertainty',
  'lector.insufficient': 'The available evidence was insufficient to answer.',
  'lector.backend_required':
    'Start the local Python RAG API to index or ask questions.',
  'lector.index_not_ready':
    'Wait until all EPUB spine sections have loaded before indexing.',
  'lector.extract_failed':
    'Extraction failed before upload. The book was not indexed.',
  'lector.boundary_invalidated':
    'The EPUB content changed. Confirm reading progress again.',
  'lector.graph_ready':
    'Story graph updated with {entities} entities, {events} events, and {relationships} relationships.',
  'lector.graph_unavailable':
    'Story extraction is unavailable. Keyword retrieval can still be used.',
  'theme.source_color': 'Source Color',
  'theme.background_color': 'Background Color',

  'settings.title': 'Settings',
  'settings.language': 'Language',
  'settings.color_scheme': 'Color Scheme',
  'settings.color_scheme.system': 'System',
  'settings.color_scheme.light': 'Light',
  'settings.color_scheme.dark': 'Dark',
  'settings.synchronization.title': 'Synchronization',
  'settings.synchronization.authorize': 'Authorize',
  'settings.synchronization.unauthorize': 'Unauthorize',
  'settings.cache': 'Cache',
  'settings.cache.clear': 'Clear',
  'settings.text_selection_menu': 'Text Selection Menu',
  'settings.text_selection_menu.enable': 'Enable',

  'menu.copy': 'Copy',
  'menu.search_in_book': 'Search in book',
  'menu.annotate': 'Annotate',
  'menu.define': 'Define',
  'menu.undefine': 'Undefine',
  'menu.create': 'Create',
  'menu.delete': 'Delete',
  'menu.update': 'Update',

  'action.expand_all': 'Expand All',
  'action.collapse_all': 'Collapse All',
  'action.close': 'Close',
  'action.clear': 'Clear',
  'action.step_down': 'Step Down',
  'action.step_up': 'Step Up',

  untitled: 'Untitled',
} as const
