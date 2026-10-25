export default {
  'home.title': 'Startseite',
  'home.share': 'Teilen',
  'home.download': 'Herunterladen',
  'home.download_sample_book': 'Beispielbuch herunterladen',
  'home.select': 'Auswählen',
  'home.cancel': 'Abbrechen',
  'home.select_all': 'Alle auswählen',
  'home.deselect_all': 'Alle abwählen',
  'home.export': 'Exportieren',
  'home.import': 'Importieren',
  'home.upload': 'Hochladen',
  'home.delete': 'Löschen',

  'toc.title': 'Inhaltsverzeichnis',
  'toc.contents': 'Inhalt',
  'toc.library': 'Bibliothek',

  'search.title': 'Suchen',
  'search.files.result': '{n} Ergebnisse in {m} Abschnitten',

  'annotation.title': 'Anmerkung',
  'annotation.definitions': 'Definitionen',
  'annotation.annotations': 'Anmerkungen',
  'annotation.copy_as_markdown': 'Als Markdown kopieren',

  'image.title': 'Bild',

  'timeline.title': 'Zeitleiste',

  'typography.title': 'Typografie',
  'typography.scope.book': 'Buch',
  'typography.scope.global': 'Global',
  'typography.page_view': 'Seitenansicht',
  'typography.page_view.single_page': 'Einzelseite',
  'typography.page_view.double_page': 'Doppelseite',
  'typography.font_family': 'Schriftart',
  'typography.font_size': 'Schriftgröße',
  'typography.font_weight': 'Schriftstärke',
  'typography.line_height': 'Zeilenhöhe',
  'typography.zoom': 'Zoom',

  'theme.title': 'Design',
  'lector.title': 'Lector',
  'lector-info.title': 'Lector-Hilfe',
  'lector.open_book': 'Öffne ein Buch, um Lector zu verwenden.',
  'lector.reading_boundary': 'Bestätigte Leseposition',
  'lector.index_first':
    'Indexiere dieses Buch, bevor du eine Leseposition bestätigst.',
  'lector.boundary_unavailable':
    'Die aktuelle Position konnte keinem vollständigen Quellabsatz zugeordnet werden. Versuche es an einer eindeutig zugeordneten Textstelle erneut.',
  'lector.navigation_unavailable':
    'Für diesen Beleg gibt es weder einen gültigen CFI noch einen bekannten Abschnitt.',
  'lector.index': 'Buch indexieren',
  'lector.indexing': 'Abschnitt {current} von {total} wird indexiert…',
  'lector.uploading': 'Text wird an das lokale Lector-Backend gesendet…',
  'lector.indexed': '{units} Quelleinheiten in {chunks} Textblöcke indexiert.',
  'lector.partial':
    'Das Buch wurde nicht vollständig indexiert; ein Teilergebnis wurde nicht gespeichert.',
  'lector.confirmed': 'Bestätigte Grenze: {section}, Absatz {paragraph}.',
  'lector.no_boundary': 'Bestätige zuerst eine Leseposition.',
  'lector.confirm_description':
    'Die Bestätigung umfasst alle Inhalte bis zu dieser Position, auch vorherige Abschnitte. Bei übersprungenen Inhalten bestätige eine frühere Position.',
  'lector.confirm_read':
    'Ich habe alle Inhalte bis zu dieser Position gelesen.',
  'lector.confirm_progress':
    'Bis zum letzten abgeschlossenen Absatz vor der aktuellen Position bestätigen',
  'lector.ask': 'Frage zu diesem Buch',
  'lector.question': 'Frage',
  'lector.send': 'Fragen',
  'lector.mode': 'Suchmodus',
  'lector.loading': 'Zulässige Textbelege werden gesucht…',
  'lector.retry': 'Erneut versuchen',
  'lector.navigate_exact': 'Zitierte Stelle öffnen',
  'lector.navigate_section':
    'Abschnitt öffnen (genaue Absatzposition nicht verfügbar)',
  'lector.citations': 'Quellenangaben',
  'lector.graph_facts': 'Unterstützende Graph-Fakten-IDs',
  'lector.uncertainty': 'Unsicherheit',
  'lector.insufficient':
    'Die verfügbaren Belege reichen für eine Antwort nicht aus.',
  'lector.backend_required': 'Starte die lokale Python-RAG-API.',
  'lector.index_not_ready': 'Warte, bis alle EPUB-Abschnitte geladen sind.',
  'lector.extract_failed':
    'Extraktion fehlgeschlagen; das Buch wurde nicht indexiert.',
  'lector.boundary_invalidated':
    'Der EPUB-Inhalt wurde geändert. Bestätige den Lesefortschritt erneut.',
  'lector.graph_ready':
    'Graph mit {entities} Entitäten, {events} Ereignissen und {relationships} Beziehungen aktualisiert.',
  'lector.graph_unavailable':
    'Story-Extraktion nicht verfügbar. Stichwortsuche ist weiterhin möglich.',
  'theme.source_color': 'Quellfarbe',
  'theme.background_color': 'Hintergrundfarbe',

  'settings.title': 'Einstellungen',
  'settings.language': 'Sprache',
  'settings.color_scheme': 'Farbschema',
  'settings.color_scheme.system': 'System',
  'settings.color_scheme.light': 'Hell',
  'settings.color_scheme.dark': 'Dunkel',
  'settings.synchronization.title': 'Synchronisation',
  'settings.synchronization.authorize': 'Autorisieren',
  'settings.synchronization.unauthorize': 'Deautorisieren',
  'settings.cache': 'Cache',
  'settings.cache.clear': 'Leeren',
  'settings.text_selection_menu': 'Textauswahlmenü',
  'settings.text_selection_menu.enable': 'Aktivieren',

  'menu.copy': 'Kopieren',
  'menu.search_in_book': 'Im Buch suchen',
  'menu.annotate': 'Kommentieren',
  'menu.define': 'Definieren',
  'menu.undefine': 'Definition entfernen',
  'menu.create': 'Erstellen',
  'menu.delete': 'Löschen',
  'menu.update': 'Aktualisieren',

  'action.expand_all': 'Alle erweitern',
  'action.collapse_all': 'Alle einklappen',
  'action.close': 'Schließen',
  'action.clear': 'Leeren',
  'action.step_down': 'Schritt zurück',
  'action.step_up': 'Schritt vor',

  untitled: 'Ohne Titel',
} as const
