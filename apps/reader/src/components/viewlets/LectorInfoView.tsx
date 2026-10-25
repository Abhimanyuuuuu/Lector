import React from 'react'

import { PaneView, PaneViewProps } from '../base'

export const LectorInfoView: React.FC<PaneViewProps> = (props) => (
  <PaneView {...props}>
    <div className="scroll-parent rag-panel-scroll space-y-3 px-5 pb-5">
      <h2 className="text-lg font-semibold">Using Lector</h2>
      <ol className="list-decimal space-y-2 pl-5">
        <li>
          <strong>Import a book.</strong> Lector asks whether to enable GraphRAG
          for that book. Indexing and local embeddings prepare automatically.
        </li>
        <li>
          <strong>Build story memory.</strong> For opted-in books, local BookNLP
          builds character profiles and candidate relationships from the full
          book in the background. It does not use Qwen or Groq for graph
          construction.
        </li>
        <li>
          <strong>Explore characters.</strong> Open the graph from the sidebar.
          It shows only characters connected by currently eligible
          relationships. Hover a character for its boundary-filtered aliases and
          pronouns, or hover a connection for its supporting passage and
          candidate actions.
        </li>
        <li>
          <strong>Confirm reading progress.</strong> Graph construction covers
          the book, but displayed profiles and graph retrieval stop at your last
          confirmed position. Navigation alone never confirms progress.
        </li>
        <li>
          <strong>Ask questions.</strong> Keyword, Semantic, and Hybrid retrieve
          passages. Hybrid + GraphRAG combines normal Hybrid retrieval with
          relevant, boundary-eligible story evidence.
        </li>
      </ol>
      <p className="text-outline text-sm">
        Graph connections are candidate heuristics based on sentence co-mentions
        and verbs; they do not prove a direct relationship. GraphRAG is optional
        for each book. Book text and selected eligible evidence are sent to Groq
        only for final answer generation when configured.
      </p>
    </div>
  </PaneView>
)
