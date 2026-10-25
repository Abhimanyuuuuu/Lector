# Lector

**A browser-based EPUB reader with AI-powered story understanding.**

Lector combines a reading experience with Retrieval-Augmented Generation (RAG), allowing readers to ask questions about a book and retrieve relevant passages from its content. It also builds a character and story graph to provide additional narrative context.

## Features

- **EPUB reader:** Read books in a browser-based interface.
- **AI-powered book Q&A:** Ask questions about the current book using relevant retrieved passages.
- **Hybrid retrieval:** Combine semantic and keyword search to find relevant text.
- **Character graph:** Explore characters and their relationships using extracted story information.
- **Story context:** Use graph information alongside retrieved passages to help answer questions about the narrative.
- **Reading progress:** Resume reading from your saved position.
- **Reading customization:** Customize typography, themes, and other reading preferences.
- **Library and annotations:** Manage books and use supported reading and annotation features.

## Technology

- **Frontend:** React, Next.js, TypeScript
- **EPUB rendering:** epub.js
- **Backend:** Python and FastAPI
- **Retrieval:** Keyword search, semantic search, and hybrid retrieval
- **Embeddings:** Ollama with `nomic-embed-text`
- **Answer generation:** Groq
- **Graph extraction:** BookNLP
- **Persistence:** SQLite

## Getting started

### Prerequisites

- Git
- Node.js
- pnpm
- Python 3.11 or later
- Ollama
- A Groq API key for AI-generated answers

### Setup

Clone your repository:

```bash
git clone https://github.com/Abhimanyuuuuu/Lector.git
cd Lector
```

Install the JavaScript dependencies:

```bash
pnpm install
```

Run the project setup script in PowerShell:

```powershell
.\Setup-Lector.ps1
```

Follow the prompts to configure the environment and required services.

### Run Lector

Use the Windows launcher:

```powershell
.\Start-Lector.ps1
```

Alternatively, start the development apps using the repository's configured pnpm scripts:

```bash
pnpm dev
```

The RAG backend requires its own Python environment and configuration. See [`apps/rag-api/README.md`](apps/rag-api/README.md) and [`docs/lector-python-backend.md`](docs/lector-python-backend.md) for details.

## Project structure

- `apps/reader` — EPUB reader and frontend RAG integration
- `apps/rag-api` — Python API, retrieval, book processing, and story graph
- `apps/website` — Project website
- `packages/epubjs` — EPUB rendering library
- `docs` — Project and backend documentation

## License

Lector is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](LICENSE).

This project builds on [Flow](https://github.com/pacexy/flow) and uses [epub.js](https://github.com/futurepress/epub.js/) and other open-source software. Please refer to the relevant third-party licenses and notices.