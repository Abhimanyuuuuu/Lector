"""FastAPI entry point for Lector's versioned local retrieval API."""

from __future__ import annotations
import importlib.util
import logging
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .answering import (
    AnswerGenerationProvider,
    AnswerProviderError,
    GroundedAnswer,
    GenerationAuthenticationError,
    GenerationConfigurationError,
    GenerationMalformedResponseError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GroqAnswerProvider,
    generate_grounded_answer,
)
from .qa_logging import QARequestLog, QAAuditLogError
from .context import AssembledContext, ContextOptions, assemble_context
from .hybrid import HybridConfig
from .graph_retrieval import GraphRetrievalOptions
from .retrieval_types import ReadingBoundary, RetrievalResult, is_chunk_eligible
from .schemas import (
    AllBooksQueryRequest,
    AnswerCitationResponse,
    AnswerContextOptions,
    AnswerRequest,
    AnswerSourceResponse,
    BookReadingBoundary,
    ConfirmedReadingBoundary,
    ChunkResponse,
    IndexAcceptedResponse,
    IndexBookRequest,
    BookPreparationRequest,
    BookPreparationResponse,
    HybridOptions,
    GraphOptions,
    GraphReasonResponse,
    GraphExtractionResponse,
    GraphExtractionRequest,
    CharacterGraphViewRequest,
    CharacterGraphViewResponse,
    GroundedAnswerResponse,
    QueryRequest,
    RetrievalQueryResponse,
    RetrievalResultResponse,
    SavedBookListResponse,
    SavedBookResponse,
    SavedEpubResponse,
    SourcePositionResponse,
    SourceUnitInput,
)
from .semantic import SemanticUnavailableError
from .local_providers import LocalProviderError, OllamaEmbeddingProvider
from .booknlp_graph import LocalBookNLPRunner
from .persistence import ContentVersionConflictError, PersistenceIntegrityError, default_database_path
from .service import (
    BookNotIndexedError,
    ContentVersionMismatchError,
    EmbeddingIndexError,
    GraphRagDisabledError,
    InMemoryRagService,
    InvalidReadingBoundaryError,
    StoryExtractionUnavailableError,
    StoryExtractionProviderConflictError,
    StoryGraphUnavailableError,
)
from .library import InvalidEpubUpload, default_epub_directory, save_epub_upload
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _application_lifespan(application: FastAPI):
    application.state.rag_service.resume_graph_jobs()
    for job in application.state.rag_service.store.pending_book_preparations():
        _schedule_book_preparation(application, job["book_id"], job["content_version_id"], job["source_units"])
    yield
    application.state.preparation_executor.shutdown(wait=False, cancel_futures=False)

class AnswerEvidenceChangedError(RuntimeError):
    pass


def create_app(
    rag_service: InMemoryRagService | None = None,
    answer_provider: AnswerGenerationProvider | None = None,
    *,
    database_path: str | None = None,
) -> FastAPI:
    application = FastAPI(
        title="Lector RAG API",
        version="1.0.0",
        description=(
            "Local-first chunking and retrieval API. Query results are evidence "
            "candidates, not generated answers."
        ),
        lifespan=_application_lifespan,
    )
    cors_origins = [
        origin.strip()
        for origin in os.getenv(
            "LECTOR_CORS_ORIGINS",
            "http://localhost:7127,http://127.0.0.1:7127",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )
    local_base_url = os.getenv("LECTOR_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embedding_model = os.getenv("LECTOR_EMBEDDING_MODEL", "").strip()
    embedding_dimensions = os.getenv("LECTOR_EMBEDDING_DIMENSIONS", "").strip()
    embedding_provider = None
    booknlp_runner = None
    graph_enabled = os.getenv("LECTOR_BOOKNLP_ENABLED", "true").strip().casefold() in ("1", "true", "yes", "on")
    provider_status = {
        "embeddings": "unconfigured",
        "story_extraction": "not_installed" if graph_enabled else "disabled",
    }
    if embedding_model and embedding_dimensions:
        try:
            embedding_provider = OllamaEmbeddingProvider(
                base_url=local_base_url,
                model_id=embedding_model,
                dimensions=int(embedding_dimensions),
            )
            provider_status["embeddings"] = "configured"
        except (TypeError, ValueError, RuntimeError):
            provider_status["embeddings"] = "misconfigured"
    elif embedding_model or embedding_dimensions:
        provider_status["embeddings"] = "misconfigured"
    if graph_enabled:
        try:
            booknlp_runner = LocalBookNLPRunner(
                model=os.getenv("LECTOR_BOOKNLP_MODEL", "small").strip() or "small",
                model_path=os.getenv("LECTOR_BOOKNLP_MODEL_DIR") or None,
            )
            if importlib.util.find_spec("booknlp") is None:
                provider_status["story_extraction"] = "not_installed"
            elif booknlp_runner.missing_model_files:
                provider_status["story_extraction"] = "model_assets_missing"
            else:
                provider_status["story_extraction"] = "configured"
        except (TypeError, ValueError, RuntimeError):
            provider_status["story_extraction"] = "misconfigured"
    application.state.rag_service = rag_service or InMemoryRagService(
        embedding_provider=embedding_provider,
        booknlp_runner=booknlp_runner,
        database_path=database_path or default_database_path()
    )
    if rag_service is not None:
        provider_status["embeddings"] = (
            "configured" if rag_service.semantic_index.provider is not None else "unconfigured"
        )
        runner = rag_service.booknlp_runner
        if runner is None:
            provider_status["story_extraction"] = "disabled"
        elif importlib.util.find_spec("booknlp") is None:
            provider_status["story_extraction"] = "not_installed"
        elif getattr(runner, "missing_model_files", ()):
            provider_status["story_extraction"] = "model_assets_missing"
        else:
            provider_status["story_extraction"] = "configured"
    application.state.local_provider_status = provider_status
    application.state.answer_provider = (
        answer_provider if answer_provider is not None else GroqAnswerProvider()
    )
    application.state.preparation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lector-index")
    application.state.preparation_lock = Lock()
    application.state.preparation_active = set()

    @application.exception_handler(HTTPException)
    async def structured_http_error(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            error = exc.detail
        else:
            error = {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error},
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def structured_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request does not match the API schema.",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @application.get("/api/v1/health", tags=["operations"])
    def health() -> dict[str, str]:
        """Report that the HTTP application is serving requests."""
        return {
            "status": "ready",
            "service": "lector-rag-api",
            "api_version": "v1",
            "local_embeddings": application.state.local_provider_status["embeddings"],
            "local_story_extraction": application.state.local_provider_status["story_extraction"],
        }

    @application.get(
        "/api/v1/books", response_model=SavedBookListResponse, tags=["books"]
    )
    def saved_book_library() -> SavedBookListResponse:
        service: InMemoryRagService = application.state.rag_service
        entries = []
        root = default_epub_directory().resolve()
        for record in service.store.saved_book_library():
            path = (root / str(record["storage_key"])).resolve()
            available = path.parent == root and path.is_file()
            entries.append(
                SavedBookResponse(
                    book_id=record["book_id"], title=record["title"],
                    original_filename=record["original_filename"],
                    content_version_id=record["content_version_id"],
                    file_size=record["file_size"],
                    indexing_status=record["indexing_status"],
                    **_saved_preparation_fields(record.get("preparation"), record["indexing_status"]),
                    file_available=available, updated_at=record["updated_at"],
                )
            )
        return SavedBookListResponse(books=entries)

    @application.put(
        "/api/v1/books/{book_id}/epub", response_model=SavedEpubResponse,
        tags=["books"],
    )
    async def save_uploaded_epub(
        book_id: str,
        request: Request,
        content_version_id: str = Query(min_length=1),
        title: str = Query(min_length=1, max_length=500),
        original_filename: str = Query(min_length=1, max_length=1024),
    ) -> SavedEpubResponse:
        # Only a basename is persisted. Storage paths are generated by the
        # backend from book/version identity, never supplied by the browser.
        clean_filename = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not clean_filename.lower().endswith(".epub") or clean_filename in ("", ".", ".."):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_epub_filename", "message": "A valid EPUB filename is required."},
            )
        try:
            path, storage_key, size = await save_epub_upload(
                request.stream(), book_id=book_id,
                content_version_id=content_version_id,
            )
            service: InMemoryRagService = application.state.rag_service
            service.store.register_saved_epub(
                book_id=book_id, title=title.strip(),
                original_filename=clean_filename, storage_key=storage_key,
                content_version_id=content_version_id, file_size=size,
            )
        except InvalidEpubUpload as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_epub_upload", "message": str(exc)},
            ) from exc
        except sqlite3.Error as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "persistence_unavailable", "message": "The EPUB file was written locally but its library record could not be committed."},
            ) from exc
        current = service.store.saved_book(book_id)
        assert current is not None
        return SavedEpubResponse(
            book_id=book_id, content_version_id=content_version_id,
            file_size=size, indexing_status=current["indexing_status"],
        )

    @application.get("/api/v1/books/{book_id}/epub", tags=["books"])
    def download_saved_epub(book_id: str):
        service: InMemoryRagService = application.state.rag_service
        record = service.store.saved_book(book_id)
        if record is None:
            raise HTTPException(status_code=404, detail={"code": "saved_book_not_found", "message": "No saved EPUB exists for this book."})
        root = default_epub_directory().resolve()
        path = (root / str(record["storage_key"])).resolve()
        if path.parent != root or not path.is_file():
            raise HTTPException(status_code=410, detail={"code": "saved_epub_missing", "message": "The saved EPUB file is missing; its indexed text remains available."})
        return FileResponse(path, media_type="application/epub+zip", filename=str(record["original_filename"]))

    @application.delete("/api/v1/books/{book_id}", tags=["books"])
    def delete_saved_book(book_id: str) -> dict[str, object]:
        service: InMemoryRagService = application.state.rag_service
        record = service.store.saved_book(book_id)
        if record is None and service.store.get_book(book_id) is None:
            raise HTTPException(status_code=404, detail={"code":"book_not_found","message":"No stored book data exists for this book."})
        root = default_epub_directory().resolve()
        path = (root / str(record["storage_key"])).resolve() if record else None
        if path is not None and path.parent != root:
            raise HTTPException(status_code=500, detail={"code":"invalid_storage_key","message":"The saved EPUB location is invalid."})
        tombstone = None
        if path is not None and path.is_file():
            tombstone = root / f".{path.name}.deleting-{uuid.uuid4().hex}"
            os.replace(path, tombstone)
        try:
            service.delete_book(book_id)
        except Exception as exc:
            if tombstone is not None and tombstone.exists():
                os.replace(tombstone, path)
            raise HTTPException(status_code=503, detail={"code":"book_delete_failed","message":"Stored book data could not be deleted; its EPUB file was restored."}) from exc
        if tombstone is not None:
            try:
                tombstone.unlink(missing_ok=True)
            except OSError:
                logger.exception("Deleted book record but could not remove EPUB tombstone book=%s", book_id)
                raise HTTPException(status_code=503, detail={"code":"epub_cleanup_failed","message":"The book record was deleted, but the local EPUB file could not be removed."})
        return {"book_id":book_id,"deleted":True}

    @application.post(
        "/api/v1/books/{book_id}/source-units",
        status_code=202,
        response_model=IndexAcceptedResponse,
        tags=["books"],
    )
    def ingest_source_units(book_id: str, payload: IndexBookRequest) -> IndexAcceptedResponse:
        service: InMemoryRagService = application.state.rag_service
        try:
            chunks = service.ingest_book(
                book_id, payload.content_version_id, payload.source_units
            )
        except EmbeddingIndexError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "embedding_index_failed", "message": str(exc)},
            ) from exc
        except ContentVersionConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "content_version_conflict", "message": str(exc)},
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_source_units", "message": str(exc)},
            ) from exc
        except sqlite3.Error as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "persistence_unavailable",
                    "message": "The local index could not be committed; existing book data was retained.",
                },
            ) from exc
        return IndexAcceptedResponse(
            book_id=book_id,
            content_version_id=payload.content_version_id,
            accepted_source_units=len(payload.source_units),
            chunk_count=len(chunks),
        )

    @application.post(
        "/api/v1/books/{book_id}/prepare",
        status_code=202,
        response_model=BookPreparationResponse,
        tags=["books"],
    )
    def prepare_book(book_id: str, payload: BookPreparationRequest) -> BookPreparationResponse:
        service: InMemoryRagService = application.state.rag_service
        units = [item.model_dump(mode="json") for item in payload.source_units]
        try:
            service.validate_preparation_content_version(
                book_id, payload.content_version_id, payload.source_units
            )
            service.store.schedule_book_preparation(
                book_id=book_id, content_version_id=payload.content_version_id,
                source_units=units, graph_rag_enabled=payload.graph_rag_enabled,
            )
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail={"code":"persistence_unavailable","message":"Preparation could not be saved locally."}) from exc
        except ContentVersionConflictError as exc:
            raise HTTPException(status_code=409, detail={"code":"content_version_conflict","message":str(exc)}) from exc
        _schedule_book_preparation(application, book_id, payload.content_version_id, units)
        return _preparation_response(service.store.book_preparation_status(book_id, payload.content_version_id), book_id, payload.content_version_id)

    @application.get(
        "/api/v1/books/{book_id}/preparation",
        response_model=BookPreparationResponse,
        tags=["books"],
    )
    def book_preparation_status(book_id: str, content_version_id: str = Query(min_length=1)) -> BookPreparationResponse:
        service: InMemoryRagService = application.state.rag_service
        status = service.store.book_preparation_status(book_id, content_version_id)
        if status is None:
            raise HTTPException(status_code=404, detail={"code":"preparation_not_found","message":"No preparation state exists for this book revision."})
        return _preparation_response(status, book_id, content_version_id)

    @application.post(
        "/api/v1/books/{book_id}/preparation/graph/retry",
        status_code=202,
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def retry_full_book_graph(
        book_id: str, content_version_id: str = Query(min_length=1)
    ) -> GraphExtractionResponse:
        service: InMemoryRagService = application.state.rag_service
        try:
            return GraphExtractionResponse(**service.schedule_full_book_story_graph(
                book_id=book_id, content_version_id=content_version_id, retry=True,
            ))
        except Exception as exc:
            logger.exception("Full-book graph retry could not be scheduled for book %s", book_id)
            raise _http_error(exc) from exc

    @application.post(
        "/api/v1/books/{book_id}/graph/rebuild",
        status_code=202,
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def rebuild_full_book_graph(
        book_id: str, content_version_id: str = Query(min_length=1)
    ) -> GraphExtractionResponse:
        """Explicitly rebuild BookNLP data without touching canonical RAG state."""
        service: InMemoryRagService = application.state.rag_service
        try:
            return GraphExtractionResponse(**service.schedule_booknlp_graph_rebuild(
                book_id=book_id, content_version_id=content_version_id,
            ))
        except Exception as exc:
            logger.exception("Full-book graph rebuild could not be scheduled for book %s", book_id)
            raise _http_error(exc) from exc

    @application.get(
        "/api/v1/books/{book_id}/graph/rebuild/status",
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def full_book_graph_rebuild_status(
        book_id: str, content_version_id: str = Query(min_length=1)
    ) -> GraphExtractionResponse:
        service: InMemoryRagService = application.state.rag_service
        job = service.booknlp_rebuild_status(book_id=book_id, content_version_id=content_version_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "graph_rebuild_not_found", "message": "No graph rebuild has been scheduled for this book revision."})
        return GraphExtractionResponse(**job)

    @application.post(
        "/api/v1/books/{book_id}/graph/extract",
        status_code=202,
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def extract_book_graph(
        book_id: str, boundary_input: GraphExtractionRequest
    ) -> GraphExtractionResponse:
        service: InMemoryRagService = application.state.rag_service
        boundary = ReadingBoundary(
            book_id=book_id,
            section_index=boundary_input.section_index,
            unit_index=boundary_input.unit_index,
            source_unit_id=boundary_input.source_unit_id,
        )
        try:
            job = service.schedule_story_graph(
                book_id=book_id, reading_boundary=boundary,
            )
        except Exception as exc:
            logger.exception("Story graph job could not be scheduled for book %s", book_id)
            raise _http_error(exc) from exc
        return GraphExtractionResponse(**job)

    @application.post(
        "/api/v1/books/{book_id}/graph/extract/retry",
        status_code=202,
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def retry_book_graph(
        book_id: str, boundary_input: GraphExtractionRequest
    ) -> GraphExtractionResponse:
        service: InMemoryRagService = application.state.rag_service
        boundary = ReadingBoundary(
            book_id=book_id, section_index=boundary_input.section_index,
            unit_index=boundary_input.unit_index, source_unit_id=boundary_input.source_unit_id,
        )
        try:
            return GraphExtractionResponse(**service.schedule_story_graph(
                book_id=book_id, reading_boundary=boundary, retry=True,
            ))
        except Exception as exc:
            logger.exception("Story graph retry could not be scheduled for book %s", book_id)
            raise _http_error(exc) from exc

    @application.get(
        "/api/v1/books/{book_id}/graph/status",
        response_model=GraphExtractionResponse,
        tags=["books", "graph"],
    )
    def graph_status(
        book_id: str, content_version_id: str = Query(min_length=1)
    ) -> GraphExtractionResponse:
        service: InMemoryRagService = application.state.rag_service
        job = service.graph_job_status(book_id=book_id, content_version_id=content_version_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "graph_job_not_found", "message": "No graph extraction job exists for this book revision."},
            )
        return GraphExtractionResponse(**job)

    @application.post(
        "/api/v1/books/{book_id}/graph/view",
        response_model=CharacterGraphViewResponse,
        tags=["books", "graph"],
    )
    def character_graph_view(
        book_id: str, payload: CharacterGraphViewRequest
    ) -> CharacterGraphViewResponse:
        service: InMemoryRagService = application.state.rag_service
        stored = service.store.get_book(book_id)
        if stored is None:
            raise HTTPException(status_code=404, detail={"code": "book_not_indexed", "message": "Book is not indexed."})
        if stored[0] != payload.content_version_id:
            raise HTTPException(status_code=409, detail={"code": "content_version_mismatch", "message": "The graph view request is for a different book revision."})
        boundary = (
            ReadingBoundary(
                book_id=book_id, section_index=payload.section_index,
                unit_index=payload.unit_index, source_unit_id=payload.source_unit_id,
            )
            if payload.view_mode == "till_now"
            else None
        )
        try:
            return CharacterGraphViewResponse(**service.eligible_character_graph(
                book_id=book_id, reading_boundary=boundary,
                view_mode=payload.view_mode,
            ))
        except Exception as exc:
            raise _http_error(exc) from exc

    @application.post(
        "/api/v1/books/{book_id}/query",
        response_model=RetrievalQueryResponse,
        tags=["query"],
    )
    def query_book(book_id: str, payload: QueryRequest) -> RetrievalQueryResponse:
        boundary = ReadingBoundary(
            book_id=book_id,
            section_index=payload.reading_boundary.section_index,
            unit_index=payload.reading_boundary.unit_index,
            source_unit_id=payload.reading_boundary.source_unit_id,
        )
        service: InMemoryRagService = application.state.rag_service
        results = _search_book(
            service,
            book_id=book_id,
            payload=payload,
            boundary=boundary,
            hybrid_config=_hybrid_config(payload.hybrid_options),
            graph_options=_graph_options(payload.graph_options),
        )
        return _response(payload.retrieval_mode, results)

    @application.post(
        "/api/v1/query",
        response_model=RetrievalQueryResponse,
        tags=["query"],
    )
    def query_all_books(payload: AllBooksQueryRequest) -> RetrievalQueryResponse:
        boundaries = {
            item.book_id: _to_reading_boundary(item) for item in payload.reading_boundaries
        }
        service: InMemoryRagService = application.state.rag_service
        try:
            results = service.search_all_books(
                query=payload.query,
                retrieval_mode=payload.retrieval_mode,
                boundaries=boundaries,
                limit=payload.limit,
                hybrid_config=_hybrid_config(payload.hybrid_options),
                graph_options=_graph_options(payload.graph_options),
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _response(payload.retrieval_mode, results)

    @application.post(
        "/api/v1/answer",
        response_model=GroundedAnswerResponse,
        tags=["answer"],
    )
    def answer_question(payload: AnswerRequest) -> GroundedAnswerResponse:
        service: InMemoryRagService = application.state.rag_service
        if payload.book_id is not None:
            saved = service.store.get_book(payload.book_id)
            version_metadata: object = {
                "book_id": payload.book_id,
                "requested_content_version_id": payload.content_version_id,
                "indexed_content_version_id": saved[0] if saved else None,
            }
        else:
            version_metadata = {}
            for item in payload.reading_boundaries:
                saved = service.store.get_book(item.book_id)
                version_metadata[item.book_id] = saved[0] if saved else None
        boundary_metadata = (
            payload.reading_boundary.model_dump(mode="json")
            if payload.reading_boundary is not None
            else [item.model_dump(mode="json") for item in payload.reading_boundaries]
        )
        try:
            audit_log = QARequestLog.begin(
                query=payload.query,
                retrieval_mode=payload.retrieval_mode,
                book_id=payload.book_id,
                content_version_id=version_metadata,
                boundaries=boundary_metadata,
            )
        except QAAuditLogError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "qa_audit_log_unavailable", "message": str(exc)},
            ) from exc
        try:
            if payload.book_id is not None:
                assert payload.reading_boundary is not None
                boundaries = {
                    payload.book_id: ReadingBoundary(
                        book_id=payload.book_id,
                        section_index=payload.reading_boundary.section_index,
                        unit_index=payload.reading_boundary.unit_index,
                        source_unit_id=payload.reading_boundary.source_unit_id,
                    )
                }
                results = service.search_book(
                    book_id=payload.book_id,
                    query=payload.query,
                    retrieval_mode=payload.retrieval_mode,
                    reading_boundary=boundaries[payload.book_id],
                    content_version_id=payload.content_version_id,
                    limit=payload.limit,
                    hybrid_config=_hybrid_config(payload.hybrid_options),
                    graph_options=_graph_options(payload.graph_options),
                )
            else:
                boundaries = {
                    item.book_id: _to_reading_boundary(item)
                    for item in payload.reading_boundaries
                }
                results = service.search_all_books(
                    query=payload.query,
                    retrieval_mode=payload.retrieval_mode,
                    boundaries=boundaries,
                    limit=payload.limit,
                    hybrid_config=_hybrid_config(payload.hybrid_options),
                    graph_options=_graph_options(payload.graph_options),
                )

            options = _context_options(payload.context_options)
            if _is_broad_summary_question(payload.query):
                results = _summary_evidence(
                    service, boundaries, limit=options.max_passages,
                    retrieval_mode=payload.retrieval_mode,
                )
            context = _assemble_current_context(service, results, boundaries, options)
            answer = generate_grounded_answer(
                provider=application.state.answer_provider,
                question=payload.query,
                context=context,
                audit_log=audit_log,
            )
            # Re-resolve canonical content and graph eligibility after the provider
            # call. If ingestion changed evidence while generation was running,
            # fail closed instead of returning an answer against stale citations.
            try:
                current_context = _assemble_current_context(
                    service, results, boundaries, options
                )
            except Exception as exc:
                raise AnswerEvidenceChangedError(
                    "indexed evidence changed while the answer was being generated"
                ) from exc
            if current_context != context:
                raise AnswerEvidenceChangedError(
                    "indexed evidence changed while the answer was being generated"
                )
            response = _answer_response(payload.retrieval_mode, answer, context)
            audit_log.finish(
                f"{answer.status}; citations={len(answer.citations)}; "
                f"selected_passages={len(answer.selected_passage_ids)}"
            )
            logger.info(
                "QA request completed request_id=%s mode=%s status=%s selected_passages=%d",
                audit_log.request_id,
                payload.retrieval_mode,
                answer.status,
                len(answer.selected_passage_ids),
            )
            return response
        except Exception as exc:
            try:
                audit_log.finish("failed", _audit_failure_label(exc))
            except QAAuditLogError:
                logger.exception("QA request audit log finalization failed")
            logger.error(
                "QA request failed request_id=%s mode=%s category=%s",
                audit_log.request_id,
                payload.retrieval_mode,
                getattr(exc, "code", type(exc).__name__),
            )
            raise _answer_http_error(exc) from exc

    return application


def _saved_preparation_fields(preparation: object, indexing_status: str) -> dict[str, object]:
    if not isinstance(preparation, dict):
        ready = indexing_status == "indexed"
        return {"preparation_indexing_status": "complete" if ready else "pending",
                "indexing_progress": 1.0 if ready else 0.0,
                "indexing_completed": 0, "indexing_total": 0,
                "indexing_error": None,
                "embedding_status": "unavailable" if ready else "pending",
                "embedding_progress": 0.0, "embedding_completed": 0,
                "embedding_total": 0, "embedding_error": None,
                "graph_rag_enabled": False, "graph_status": "not_enabled",
                "graph_progress": 0.0, "graph_completed": 0,
                "graph_total": 0, "graph_error": None}
    indexing_total = int(preparation.get("indexing_total") or 0)
    indexing_completed = int(preparation.get("indexing_completed") or 0)
    embedding_total = int(preparation.get("embedding_total") or 0)
    embedding_completed = int(preparation.get("embedding_completed") or 0)
    graph_total = int(preparation.get("graph_total") or 0)
    graph_completed = int(preparation.get("graph_completed") or 0)
    graph_status = str(preparation.get("graph_status") or "not_enabled")
    return {
        "preparation_indexing_status": preparation.get("indexing_status") or "pending",
        "indexing_progress": 1.0 if preparation.get("indexing_status") == "complete" else (indexing_completed / indexing_total if indexing_total else 0.0),
        "indexing_completed": indexing_completed, "indexing_total": indexing_total,
        "indexing_error": preparation.get("indexing_error"),
        "embedding_status": preparation.get("embedding_status") or "pending",
        "embedding_progress": 1.0 if preparation.get("embedding_status") == "complete" else (embedding_completed / embedding_total if embedding_total else 0.0),
        "embedding_completed": embedding_completed, "embedding_total": embedding_total,
        "embedding_error": preparation.get("embedding_error"),
        "graph_rag_enabled": bool(preparation.get("graph_rag_enabled")),
        "graph_status": graph_status,
        "graph_progress": 1.0 if graph_status == "complete" else (graph_completed / graph_total if graph_total else 0.0),
        "graph_completed": graph_completed, "graph_total": graph_total,
        "graph_error": preparation.get("graph_error"),
    }


def _preparation_response(status: dict[str, object] | None, book_id: str, version: str) -> BookPreparationResponse:
    if status is None:
        raise HTTPException(status_code=404, detail={"code":"preparation_not_found","message":"No preparation state exists for this book revision."})
    return BookPreparationResponse(
        book_id=book_id, content_version_id=version,
        indexing_status=str(status.get("indexing_status", "queued")),
        embedding_status=str(status.get("embedding_status", "pending")),
        indexing_completed=int(status.get("indexing_completed") or 0),
        indexing_total=int(status.get("indexing_total") or 0),
        embedding_completed=int(status.get("embedding_completed") or 0),
        embedding_total=int(status.get("embedding_total") or 0),
        indexing_error=status.get("indexing_error"),
        embedding_error=status.get("embedding_error"),
        graph_rag_enabled=bool(status.get("graph_rag_enabled")),
        graph_status=str(status.get("graph_status") or "not_enabled"),
        graph_completed=int(status.get("graph_completed") or 0),
        graph_total=int(status.get("graph_total") or 0),
        graph_error=status.get("graph_error"),
        updated_at=str(status.get("updated_at") or ""),
    )


def _schedule_book_preparation(application: FastAPI, book_id: str, version: str, raw_units: object) -> None:
    service: InMemoryRagService = application.state.rag_service
    status = service.store.book_preparation_status(book_id, version)
    if status is None:
        return
    if status["indexing_status"] == "complete" and status["embedding_status"] in ("complete", "unavailable", "not_applicable") and status["graph_status"] in ("not_enabled", "complete", "failed", "unavailable"):
        return
    key = (book_id, version)
    with application.state.preparation_lock:
        if key in application.state.preparation_active:
            return
        application.state.preparation_active.add(key)
    application.state.preparation_executor.submit(_run_book_preparation, application, key, raw_units)


def _run_book_preparation(application: FastAPI, key: tuple[str, str], raw_units: object) -> None:
    book_id, version = key
    service: InMemoryRagService = application.state.rag_service
    started = time.monotonic()
    try:
        status = service.store.book_preparation_status(book_id, version)
        if status is None:
            return
        if status["indexing_status"] != "complete":
            units = [SourceUnitInput.model_validate(item) for item in raw_units] if isinstance(raw_units, list) else []
            if not units:
                raise ValueError("no source units are available to prepare this book")
            service.store.update_book_preparation(book_id, version, indexing_status="processing", indexing_completed=0, indexing_total=len(units), indexing_error=None, embedding_status="pending", embedding_error=None)
            def report_indexing(completed: int, total: int) -> None:
                service.store.update_book_preparation(book_id, version, indexing_completed=completed, indexing_total=total)
            chunks = service.ingest_book_keyword_stage(
                book_id, version, units, require_scheduled=True, progress=report_indexing
            )
            service.store.update_book_preparation(book_id, version, indexing_status="complete", indexing_completed=len(units), indexing_total=len(units), indexing_error=None, embedding_status="pending", embedding_completed=0, embedding_total=sum(bool(chunk.text.strip()) for chunk in chunks), embedding_error=None)
        else:
            stored = service.store.get_book(book_id)
            if stored is None or stored[0] != version:
                raise ContentVersionConflictError("indexed book content version changed")
            chunks = stored[1]
        # Graph construction starts from canonical full-book chunks, never
        # from a displayed or confirmed reading position.
        service.start_graph_preparation_if_enabled(book_id, version)
        if service.semantic_index.provider is None:
            service.store.update_book_preparation(book_id, version, embedding_status="unavailable", embedding_completed=0, embedding_total=sum(bool(chunk.text.strip()) for chunk in chunks), embedding_error="No local embedding provider is configured; keyword retrieval remains available.")
            return
        embedding_total = report_total(chunks)
        if embedding_total == 0:
            service.store.update_book_preparation(book_id, version, embedding_status="not_applicable", embedding_completed=0, embedding_total=0, embedding_error=None)
            return
        if service.semantic_is_compatible(book_id, version):
            service.store.update_book_preparation(book_id, version, embedding_status="complete", embedding_completed=embedding_total, embedding_total=embedding_total, embedding_error=None)
            return
        current = service.store.book_preparation_status(book_id, version)
        if current and current["embedding_status"] == "complete":
            return
        service.store.update_book_preparation(book_id, version, embedding_status="processing", embedding_completed=0, embedding_total=sum(bool(chunk.text.strip()) for chunk in chunks), embedding_error=None)
        def report(completed: int, total: int) -> None:
            service.store.update_book_preparation(book_id, version, embedding_completed=completed, embedding_total=total)
        service.embed_indexed_book(book_id, version, report, require_scheduled=True)
        service.store.update_book_preparation(book_id, version, embedding_status="complete", embedding_completed=embedding_total, embedding_total=embedding_total, embedding_error=None)
        logger.info("Book preparation completed book=%s version=%s chunks=%d elapsed=%.2fs", book_id, version, len(chunks), time.monotonic()-started)
    except ContentVersionConflictError as exc:
        service.store.update_book_preparation(book_id, version, indexing_status="failed", indexing_error=str(exc))
        logger.warning("Book preparation rejected book=%s version=%s reason=%s", book_id, version, exc)
    except Exception as exc:
        status = service.store.book_preparation_status(book_id, version)
        if status and status["indexing_status"] == "complete":
            service.store.update_book_preparation(book_id, version, embedding_status="failed", embedding_error=f"{type(exc).__name__}: {str(exc)[:500]}")
        else:
            service.store.update_book_preparation(book_id, version, indexing_status="failed", indexing_error=f"{type(exc).__name__}: {str(exc)[:500]}")
        logger.exception("Book preparation failed book=%s version=%s elapsed=%.2fs", book_id, version, time.monotonic()-started)
    finally:
        with application.state.preparation_lock:
            application.state.preparation_active.discard(key)


def report_total(chunks: object) -> int:
    return sum(bool(getattr(chunk, "text", "").strip()) for chunk in chunks) if isinstance(chunks, (list, tuple)) else 0


def _search_book(
    service: InMemoryRagService,
    *,
    book_id: str,
    payload: QueryRequest,
    boundary: ReadingBoundary,
    hybrid_config: HybridConfig,
    graph_options: GraphRetrievalOptions,
) -> list[RetrievalResult]:
    try:
        return service.search_book(
            book_id=book_id,
            query=payload.query,
            retrieval_mode=payload.retrieval_mode,
            reading_boundary=boundary,
            content_version_id=payload.content_version_id,
            limit=payload.limit,
            hybrid_config=hybrid_config,
            graph_options=graph_options,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, StoryExtractionProviderConflictError):
        return HTTPException(
            status_code=409,
            detail={"code": "graph_provider_conflict", "message": str(exc)},
        )
    if isinstance(exc, PersistenceIntegrityError) and "active job" in str(exc):
        return HTTPException(
            status_code=409,
            detail={"code": "graph_provider_conflict", "message": str(exc)},
        )
    if isinstance(exc, LocalProviderError):
        return HTTPException(
            status_code=503,
            detail={"code": "local_model_unavailable", "message": str(exc)},
        )
    if isinstance(exc, sqlite3.Error):
        return HTTPException(
            status_code=503,
            detail={
                "code": "persistence_unavailable",
                "message": "The local index could not be read or committed.",
            },
        )
    if isinstance(exc, BookNotIndexedError):
        return HTTPException(
            status_code=404,
            detail={"code": "book_not_indexed", "message": str(exc)},
        )
    if isinstance(exc, ContentVersionMismatchError):
        return HTTPException(
            status_code=409,
            detail={"code": "content_version_mismatch", "message": str(exc)},
        )
    if isinstance(exc, InvalidReadingBoundaryError):
        return HTTPException(
            status_code=422,
            detail={"code": "invalid_reading_boundary", "message": str(exc)},
        )
    if isinstance(exc, SemanticUnavailableError):
        return HTTPException(
            status_code=503,
            detail={"code": "semantic_unavailable", "message": str(exc)},
        )
    if isinstance(exc, EmbeddingIndexError):
        return HTTPException(
            status_code=503,
            detail={"code": "embedding_index_failed", "message": str(exc)},
        )
    if isinstance(exc, StoryExtractionUnavailableError):
        return HTTPException(
            status_code=503,
            detail={"code": "story_extraction_unavailable", "message": str(exc)},
        )
    if isinstance(exc, GraphRagDisabledError):
        return HTTPException(
            status_code=409,
            detail={"code": "graph_rag_disabled", "message": str(exc)},
        )
    if isinstance(exc, StoryGraphUnavailableError):
        return HTTPException(
            status_code=503,
            detail={"code": "story_graph_unavailable", "message": str(exc)},
        )
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPException(
            status_code=503,
            detail={"code": "retrieval_failed", "message": str(exc)},
        )
    return HTTPException(
        status_code=500,
        detail={"code": "internal_error", "message": "The retrieval request failed."},
    )


def _answer_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, QAAuditLogError):
        return HTTPException(
            status_code=503,
            detail={"code": "qa_audit_log_unavailable", "message": str(exc)},
        )
    if isinstance(exc, AnswerEvidenceChangedError):
        return HTTPException(
            status_code=409,
            detail={"code": "answer_evidence_changed", "message": str(exc)},
        )
    if isinstance(exc, GenerationConfigurationError):
        return HTTPException(
            status_code=503,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, GenerationAuthenticationError):
        return HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, GenerationRateLimitError):
        return HTTPException(
            status_code=429,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, GenerationTimeoutError):
        return HTTPException(
            status_code=504,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, (GenerationMalformedResponseError, GenerationUnavailableError)):
        return HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, AnswerProviderError):
        return HTTPException(
            status_code=502,
            detail={"code": exc.code, "message": str(exc)},
        )
    mapped = _http_error(exc)
    return mapped


def _audit_failure_label(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return f"{code}: {str(exc)}"
    return f"{type(exc).__name__}: {str(exc)}"


def _to_reading_boundary(item: BookReadingBoundary) -> ReadingBoundary:
    return ReadingBoundary(
        book_id=item.book_id,
        section_index=item.section_index,
        unit_index=item.unit_index,
        source_unit_id=item.source_unit_id,
    )


def _hybrid_config(options: HybridOptions) -> HybridConfig:
    return HybridConfig(
        k=options.k,
        keyword_weight=options.keyword_weight,
        semantic_weight=options.semantic_weight,
    )


def _graph_options(options: GraphOptions) -> GraphRetrievalOptions:
    return GraphRetrievalOptions(
        max_depth=options.max_depth,
        max_facts=options.max_facts,
        max_additional_chunks=options.max_additional_chunks,
        max_facts_per_entity=options.max_facts_per_entity,
    )


def _context_options(options: AnswerContextOptions) -> ContextOptions:
    return ContextOptions(
        max_chars=options.max_chars,
        max_passages=options.max_passages,
        max_graph_facts=options.max_graph_facts,
    )


def _assemble_current_context(
    service: InMemoryRagService,
    results: list[RetrievalResult],
    boundaries: dict[str, ReadingBoundary],
    options: ContextOptions,
) -> AssembledContext:
    chunks_by_book: dict[str, tuple] = {}
    graphs = {}
    for book_id in boundaries:
        stored = service.store.get_book(book_id)
        if stored is None:
            raise BookNotIndexedError(f"book {book_id!r} is not indexed")
        chunks_by_book[book_id] = stored[1]
        graph = service.story_graph.get_book(book_id)
        if graph is not None:
            graphs[book_id] = graph
    return assemble_context(
        results=results,
        chunks_by_book=chunks_by_book,
        graphs=graphs,
        boundaries=boundaries,
        options=options,
    )


def _is_broad_summary_question(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    return any(
        phrase in normalized
        for phrase in (
            "story so far", "summarize", "summarise", "summary of",
            "recap", "tell me the story", "what has happened so far",
            "what happened so far",
        )
    )


def _summary_evidence(
    service: InMemoryRagService,
    boundaries: dict[str, ReadingBoundary],
    *,
    limit: int,
    retrieval_mode: str,
) -> list[RetrievalResult]:
    """Sample eligible canonical chunks across the story instead of top term hits."""
    ordered = []
    for book_id in sorted(boundaries):
        stored = service.store.get_book(book_id)
        if stored is None:
            continue
        boundary = boundaries[book_id]
        ordered.extend(
            chunk for chunk in stored[1] if is_chunk_eligible(chunk, boundary)
        )
    ordered.sort(key=lambda chunk: (
        chunk.book_id, chunk.section_index, chunk.chunk_index, chunk.id
    ))
    if len(ordered) <= limit:
        selected = ordered
    elif limit == 1:
        selected = [ordered[-1]]
    else:
        indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
        selected = [ordered[index] for index in dict.fromkeys(indices)]
    return [
        RetrievalResult(chunk=chunk, score=0.0, retrieval_method=retrieval_mode)
        for chunk in selected
    ]


def _answer_response(
    retrieval_mode: str,
    answer: GroundedAnswer,
    context: AssembledContext,
) -> GroundedAnswerResponse:
    citations = answer.citations
    return GroundedAnswerResponse(
        status=answer.status,
        answer=answer.answer,
        uncertainty=answer.uncertainty,
        retrieval_mode=retrieval_mode,
        citations=[
            AnswerCitationResponse(
                passage_id=item.passage.passage_id,
                book_id=item.passage.chunk.book_id,
                content_version_id=item.passage.chunk.content_version_id,
                section_index=item.passage.chunk.section_index,
                chunk_index=item.passage.chunk.chunk_index,
                source_unit_ids=list(item.passage.chunk.source_unit_ids),
                source_positions=[
                    SourcePositionResponse.model_validate(asdict(position))
                    for position in item.passage.chunk.source_positions
                ],
                href=item.passage.chunk.href,
                start_cfi=item.passage.chunk.start_cfi,
                end_cfi=item.passage.chunk.end_cfi,
                cfi_status=item.passage.chunk.cfi_status,
                excerpt=item.passage.chunk.text,
            )
            for item in citations
        ],
        sources=[
            AnswerSourceResponse(
                passage_id=passage.passage_id,
                score=passage.score,
                retrieval_method=passage.retrieval_method,
                keyword_rank=passage.keyword_rank,
                keyword_score=passage.keyword_score,
                semantic_rank=passage.semantic_rank,
                semantic_score=passage.semantic_score,
                graph_score=passage.graph_score,
                graph_depth=passage.graph_depth,
            )
            for passage in context.passages
        ],
        graph_fact_ids=[fact.fact_id for fact in context.graph_facts],
        selected_passage_ids=list(answer.selected_passage_ids),
    )


def _response(
    retrieval_mode: str, results: list[RetrievalResult]
) -> RetrievalQueryResponse:
    return RetrievalQueryResponse(
        retrieval_mode=retrieval_mode,
        results=[
            RetrievalResultResponse(
                chunk=ChunkResponse.model_validate(asdict(result.chunk)),
                score=result.score,
                retrieval_method=result.retrieval_method,
                matched_terms=list(result.matched_terms),
                keyword_rank=result.keyword_rank,
                keyword_score=result.keyword_score,
                semantic_rank=result.semantic_rank,
                semantic_score=result.semantic_score,
                graph_score=result.graph_score,
                graph_depth=result.graph_depth,
                graph_reasons=[
                    GraphReasonResponse.model_validate(asdict(reason))
                    for reason in result.graph_reasons
                ],
            )
            for result in results
        ],
    )


# Keep module import side-effect-free for test discovery and WSGI inspection.
# Production startup uses the factory so it gets the durable default database.
app = create_app(database_path=":memory:")
