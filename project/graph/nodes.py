import logging
import re
from uuid import UUID

from langchain_core.messages import AIMessage

from project.application.retrieval.vector_index_cache import VectorIndexCache
from project.application.retrieval.evidence import (
    chunk_from_record,
    chunk_to_record,
    result_from_record,
    result_to_record,
)
from project.application.papers.prepare_paper import PaperPreparationService
from project.graph.state import AgentState
from project.infrastructure.database.papers import (
    add_paper_to_session,
    get_paper_artifact,
    get_paper_content,
    get_session_papers,
    register_paper,
)
from project.nodes.answer import AnswerBuilder
from project.nodes.arxiv import ArxivClient, ArxivError
from project.nodes.llm import LLMError, OllamaLLM
from project.nodes.query import QueryProcessor
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore
from project.processing.chunker import TextChunk


logger = logging.getLogger(__name__)


def process_query(state: AgentState, query_processor: QueryProcessor):
    try:
        processed_query = query_processor.process(state["query"])
    except LLMError:
        logger.exception("Query processing failed.")
        return {"error": "Failed to process query"}
    except (TypeError, ValueError) as error:
        return {"error": str(error)}
    except Exception:
        logger.exception("Unexpected query processing failure.")
        return {"error": "Unexpected error while processing query"}

    return {"processed_query": processed_query}


def route_after_query(state: AgentState):
    if not state.get("processed_query"):
        return "error"

    if state.get("paper"):
        return "search_paper"

    # A resumed thread has its current paper and serializable chunks restored
    # from the checkpoint.  Continue against the session rather than silently
    # initiating a new single-paper/topic search.
    if state.get("session_id") and state.get("paper_id") and state.get("chunks"):
        return "retrieve_chunks"

    return "search_arxiv"


def search_arxiv(state: AgentState, arxiv_client: ArxivClient):
    try:
        papers = arxiv_client.search_by_topic(state["processed_query"])
    except ArxivError:
        logger.exception("arXiv search failed.")
        return {"papers": [], "error": "Failed to search arXiv"}
    except Exception:
        logger.exception("Unexpected arXiv search failure.")
        return {"papers": [], "error": "Unexpected error while searching arXiv"}

    if not papers:
        return {"papers": [], "error": "No papers found for the query"}

    return {"papers": papers}


def select_paper(state: AgentState):
    papers = state["papers"]

    if not papers:
        raise ValueError("No papers available for selection")

    query = state["processed_query"].lower()
    query_terms = set(re.findall(r"\b[a-z0-9]+\b", query))

    best_paper = papers[0]
    best_score = -1

    for paper in papers:
        text = f"{paper.title} {paper.abstract}".lower()
        terms = set(re.findall(r"\b[a-z0-9]+\b", text))
        score = len(query_terms & terms)

        if score > best_score:
            best_score = score
            best_paper = paper

    return {"selected_paper": best_paper}


def prepare_paper(
    state: AgentState,
    preparation_service: PaperPreparationService,
):
    paper = state["selected_paper"]

    try:
        paper_record = register_paper({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published": paper.published,
            "updated": paper.updated,
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "categories": paper.categories,
        })
    except (ValueError, KeyError) as error:
        logger.exception("Paper registration failed.")
        return {"error": f"Failed to register paper: {error}"}
    except Exception:
        logger.exception("Unexpected database error during paper registration.")
        return {"error": "Failed to register paper"}

    paper_id = paper_record["id"]

    try:
        result = preparation_service.prepare(
            pdf_url=paper.pdf_url,
            arxiv_id=paper.arxiv_id,
            paper_id=paper_id,
        )
    except Exception:
        logger.exception("Unexpected coordinated paper preparation failure.")
        return {"error": "Failed to prepare paper"}

    if result.error:
        return {"error": result.error}

    session_id = state.get("session_id")

    if session_id:
        try:
            add_paper_to_session(
                session_id=UUID(str(session_id)),
                paper_id=UUID(str(paper_id)),
                is_active=True,
            )
        except Exception:
            logger.exception("Failed to associate paper with session.")
            return {"error": "Failed to associate paper with session"}

    return {
        "pdf_path": result.pdf_path,
        "paper_id": str(paper_id),
        "chunks": [chunk_to_record(chunk) for chunk in result.chunks or []],
        "content_fingerprint": result.content_fingerprint,
        "chunking_fingerprint": result.chunking_fingerprint,
    }


def parse_and_chunk(
    state: AgentState,
    preparation_service: PaperPreparationService,
):
    paper = state["selected_paper"]

    try:
        result = preparation_service.parse_and_chunk(
            pdf_path=state["pdf_path"],
            arxiv_id=paper.arxiv_id,
            paper_id=state["paper_id"],
        )
    except Exception:
        logger.exception("Unexpected parse and chunk service failure.")
        return {"chunks": [], "error": "Failed to process paper content"}

    if result.error:
        return {
            "chunks": [chunk_to_record(chunk) for chunk in result.chunks or []],
            "error": result.error,
        }

    return {
        "chunks": [chunk_to_record(chunk) for chunk in result.chunks or []],
        "content_fingerprint": result.content_fingerprint,
        "chunking_fingerprint": result.chunking_fingerprint,
    }


def retrieve_chunks(
    state: AgentState,
    vector_store: VectorStore,
    retriever: Retriever,
    vector_index_cache: VectorIndexCache,
    candidates_per_paper: int = 5,
    final_top_k: int = 5,
):
    session_id = state.get("session_id")
    try:
        session_papers = get_session_papers(UUID(str(session_id))) if session_id else []
    except Exception:
        logger.exception("Failed to resolve session papers | session_id=%s", session_id)
        return {"results": [], "error": "Failed to resolve the session's papers"}

    current_paper_id = str(state["paper_id"])
    if not session_papers:
        selected = state["selected_paper"]
        session_papers = [{
            "id": current_paper_id,
            "arxiv_id": selected.arxiv_id,
            "title": selected.title,
        }]
    logger.info("Session retrieval papers resolved | session_id=%s | papers=%s", session_id, [str(p["id"]) for p in session_papers])

    candidates = []
    for paper in session_papers:
        paper_id = str(paper["id"])
        try:
            if paper_id == current_paper_id:
                chunks = [chunk_from_record(chunk) for chunk in state.get("chunks", [])]
                content_fingerprint = state.get("content_fingerprint")
                chunking_fingerprint = state.get("chunking_fingerprint")
            else:
                content = get_paper_content(UUID(paper_id))
                artifact = get_paper_artifact(UUID(paper_id)) or {}
                chunks = [TextChunk(chunk_id=row["chunk_id"], page=row["page_number"], text=row["text"]) for row in content["chunks"]]
                content_fingerprint = artifact.get("content_fingerprint")
                chunking_fingerprint = artifact.get("chunking_fingerprint")
            if not chunks:
                logger.warning("Paper retrieval skipped | paper_id=%s | reason=no_chunks", paper_id)
                continue
            vector_index_cache.get_or_build_index(
                paper_id=paper_id, chunks=chunks, vector_store=vector_store,
                content_fingerprint=content_fingerprint, chunking_fingerprint=chunking_fingerprint,
            )
            paper_results = vector_store.search(state["processed_query"], top_k=candidates_per_paper)
            for result in paper_results:
                result.paper_id, result.arxiv_id, result.paper_title = paper_id, paper["arxiv_id"], paper["title"]
            candidates.extend(paper_results)
            logger.info("Paper retrieval completed | paper_id=%s | arxiv_id=%s | candidates=%d", paper_id, paper["arxiv_id"], len(paper_results))
        except Exception:
            logger.exception("Paper retrieval failed | paper_id=%s | session_id=%s", paper_id, session_id)

    deduplicated = []
    seen = set()
    for result in candidates:
        key = (result.paper_id, result.chunk.chunk_id)
        if key not in seen:
            seen.add(key)
            deduplicated.append(result)
    logger.info("Session candidates merged | total=%d | deduplicated=%d", len(candidates), len(deduplicated))
    if not deduplicated:
        return {"results": [], "error": "No relevant information was found in the session's papers"}

    ranked = retriever.reranker.rerank(state["processed_query"], deduplicated) if retriever.reranker else sorted(deduplicated, key=lambda result: result.score, reverse=True)
    final_results = ranked[:final_top_k]
    logger.info("Global reranking completed | candidates=%d | final_hits=%d", len(deduplicated), len(final_results))
    for rank, result in enumerate(final_results, 1):
        logger.info("Final evidence | rank=%d | paper_id=%s | chunk_id=%s", rank, result.paper_id, result.chunk.chunk_id)
    return {"results": [result_to_record(result) for result in final_results]}


def generate_answer(state: AgentState, llm: OllamaLLM):
    try:
        messages = state.get("messages", [])

        answer = llm.generate_answer(
            state["query"],
            [result_from_record(result) for result in state["results"]],
            messages[:-1],
        )
    except LLMError:
        logger.exception("Answer generation failed.")
        return {"error": "Failed to generate answer"}
    except Exception:
        logger.exception("Unexpected answer generation failure.")
        return {"error": "Unexpected error while generating answer"}

    return {"answer": answer}


def route_after_search(state: AgentState):
    if state.get("papers"):
        return "select_paper"

    return "error"


def handle_error(state: AgentState):
    return {"answer": state.get("error", "No relevant papers were found.")}


def build_answer(state: AgentState, answer_builder: AnswerBuilder):
    answer = answer_builder.build(
        state["answer"],
        state.get("selected_paper"),
        [result_from_record(result) for result in state["results"]],
    )

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def route_after_chunking(state: AgentState):
    if state.get("chunks"):
        return "retrieve_chunks"

    return "error"


def route_after_retrieval(state: AgentState):
    if state.get("results"):
        return "generate_answer"

    return "error"


def route_after_download(state: AgentState):
    if state.get("pdf_path"):
        return "parse_and_chunk"

    return "error"


def route_after_answer(state: AgentState):
    if state.get("answer"):
        return "build_answer"

    return "error"


def search_paper(state: AgentState, arxiv_client: ArxivClient):
    try:
        paper = arxiv_client.search_by_id(state["paper"])
    except Exception:
        logger.exception("Failed to find paper on arXiv.")
        return {"papers": [], "error": "Failed to find paper"}

    if paper is None:
        return {"papers": [], "error": "Paper not found"}

    return {"papers": [paper]}
