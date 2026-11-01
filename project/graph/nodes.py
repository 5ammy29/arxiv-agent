import logging
import re
from functools import partial

from langchain_core.messages import AIMessage

from project.application.retrieval.vector_index_cache import VectorIndexCache
from project.application.papers.prepare_paper import PaperPreparationService
from project.graph.state import AgentState
from project.infrastructure.database.papers import register_paper
from project.nodes.answer import AnswerBuilder
from project.nodes.arxiv import ArxivClient, ArxivError
from project.nodes.llm import LLMError, OllamaLLM
from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore
from project.processing.chunker import TextChunker
from project.processing.pdf import PDFDownloadError, PDFParseError, PDFProcessor


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


def download_pdf(
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
        result = preparation_service.download(
            pdf_url=paper.pdf_url,
            arxiv_id=paper.arxiv_id,
            paper_id=paper_id,
        )
    except Exception:
        logger.exception("Unexpected PDF preparation service failure.")
        return {"error": "Failed to prepare paper PDF"}

    if result.error:
        return {"error": result.error}

    return {
        "pdf_path": result.pdf_path,
        "paper_id": str(paper_id),
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
        return {"chunks": result.chunks or [], "error": result.error}

    return {
        "chunks": result.chunks,
        "content_fingerprint": result.content_fingerprint,
        "chunking_fingerprint": result.chunking_fingerprint,
    }


def retrieve_chunks(
    state: AgentState,
    vector_store: VectorStore,
    retriever: Retriever,
    vector_index_cache: VectorIndexCache,
):
    try:
        vector_index_cache.get_or_build_index(
            paper_id=state["paper_id"],
            chunks=state["chunks"],
            vector_store=vector_store,
            content_fingerprint=state.get("content_fingerprint"),
            chunking_fingerprint=state.get("chunking_fingerprint"),
        )

        results = retriever.retrieve(state["processed_query"])

    except Exception:
        logger.exception(
            "Index construction or retrieval failed for paper %s.",
            state.get("paper_id"),
        )
        return {
            "results": [],
            "error": "Failed to build the search index or retrieve paper content",
        }

    if not results:
        return {
            "results": [],
            "error": "No relevant information was found in the paper",
        }

    return {"results": results}


def generate_answer(state: AgentState, llm: OllamaLLM):
    try:
        messages = state.get("messages", [])

        answer = llm.generate_answer(
            state["query"],
            state["results"],
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
        state["selected_paper"],
        state["results"],
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
