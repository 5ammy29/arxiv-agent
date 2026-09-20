import re
from project.graph.state import AgentState
from project.nodes.arxiv import ArxivClient, ArxivError
from project.nodes.answer import AnswerBuilder
from project.nodes.chunker import TextChunker
from project.nodes.llm import LLMError, OllamaLLM
from project.nodes.pdf import PDFDownloadError, PDFParseError, PDFProcessor
from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore

def process_query(state: AgentState, query_processor: QueryProcessor):
    try:
        processed_query = query_processor.process(state["query"])
    except LLMError:
        return {"error": "Failed to process query"}
    except (TypeError, ValueError) as error:
        return {"error": str(error)}

    return {"processed_query": processed_query}

def route_after_query(state: AgentState):
    if state.get("processed_query"):
        return "search_arxiv"

    return "error"

def search_arxiv(state: AgentState, arxiv_client: ArxivClient):
    try:
        papers = arxiv_client.search_by_topic(state["processed_query"])
    except ArxivError:
        return {"papers": [], "error": "Failed to search arXiv"}

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

def download_pdf(state: AgentState, pdf_processor: PDFProcessor):
    paper = state["selected_paper"]

    try:
        pdf_path = pdf_processor.download_pdf(paper.pdf_url, paper.arxiv_id)
    except PDFDownloadError:
        return {"error": f"Failed to download PDF for {paper.arxiv_id}"}

    return {"pdf_path": str(pdf_path)}

def parse_and_chunk(state: AgentState, pdf_processor: PDFProcessor, chunker: TextChunker):
    try:
        pages = pdf_processor.parse_pdf(state["pdf_path"])
    except PDFParseError:
        return {"chunks": [], "error": "Failed to parse PDF"}

    chunks = chunker.chunk(pages)

    if not chunks:
        return {"chunks": [], "error": "No usable text was found in the paper"}

    return {"chunks": chunks}


def retrieve_chunks(state: AgentState, vector_store: VectorStore, retriever: Retriever):
    vector_store.reset()
    vector_store.add_chunks(state["chunks"])

    results = retriever.retrieve(state["processed_query"])

    if not results:
        return {
            "results": [],
            "error": "No relevant information was found in the paper",
        }

    return {"results": results}


def generate_answer(state: AgentState, llm: OllamaLLM):
    try:
        answer = llm.generate_answer(state["query"], state["results"])
    except LLMError:
        return {"error": "Failed to generate answer"}

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

    return {"answer": answer}

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