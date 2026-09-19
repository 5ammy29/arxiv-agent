from project.graph.state import AgentState
from project.nodes.arxiv import ArxivClient
from project.nodes.chunker import TextChunker
from project.nodes.llm import OllamaLLM
from project.nodes.pdf import PDFProcessor
from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore

def process_query(state: AgentState, query_processor: QueryProcessor):
    processed_query = query_processor.process(state["query"])

    return {"processed_query": processed_query}

def search_arxiv(state: AgentState, arxiv_client: ArxivClient):
    papers = arxiv_client.search_by_topic(state["processed_query"])

    if not papers:
        return {"papers": [], "error": "No papers found for the query"}

    return {"papers": papers}

def select_paper(state: AgentState):
    papers = state["papers"]

    if not papers:
        raise ValueError("No papers available for selection")

    return {"selected_paper": papers[0]}

def download_pdf(state: AgentState, pdf_processor: PDFProcessor):
    paper = state["selected_paper"]
    pdf_path = pdf_processor.download_pdf(paper.pdf_url, paper.arxiv_id)

    return {"pdf_path": str(pdf_path)}

def parse_and_chunk(state: AgentState, pdf_processor: PDFProcessor, chunker: TextChunker):
    pages = pdf_processor.parse_pdf(state["pdf_path"])
    chunks = chunker.chunk(pages)

    return {"chunks": chunks}


def retrieve_chunks(state: AgentState, vector_store: VectorStore, retriever: Retriever):
    vector_store.add_chunks(state["chunks"])
    results = retriever.retrieve(state["processed_query"])

    return {"results": results}


def generate_answer(state: AgentState, llm: OllamaLLM):
    answer = llm.generate_answer(state["query"], state["results"])

    return {"answer": answer}

def route_after_search(state: AgentState):
    if state.get("papers"):
        return "select_paper"

    return "error"

def handle_error(state: AgentState):
    return {"answer": state.get("error", "No relevant papers were found.")}
