from functools import partial
from project.nodes.answer import AnswerBuilder
from langgraph.graph import END, START, StateGraph
from project.graph.nodes import (
    download_pdf,
    generate_answer,
    handle_error,
    parse_and_chunk,
    process_query,
    retrieve_chunks,
    route_after_search,
    search_arxiv,
    select_paper,
    build_answer,
    route_after_chunking,
    route_after_retrieval,
    route_after_download,
)
from project.graph.state import AgentState
from project.nodes.arxiv import ArxivClient
from project.nodes.chunker import TextChunker
from project.nodes.llm import OllamaLLM
from project.nodes.pdf import PDFProcessor
from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore

def build_workflow(
    arxiv_client=None,
    pdf_processor=None,
    chunker=None,
    llm=None,
    query_processor=None,
    vector_store=None,
    reranker=None,
    retriever=None,
    answer_builder=None,
):
    if arxiv_client is None:
        arxiv_client = ArxivClient()

    if pdf_processor is None:
        pdf_processor = PDFProcessor()

    if chunker is None:
        chunker = TextChunker()

    if llm is None:
        llm = OllamaLLM()

    if query_processor is None:
        query_processor = QueryProcessor(llm=llm)

    if vector_store is None:
        vector_store = VectorStore()

    if reranker is None:
        reranker = Reranker()

    if retriever is None:
        retriever = Retriever(vector_store, reranker=reranker)

    if answer_builder is None:
        answer_builder = AnswerBuilder()

    graph = StateGraph(AgentState)

    graph.add_node("process_query", partial(process_query, query_processor=query_processor))
    graph.add_node("search_arxiv", partial(search_arxiv, arxiv_client=arxiv_client))
    graph.add_node("select_paper", select_paper)
    graph.add_node("download_pdf", partial(download_pdf, pdf_processor=pdf_processor))
    graph.add_node("parse_and_chunk", partial(parse_and_chunk, pdf_processor=pdf_processor, chunker=chunker))
    graph.add_node("retrieve_chunks", partial(retrieve_chunks, vector_store=vector_store, retriever=retriever))
    graph.add_node("generate_answer", partial(generate_answer, llm=llm))
    graph.add_node("build_answer", partial(build_answer, answer_builder=answer_builder))
    graph.add_node("error", handle_error)

    graph.add_edge(START, "process_query")
    graph.add_edge("process_query", "search_arxiv")

    graph.add_conditional_edges(
        "search_arxiv",
        route_after_search,
        {
            "select_paper": "select_paper",
            "error": "error",
        },
    )

    graph.add_edge("select_paper", "download_pdf")

    graph.add_conditional_edges(
        "download_pdf",
        route_after_download,
        {
            "parse_and_chunk": "parse_and_chunk",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "parse_and_chunk",
        route_after_chunking,
        {
            "retrieve_chunks": "retrieve_chunks",
            "error": "error",
        },
    )

    graph.add_conditional_edges(
        "retrieve_chunks",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "error": "error",
        },
    )

    graph.add_edge("generate_answer", "build_answer")
    graph.add_edge("build_answer", END)
    graph.add_edge("error", END)

    return graph.compile()
