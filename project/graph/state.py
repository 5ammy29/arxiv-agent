from typing import TypedDict
from project.nodes.arxiv import Paper
from project.nodes.chunker import TextChunk
from project.nodes.vector_store import SearchResult

class AgentState(TypedDict, total=False):
    query: str
    paper: str
    processed_query: str
    papers: list[Paper]
    selected_paper: Paper
    pdf_path: str
    chunks: list[TextChunk]
    results: list[SearchResult]
    answer: str
    error: str