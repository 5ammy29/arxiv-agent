from typing import TypedDict

from langgraph.graph import MessagesState

from project.nodes.arxiv import Paper


class AgentState(MessagesState, total=False):
    session_id: str
    query: str
    paper: str | None
    paper_id: str
    processed_query: str
    papers: list[Paper]
    selected_paper: Paper
    pdf_path: str
    # Keep checkpointed graph state made of built-in types.  TextChunk and
    # SearchResult are transient retrieval-domain objects reconstructed by nodes.
    chunks: list[dict]
    content_fingerprint: str
    chunking_fingerprint: str
    results: list[dict]
    answer: str
    error: str
