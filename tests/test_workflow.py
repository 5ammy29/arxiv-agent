from datetime import datetime
from project.nodes.arxiv import Paper
from project.graph.state import AgentState
from project.graph.workflow import build_workflow
from langgraph.graph import END, START, StateGraph
from project.nodes.chunker import TextChunk
from project.nodes.vector_store import SearchResult

def test_workflow_routes_to_error_when_no_papers():
    def search_arxiv(state: AgentState):
        return {"papers": [], "error": "No papers found for the query"}

    def route_after_search(state: AgentState):
        if state.get("papers"):
            return "select_paper"

        return "error"

    def select_paper(state: AgentState):
        return {"selected_paper": state["papers"][0]}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("search_arxiv", search_arxiv)
    graph.add_node("select_paper", select_paper)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "search_arxiv")

    graph.add_conditional_edges(
        "search_arxiv",
        route_after_search,
        {
            "select_paper": "select_paper",
            "error": "error",
        },
    )

    graph.add_edge("select_paper", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "No papers found for the query"

def test_workflow_routes_to_select_paper_when_papers_are_found():
    def search_arxiv(state: AgentState):
        return {"papers": ["paper-1"]}

    def route_after_search(state: AgentState):
        if state.get("papers"):
            return "select_paper"

        return "error"

    def select_paper(state: AgentState):
        return {"selected_paper": state["papers"][0]}

    def handle_error(state: AgentState):
        return {"answer": state.get("error", "No relevant papers were found.")}

    graph = StateGraph(AgentState)

    graph.add_node("search_arxiv", search_arxiv)
    graph.add_node("select_paper", select_paper)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "search_arxiv")

    graph.add_conditional_edges(
        "search_arxiv",
        route_after_search,
        {
            "select_paper": "select_paper",
            "error": "error",
        },
    )

    graph.add_edge("select_paper", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["selected_paper"] == "paper-1"

def test_workflow_contains_all_nodes():
    workflow = build_workflow(
        arxiv_client=object(),
        pdf_processor=object(),
        chunker=object(),
        llm=object(),
        query_processor=object(),
        vector_store=object(),
        reranker=object(),
        retriever=object(),
    )

    expected_nodes = {
        "process_query",
        "search_arxiv",
        "select_paper",
        "download_pdf",
        "parse_and_chunk",
        "retrieve_chunks",
        "generate_answer",
        "build_answer",
        "error",
    }

    assert expected_nodes.issubset(workflow.nodes.keys())

def test_full_workflow_execution():
    class FakeQueryProcessor:
        def process(self, query):
            return "processed query"

    class FakeArxivClient:
        def search_by_topic(self, query):
            paper = Paper(
                arxiv_id="1234.5678",
                title="Test Paper",
                authors=["Test Author"],
                abstract="Test abstract",
                published=datetime.now(),
                updated=datetime.now(),
                url="https://arxiv.org/abs/1234.5678",
                pdf_url="https://arxiv.org/pdf/1234.5678",
                categories=["cs.AI"],
            )

            return [paper]

    class FakePDFProcessor:
        def download_pdf(self, pdf_url, arxiv_id):
            return "paper.pdf"

        def parse_pdf(self, pdf_path):
            return ["page-1"]

    class FakeChunker:
        def chunk(self, pages):
            return ["chunk-1"]

    class FakeVectorStore:
        def add_chunks(self, chunks):
            pass

    fake_chunk = TextChunk(
        chunk_id=0,
        page=3,
        text="Test evidence",
    )

    fake_result = SearchResult(
        chunk=fake_chunk,
        score=0.9,
    )

    class FakeRetriever:
        def retrieve(self, query):
            return [fake_result]

    class FakeLLM:
        def generate_answer(self, query, results):
            return "final answer"

    workflow = build_workflow(
        arxiv_client=FakeArxivClient(),
        pdf_processor=FakePDFProcessor(),
        chunker=FakeChunker(),
        llm=FakeLLM(),
        query_processor=FakeQueryProcessor(),
        vector_store=FakeVectorStore(),
        reranker=object(),
        retriever=FakeRetriever(),
    )

    result = workflow.invoke({"query": "What did the paper find?"})

    assert result["processed_query"] == "processed query"
    assert result["papers"][0].arxiv_id == "1234.5678"
    assert result["selected_paper"].arxiv_id == "1234.5678"
    assert result["pdf_path"] == "paper.pdf"
    assert result["chunks"] == ["chunk-1"]
    assert result["results"] == [fake_result]
    assert "## Answer" in result["answer"]
    assert "final answer" in result["answer"]
    assert "## Evidence" in result["answer"]
    assert "- Page 3" in result["answer"]
    assert "## Source" in result["answer"]
    assert "Test Paper" in result["answer"]
    assert "arXiv: 1234.5678" in result["answer"]