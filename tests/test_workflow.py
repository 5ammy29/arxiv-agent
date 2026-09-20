from datetime import datetime
from project.nodes.arxiv import Paper
from project.graph.state import AgentState
from project.graph.workflow import build_workflow
from langgraph.graph import END, START, StateGraph
from project.nodes.chunker import TextChunk
from project.nodes.vector_store import SearchResult
from project.nodes.llm import LLMError
from project.graph.nodes import generate_answer, parse_and_chunk, process_query
from project.nodes.pdf import PDFParseError

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
        def reset(self):
            pass

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

def test_workflow_routes_to_error_when_no_chunks():
    def parse_and_chunk(state: AgentState):
        return {
            "chunks": [],
            "error": "No usable text was found in the paper",
        }

    def route_after_chunking(state: AgentState):
        if state.get("chunks"):
            return "retrieve_chunks"

        return "error"

    def retrieve_chunks(state: AgentState):
        return {"results": ["should-not-run"]}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("parse_and_chunk", parse_and_chunk)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "parse_and_chunk")

    graph.add_conditional_edges(
        "parse_and_chunk",
        route_after_chunking,
        {
            "retrieve_chunks": "retrieve_chunks",
            "error": "error",
        },
    )

    graph.add_edge("retrieve_chunks", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "No usable text was found in the paper"

def test_workflow_routes_to_error_when_no_results():
    def retrieve_chunks(state: AgentState):
        return {
            "results": [],
            "error": "No relevant information was found in the paper",
        }

    def route_after_retrieval(state: AgentState):
        if state.get("results"):
            return "generate_answer"

        return "error"

    def generate_answer(state: AgentState):
        return {"answer": "should-not-run"}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "retrieve_chunks")

    graph.add_conditional_edges(
        "retrieve_chunks",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "error": "error",
        },
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "No relevant information was found in the paper"

def test_workflow_routes_to_error_when_pdf_download_fails():
    def download_pdf(state: AgentState):
        return {"error": "Failed to download PDF for 1234.5678"}

    def route_after_download(state: AgentState):
        if state.get("pdf_path"):
            return "parse_and_chunk"

        return "error"

    def parse_and_chunk(state: AgentState):
        return {"chunks": ["should-not-run"]}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("download_pdf", download_pdf)
    graph.add_node("parse_and_chunk", parse_and_chunk)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "download_pdf")

    graph.add_conditional_edges(
        "download_pdf",
        route_after_download,
        {
            "parse_and_chunk": "parse_and_chunk",
            "error": "error",
        },
    )

    graph.add_edge("parse_and_chunk", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "Failed to download PDF for 1234.5678"

def test_workflow_routes_to_error_when_arxiv_search_fails():
    def search_arxiv(state: AgentState):
        return {
            "papers": [],
            "error": "Failed to search arXiv",
        }

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

    assert result["answer"] == "Failed to search arXiv"

def test_workflow_routes_to_error_when_query_processing_fails():
    def process_query(state: AgentState):
        return {"error": "Failed to process query"}

    def route_after_query(state: AgentState):
        if state.get("processed_query"):
            return "search_arxiv"

        return "error"

    def search_arxiv(state: AgentState):
        return {"papers": ["should-not-run"]}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("process_query", process_query)
    graph.add_node("search_arxiv", search_arxiv)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "process_query")

    graph.add_conditional_edges(
        "process_query",
        route_after_query,
        {
            "search_arxiv": "search_arxiv",
            "error": "error",
        },
    )

    graph.add_edge("search_arxiv", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "Failed to process query"

def test_workflow_routes_to_error_when_answer_generation_fails():
    def generate_answer(state: AgentState):
        return {"error": "Failed to generate answer"}

    def route_after_answer(state: AgentState):
        if state.get("answer"):
            return "build_answer"

        return "error"

    def build_answer(state: AgentState):
        return {"answer": "should-not-run"}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("generate_answer", generate_answer)
    graph.add_node("build_answer", build_answer)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "generate_answer")

    graph.add_conditional_edges(
        "generate_answer",
        route_after_answer,
        {
            "build_answer": "build_answer",
            "error": "error",
        },
    )

    graph.add_edge("build_answer", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "Failed to generate answer"

def test_process_query_handles_llm_error():
    class FakeQueryProcessor:
        def process(self, query):
            raise LLMError("LLM unavailable")

    state = {"query": "transformers"}

    result = process_query(state, FakeQueryProcessor())

    assert result["error"] == "Failed to process query"


def test_generate_answer_handles_llm_error():
    class FakeLLM:
        def generate_answer(self, query, results):
            raise LLMError("LLM unavailable")

    state = {
        "query": "transformers",
        "results": [],
    }

    result = generate_answer(state, FakeLLM())

    assert result["error"] == "Failed to generate answer"

def test_parse_and_chunk_handles_pdf_parse_error():
    class FakePDFProcessor:
        def parse_pdf(self, pdf_path):
            raise PDFParseError("Invalid PDF")

    class FakeChunker:
        def chunk(self, pages):
            raise AssertionError("Chunker should not run")

    state = {
        "pdf_path": "invalid.pdf",
    }

    result = parse_and_chunk(
        state,
        FakePDFProcessor(),
        FakeChunker(),
    )

    assert result["chunks"] == []
    assert result["error"] == "Failed to parse PDF"

def test_workflow_routes_to_error_when_pdf_parse_fails():
    def parse_and_chunk(state: AgentState):
        return {
            "chunks": [],
            "error": "Failed to parse PDF",
        }

    def route_after_chunking(state: AgentState):
        if state.get("chunks"):
            return "retrieve_chunks"

        return "error"

    def retrieve_chunks(state: AgentState):
        return {"results": ["should-not-run"]}

    def handle_error(state: AgentState):
        return {"answer": state["error"]}

    graph = StateGraph(AgentState)

    graph.add_node("parse_and_chunk", parse_and_chunk)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("error", handle_error)

    graph.add_edge(START, "parse_and_chunk")

    graph.add_conditional_edges(
        "parse_and_chunk",
        route_after_chunking,
        {
            "retrieve_chunks": "retrieve_chunks",
            "error": "error",
        },
    )

    graph.add_edge("retrieve_chunks", END)
    graph.add_edge("error", END)

    workflow = graph.compile()

    result = workflow.invoke({"query": "transformers"})

    assert result["answer"] == "Failed to parse PDF"

def test_process_query_handles_invalid_query():
    class FakeQueryProcessor:
        def process(self, query):
            raise ValueError("query cannot be empty")

    state = {"query": ""}

    result = process_query(state, FakeQueryProcessor())

    assert result["error"] == "query cannot be empty"