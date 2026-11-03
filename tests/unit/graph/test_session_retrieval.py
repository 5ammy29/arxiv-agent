from uuid import uuid4
from unittest.mock import patch

from project.graph.nodes import retrieve_chunks, route_after_query
from project.nodes.vector_store import SearchResult
from project.processing.chunker import TextChunk


class Cache:
    def get_or_build_index(self, paper_id, chunks, vector_store, **kwargs):
        vector_store.current = str(paper_id)


class Store:
    def __init__(self, responses):
        self.responses = responses
        self.current = None

    def search(self, query, top_k):
        return list(self.responses[self.current])


class Reranker:
    def __init__(self):
        self.received = []

    def rerank(self, query, results):
        self.received = list(results)
        return sorted(results, key=lambda item: item.chunk.text, reverse=True)


class Retriever:
    def __init__(self, reranker):
        self.reranker = reranker


def paper(paper_id, arxiv_id, title):
    return {"id": paper_id, "arxiv_id": arxiv_id, "title": title}


def content(chunk_id, page, text):
    return {"pages": [], "chunks": [{"chunk_id": chunk_id, "page_number": page, "text": text}]}


def test_session_retrieval_globally_reranks_and_preserves_metadata():
    ids = [str(uuid4()) for _ in range(3)]
    papers = [paper(ids[0], "onev1", "One"), paper(ids[1], "twov2", "Two"), paper(ids[2], "threev1", "Three")]
    store = Store({
        ids[0]: [SearchResult(TextChunk(0, 1, "alpha"), .9)],
        ids[1]: [SearchResult(TextChunk(0, 2, "bravo"), .8)],
        ids[2]: [SearchResult(TextChunk(0, 3, "charlie"), .7)],
    })
    reranker = Reranker()
    state = {
        "session_id": str(uuid4()), "paper_id": ids[0], "processed_query": "compare",
        "chunks": [{"chunk_id": 0, "page": 1, "text": "alpha"}],
        "content_fingerprint": "content", "chunking_fingerprint": "chunking",
        "selected_paper": type("Paper", (), {"arxiv_id": "onev1", "title": "One"})(),
    }
    with patch("project.graph.nodes.get_session_papers", return_value=papers), patch(
        "project.graph.nodes.get_paper_content", side_effect=[content(0, 2, "bravo"), content(0, 3, "charlie")]
    ), patch("project.graph.nodes.get_paper_artifact", return_value={"content_fingerprint": "content", "chunking_fingerprint": "chunking"}):
        output = retrieve_chunks(state, store, Retriever(reranker), Cache(), candidates_per_paper=2, final_top_k=3)

    assert {result.paper_id for result in reranker.received} == set(ids)
    assert [item["arxiv_id"] for item in output["results"]] == ["threev1", "twov2", "onev1"]
    assert output["results"][0]["chunk"]["page"] == 3
    assert output["results"][0]["paper_title"] == "Three"


def test_session_retrieval_deduplicates_and_survives_one_paper_failure():
    first, broken, last = [str(uuid4()) for _ in range(3)]
    papers = [paper(first, "firstv1", "First"), paper(broken, "brokenv1", "Broken"), paper(last, "lastv1", "Last")]
    duplicate = SearchResult(TextChunk(0, 1, "same"), .9)
    store = Store({first: [duplicate, duplicate], last: [SearchResult(TextChunk(1, 4, "last"), .8)]})
    state = {"session_id": str(uuid4()), "paper_id": first, "processed_query": "q", "chunks": [{"chunk_id": 0, "page": 1, "text": "same"}], "selected_paper": type("Paper", (), {"arxiv_id": "firstv1", "title": "First"})()}
    with patch("project.graph.nodes.get_session_papers", return_value=papers), patch(
        "project.graph.nodes.get_paper_content", side_effect=[RuntimeError("unavailable"), content(1, 4, "last")]
    ), patch("project.graph.nodes.get_paper_artifact", return_value={}):
        output = retrieve_chunks(state, store, Retriever(None), Cache())

    assert len(output["results"]) == 2
    assert {item["arxiv_id"] for item in output["results"]} == {"firstv1", "lastv1"}


def test_resumed_session_routes_directly_to_session_retrieval():
    assert route_after_query({
        "processed_query": "compare their methods",
        "session_id": str(uuid4()),
        "paper_id": str(uuid4()),
        "chunks": [{"chunk_id": 0, "page": 1, "text": "stored"}],
        "paper": None,
    }) == "retrieve_chunks"
