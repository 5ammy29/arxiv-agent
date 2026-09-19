from project.nodes.chunker import TextChunk
from project.nodes.llm import LLMError, OllamaLLM
from project.nodes.vector_store import SearchResult

class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data

class FakeRequests:
    def __init__(self, response):
        self.response = response
        self.url = None
        self.payload = None

    def post(self, url, json, timeout):
        self.url = url
        self.payload = json

        return self.response

def test_generate_returns_response():
    fake_requests = FakeRequests(FakeResponse({"response": "Generated answer"}))

    llm = OllamaLLM(session=fake_requests)

    result = llm.generate("Test prompt")

    assert result == "Generated answer"
    assert fake_requests.payload["model"] == "qwen2.5:3b"
    assert fake_requests.payload["stream"] is False


def test_generate_raises_when_response_is_missing():
    fake_requests = FakeRequests(FakeResponse({}))

    llm = OllamaLLM(session=fake_requests)

    try:
        llm.generate("Test prompt")
        assert False
    except LLMError:
        pass

def test_rewrite_query():
    class FakeLLM(OllamaLLM):
        def generate(self, prompt):
            return "model performance evaluation results"

    llm = FakeLLM()

    result = llm.rewrite_query("What did they find about the model's performance?")

    assert result == "model performance evaluation results"

def test_generate_answer():
    class FakeLLM(OllamaLLM):
        def generate(self, prompt):
            return "The model achieved strong performance."

    llm = FakeLLM()

    chunk = TextChunk(chunk_id=0, page=5, text="The model achieved strong performance on the benchmark.")

    results = []
    results.append(SearchResult(chunk=chunk, score=0.91, rerank_score=0.88))

    answer = llm.generate_answer("How did the model perform?", results)

    assert answer == "The model achieved strong performance."

def test_generate_answer_with_no_results():
    llm = OllamaLLM()

    answer = llm.generate_answer("What did the paper find?", [])

    assert answer == "No relevant information was found in the retrieved paper content."