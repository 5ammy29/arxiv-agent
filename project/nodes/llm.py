import requests
from project.nodes.vector_store import SearchResult

class LLMError(Exception):
    pass

class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434", session=None):
        self.model = model
        self.base_url = base_url.rstrip("/")

        if session is None:
            session = requests

        self.session = session

    def generate(self, prompt: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise LLMError("Failed to generate response from Ollama") from error

        data = response.json()

        if "response" not in data:
            raise LLMError("Ollama response did not contain generated text")

        return data["response"].strip()

    def rewrite_query(self, query: str) -> str:
        prompt = f"""Rewrite the following research question into a concise search query for retrieving relevant passages from an academic paper.

Keep the meaning of the question.
Do not answer the question.
Return only the rewritten search query.

Question:
{query}
"""

        return self.generate(prompt)

    def generate_answer(self, query: str, results: list[SearchResult]) -> str:
        if not results:
            return "No relevant information was found in the retrieved paper content."

        context = []

        for result in results:
            context.append(
                f"[Passage | Page {result.chunk.page}]\n{result.chunk.text}"
            )

        context_text = "\n\n".join(context)

        prompt = f"""You are answering a research question using only passages retrieved from an academic paper.

Follow these rules:
- Use only information supported by the provided passages.
- Do not use outside knowledge.
- Do not invent facts, numbers, methods, results, or conclusions.
- If the passages do not contain enough information to answer the question, say that the available evidence is insufficient.
- Cite supporting evidence using the page numbers provided in the passages.
- Give a concise and research-focused answer.

Question:
{query}

Retrieved passages:
{context_text}
"""

        return self.generate(prompt)