import requests
from project.nodes.vector_store import SearchResult

class LLMError(Exception):
    pass

class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        session=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session = requests if session is None else session

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
        prompt = f"""Rewrite the following research question into a concise search query for an academic paper.

Identify the exact technical concept, variable, parameter, or quantity being asked about.
Use standard technical terminology or notation when it is clearly implied by the question.
Preserve important model components, quantities, and distinctions.
Do not answer the question.
Do not replace a specific technical concept with a broader related concept.
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
- Answer the exact quantity or concept asked in the question.
- Do not substitute a related quantity for the requested one.
- Pay close attention to technical notation and variable names.
- When multiple values appear in the passages, identify which value corresponds to the quantity asked about.
- Prefer explicit tabulated values when they directly identify the requested quantity.
- If the passages do not contain enough information to answer the question, say that the available evidence is insufficient.
- Cite supporting evidence using the page numbers provided in the passages.
- Give a concise and research-focused answer.

Question:
{query}

Retrieved passages:
{context_text}
"""

        return self.generate(prompt)