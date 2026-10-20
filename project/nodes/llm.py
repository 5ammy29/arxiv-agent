import requests

from langchain_core.messages import BaseMessage

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


    def generate_answer(
        self,
        query: str,
        results: list[SearchResult],
        conversation_history: list[BaseMessage] | None = None,
    ) -> str:
        history_parts = []

        if conversation_history:
            for message in conversation_history:
                if message.type == "human":
                    role = "User"
                elif message.type == "ai":
                    role = "Assistant"
                else:
                    continue

                history_parts.append(f"{role}: {message.content}")

        history_text = "No previous conversation."

        if history_parts:
            history_text = "\n\n".join(history_parts)

        context = []

        if results:
            for result in results:
                context.append(
                    f"[Passage | Page {result.chunk.page}]\n{result.chunk.text}"
                )

        context_text = "\n\n".join(context)

        prompt = f"""You are a research assistant with persistent conversation history.

First, determine what the current question asks.

1. If the user asks what they previously said, preferred, requested,
   or discussed, answer using the conversation history only.
   Do not add research details or infer preferences from retrieved passages.
   If the history does not contain the answer, say you cannot determine it
   from the conversation history.

2. If the user asks a research question, answer using the retrieved passages.
   Do not invent facts or treat previous assistant answers as verified evidence.
   Cite supporting evidence using the provided page numbers.
   If the passages are insufficient, say so.

3. For a follow-up research question, use conversation history to resolve
   references, but use retrieved passages as evidence for factual claims.

Conversation history:
{history_text}

Current question:
{query}

Retrieved passages:
{context_text if context_text else "No relevant passages were retrieved."}

Answer:
"""

        return self.generate(prompt)
