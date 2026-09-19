import re
from project.nodes.llm import OllamaLLM

class QueryProcessor:
    def __init__(self, llm: OllamaLLM | None = None):
        self.llm = llm

    def process(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = re.sub(r"\s+", " ", query).strip()

        if not query:
            raise ValueError("query cannot be empty")

        if self.llm is not None:
            query = self.llm.rewrite_query(query)

            if not query:
                raise ValueError("LLM returned an empty query")

        return query