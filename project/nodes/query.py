import re

class QueryProcessor:
    def process(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = re.sub(r"\s+", " ", query).strip()

        if not query:
            raise ValueError("query cannot be empty")

        return query
