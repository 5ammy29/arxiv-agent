from datetime import datetime

import arxiv
from pydantic import BaseModel


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime
    url: str
    pdf_url: str
    categories: list[str]

class ArxivError(Exception):
    pass

class ArxivClient:
    def __init__(self):
        self.client = arxiv.Client()

    def search_by_topic(self, query: str, max_results: int = 5) -> list[Paper]:
        if not query.strip():
            raise ValueError("search query cannot be empty")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            results = self.client.results(search)

            normalized_results = []

            for result in results:
                normalized_results.append(self._normalize_result(result))

            return normalized_results
        except Exception as error:
            raise ArxivError("Failed to search arXiv") from error

    @staticmethod
    def _normalize_result(result: arxiv.Result) -> Paper:
        authors = []

        for author in result.authors:
            authors.append(author.name)

        return Paper(
            arxiv_id=result.get_short_id(),
            title=result.title.strip(),
            authors=authors,
            abstract=result.summary.strip(),
            published=result.published,
            updated=result.updated,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            categories=result.categories,
        )

    def search_by_id(self, identifier: str) -> Paper | None:
        arxiv_id = self._extract_arxiv_id(identifier)

        search = arxiv.Search(id_list=[arxiv_id], max_results=1)

        results = list(self.client.results(search))

        if not results:
            return None

        return self._normalize_result(results[0])

    @staticmethod
    def _extract_arxiv_id(identifier: str) -> str:
        identifier = identifier.strip()

        if "/abs/" in identifier:
            identifier = identifier.split("/abs/", maxsplit=1)[1]

        if "/pdf/" in identifier:
            identifier = identifier.split("/pdf/", maxsplit=1)[1]

        if identifier.endswith(".pdf"):
            identifier = identifier[:-4]

        return identifier
    