from project.nodes.arxiv import Paper
from project.nodes.vector_store import SearchResult

class AnswerBuilder:
    def build(self, answer: str, paper: Paper, results: list[SearchResult]) -> str:
        lines = ["## Answer", answer, "", "## Evidence"]
        pages = []

        for result in results:
            page = result.chunk.page

            if page not in pages:
                pages.append(page)

        for page in pages:
            lines.append(f"- Page {page}")

        lines.extend([
            "",
            "## Source",
            paper.title,
            f"arXiv: {paper.arxiv_id}",
        ])

        return "\n".join(lines)