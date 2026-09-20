from project.nodes.arxiv import Paper
from project.nodes.vector_store import SearchResult

class AnswerBuilder:
    def build(self, answer: str, paper: Paper, results: list[SearchResult]) -> str:
        lines = []

        lines.append("## Answer")
        lines.append(answer)
        lines.append("")
        lines.append("## Evidence")

        pages = []

        for result in results:
            page = result.chunk.page

            if page not in pages:
                pages.append(page)

        for page in pages:
            lines.append(f"- Page {page}")

        lines.append("")
        lines.append("## Source")
        lines.append(paper.title)
        lines.append(f"arXiv: {paper.arxiv_id}")

        return "\n".join(lines)