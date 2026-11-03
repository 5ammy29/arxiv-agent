from project.nodes.arxiv import Paper
from project.nodes.vector_store import SearchResult


class AnswerBuilder:
    def build(
        self,
        answer: str,
        paper: Paper | None,
        results: list[SearchResult],
    ) -> str:
        lines = ["## Answer", answer, "", "## Evidence"]
        evidence: dict[tuple[str, str | None, int], list[int]] = {}

        for result in results:
            title = result.paper_title or (paper.title if paper is not None else "Unknown paper")
            arxiv_id = result.arxiv_id or (paper.arxiv_id if paper is not None else None)
            reference = (
                title,
                arxiv_id,
                result.chunk.page,
            )
            evidence.setdefault(reference, [])
            if result.chunk.chunk_id not in evidence[reference]:
                evidence[reference].append(result.chunk.chunk_id)

        for (title, arxiv_id, page), chunk_ids in evidence.items():
            source = f" ({arxiv_id})" if arxiv_id else ""
            chunks = ", ".join(map(str, chunk_ids))
            lines.append(f"- Page {page} — {title}{source}, chunks {chunks}")

        lines.extend([
            "",
            "## Source",
        ])

        sources: dict[tuple[str, str | None], set[int]] = {}
        for title, arxiv_id, page in evidence:
            sources.setdefault((title, arxiv_id), set()).add(page)

        # Preserve the prior single-paper output if no evidence exists.
        if not sources and paper is not None:
            sources[(paper.title, paper.arxiv_id)] = set()

        for (title, arxiv_id), pages in sources.items():
            identifier = f" — arXiv: {arxiv_id}" if arxiv_id else ""
            page_text = f"; pages {', '.join(map(str, sorted(pages)))}" if pages else ""
            lines.append(f"- {title}{identifier}{page_text}")

        return "\n".join(lines)
