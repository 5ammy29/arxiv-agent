#!/usr/bin/env python3
"""Run baseline and reranked multi-paper retrieval and generate Qwen answers."""

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore
from project.processing.chunker import TextChunk

BASE_DIR = ROOT / "data/eval/qwen/multi_paper"
DEFAULT_DATASET = (
    BASE_DIR / "datasets/multi_paper_retrieval_eval_dataset.json"
)
DEFAULT_CHUNKS = (
    ROOT / "data/eval/qwen/single_paper/corpus/session_chunks.csv"
)


def load_dataset(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, dict):
        raise ValueError("Multi-paper dataset JSON must contain a dictionary.")

    return records


def load_chunks(path: Path) -> dict[str, list[dict]]:
    chunks_by_arxiv_id: dict[str, list[dict]] = {}
    required = {"arxiv_id", "title", "chunk_id", "page_number", "chunk_text"}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}"
            )

        for row in reader:
            arxiv_id = row["arxiv_id"].strip()

            chunk_record = {
                "arxiv_id": arxiv_id,
                "title": row["title"].strip(),
                "chunk": TextChunk(
                    chunk_id=int(row["chunk_id"]),
                    page=int(row["page_number"]),
                    text=row["chunk_text"],
                ),
            }

            chunks_by_arxiv_id.setdefault(arxiv_id, []).append(chunk_record)

    return chunks_by_arxiv_id


def select_question_chunks(
    record: dict,
    chunks_by_arxiv_id: dict[str, list[dict]],
) -> list[dict]:
    selected_chunks = []

    for paper in record["papers"]:
        arxiv_id = str(paper["arxiv_id"])
        paper_chunks = chunks_by_arxiv_id.get(arxiv_id, [])

        for item in paper_chunks:
            selected_chunks.append(item)

    return selected_chunks


def result_record(result, metadata: dict) -> dict:
    return {
        "arxiv_id": metadata["arxiv_id"],
        "title": metadata["title"],
        "chunk_id": int(metadata["chunk"].chunk_id),
        "page": int(metadata["chunk"].page),
        "text": metadata["chunk"].text,
        "retrieval_score": float(result.score),
        "rerank_score": (
            float(result.rerank_score)
            if result.rerank_score is not None
            else None
        ),
    }


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str,
    ollama_url: str,
):
    context = "\n\n".join(
        (
            f"[Paper: {item['title']} | arXiv: {item['arxiv_id']} | "
            f"Page {item['page']} | Chunk {item['chunk_id']}]\n"
            f"{item['text']}"
        )
        for item in chunks
    )

    prompt = f"""Answer the question using only the passages retrieved
from the provided research papers.

Synthesize and compare information across papers when necessary.
Clearly distinguish which paper supports each claim when relevant.
Do not invent facts or use outside knowledge.
If the passages are insufficient to answer the question, state that.
Be clear and concise.

Question:
{question}

Retrieved passages:
{context if context else "No passages were retrieved."}

Answer:
"""

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()

    return response.json()["response"].strip()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_experiment(
    mode: str,
    records: dict[str, dict],
    chunks_by_arxiv_id: dict[str, list[dict]],
    vector_store: VectorStore,
    top_k: int,
    candidate_k: int,
    model: str,
    ollama_url: str,
) -> None:
    reranker = Reranker() if mode == "reranked" else None
    retriever = Retriever(
        vector_store=vector_store,
        reranker=reranker,
    )

    retrieved_output = {}
    answers_output = {}

    completed = 0

    for question_id, record in records.items():
        question = record["question"]
        papers = record["papers"]

        question_chunks = select_question_chunks(
            record=record,
            chunks_by_arxiv_id=chunks_by_arxiv_id,
        )

        # Chunk IDs can repeat across papers.
        # Assign unique internal IDs for this question's FAISS index.
        internal_chunks = []
        metadata_by_internal_id = {}

        for internal_id, item in enumerate(question_chunks):
            original_chunk = item["chunk"]

            internal_chunk = TextChunk(
                chunk_id=internal_id,
                page=original_chunk.page,
                text=original_chunk.text,
            )

            internal_chunks.append(internal_chunk)

            metadata_by_internal_id[internal_id] = {
                "arxiv_id": item["arxiv_id"],
                "title": item["title"],
                "chunk": original_chunk,
            }

        vector_store.reset()

        if internal_chunks:
            vector_store.add_chunks(internal_chunks)

            # Retriever applies the optional reranker internally.
            results = retriever.retrieve(
                query=question,
                top_k=candidate_k,
            )[:top_k]
        else:
            results = []

        chunks = [
            result_record(
                result=result,
                metadata=metadata_by_internal_id[int(result.chunk.chunk_id)],
            )
            for result in results
        ]

        retrieved_chunk_ids = [
            {
                "arxiv_id": item["arxiv_id"],
                "chunk_id": item["chunk_id"],
            }
            for item in chunks
        ]

        retrieved_output[question_id] = {
            "question": question,
            "papers": papers,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": chunks,
        }

        answer = generate_answer(
            question=question,
            chunks=chunks,
            model=model,
            ollama_url=ollama_url,
        )

        answers_output[question_id] = {
            "question": question,
            "papers": papers,
            "answer": answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_chunks": chunks,
        }

        completed += 1
        print(f"[{mode}] {completed}/{len(records)} — {question_id}")

    prediction_dir = BASE_DIR / "predictions" / mode

    write_json(
        prediction_dir / "retrieved_chunks.json",
        retrieved_output,
    )
    write_json(
        prediction_dir / "generated_answers.json",
        answers_output,
    )

    print(f"{mode} completed: {completed} questions.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline and reranked multi-paper retrieval "
            "and generate Qwen answers."
        )
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be >= --top-k")

    records = load_dataset(args.dataset)
    chunks_by_arxiv_id = load_chunks(args.chunks)

    vector_store = VectorStore()

    for mode in ("baseline", "reranked"):
        run_experiment(
            mode=mode,
            records=records,
            chunks_by_arxiv_id=chunks_by_arxiv_id,
            vector_store=vector_store,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            model=args.model,
            ollama_url=args.ollama_url,
        )

    print("Both experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
