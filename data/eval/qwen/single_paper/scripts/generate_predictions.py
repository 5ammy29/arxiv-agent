#!/usr/bin/env python3
"""Run baseline and reranked retrieval, compute metrics, and generate Qwen answers."""

import argparse
import csv
import json
import math
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

BASE_DIR = ROOT / "data/eval/qwen/single_paper"
DEFAULT_DATASET = BASE_DIR / "datasets/single_paper_retrieval_eval_dataset.json"
DEFAULT_CHUNKS = BASE_DIR / "corpus/session_chunks.csv"


def normalize_title(title: str) -> str:
    return " ".join(title.casefold().split())


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Dataset JSON must contain a list of records.")
    return records


def load_chunks(path: Path):
    by_arxiv_id: dict[str, list[TextChunk]] = {}
    by_title: dict[str, list[TextChunk]] = {}
    required = {"arxiv_id", "title", "chunk_id", "page_number", "chunk_text"}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            arxiv_id = row["arxiv_id"].strip()
            title = normalize_title(row["title"])
            chunk = TextChunk(
                chunk_id=int(row["chunk_id"]),
                page=int(row["page_number"]),
                text=row["chunk_text"],
            )
            by_arxiv_id.setdefault(arxiv_id, []).append(chunk)
            by_title.setdefault(title, []).append(chunk)

    return by_arxiv_id, by_title


def relevance_grades(record: dict) -> dict[int, int]:
    grades = {}
    for item in record.get("relevant_chunks", []):
        chunk_id = int(item["chunk_id"])
        grade = int(item.get("relevance_candidate", 1))
        grades[chunk_id] = max(grades.get(chunk_id, 0), grade)
    return grades


def retrieval_metrics(retrieved_ids: list[int], grades: dict[int, int], k: int):
    ranked_ids = retrieved_ids[:k]
    relevant_ids = {chunk_id for chunk_id, grade in grades.items() if grade > 0}

    hits = sum(chunk_id in relevant_ids for chunk_id in ranked_ids)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    mrr = 0.0
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            mrr = 1.0 / rank
            break

    dcg = sum(
        (2**grades.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    ndcg = dcg / idcg if idcg else 0.0

    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": mrr,
        f"ndcg@{k}": ndcg,
    }


def result_record(result) -> dict:
    return {
        "chunk_id": int(result.chunk.chunk_id),
        "page": int(result.chunk.page),
        "text": result.chunk.text,
        "retrieval_score": float(result.score),
        "rerank_score": (
            float(result.rerank_score)
            if result.rerank_score is not None
            else None
        ),
    }


def generate_answer(question: str, chunks: list[dict], model: str, ollama_url: str):
    context = "\n\n".join(
        f"[Page {item['page']} | Chunk {item['chunk_id']}]\n{item['text']}"
        for item in chunks
    )
    prompt = f"""Answer the question using only the passages from the research paper.
Do not invent facts or use outside knowledge. If the passages are insufficient,
state that. Be clear and concise.

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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_metrics(mode: str, rows: list[dict], k: int) -> None:
    out_dir = BASE_DIR / "results" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [f"precision@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"]

    with (out_dir / "retrieval_metrics_per_question.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["question_id", *names])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        name: sum(row[name] for row in rows) / len(rows) if rows else 0.0
        for name in names
    }
    write_json(out_dir / "retrieval_metrics.json", summary)


def run_experiment(
    mode: str,
    records: list[dict],
    chunks_by_id: dict[str, list[TextChunk]],
    chunks_by_title: dict[str, list[TextChunk]],
    vector_store: VectorStore,
    top_k: int,
    candidate_k: int,
    model: str,
    ollama_url: str,
) -> None:
    reranker = Reranker() if mode == "reranked" else None
    retriever = Retriever(vector_store=vector_store, reranker=reranker)

    retrieved_output = {}
    answers_output = {}
    metric_rows = []
    records_by_paper: dict[str, list[dict]] = {}

    for record in records:
        arxiv_id = str(record["paper"]["arxiv_id"])
        records_by_paper.setdefault(arxiv_id, []).append(record)

    completed = 0
    missing_papers = set()

    for arxiv_id, paper_records in records_by_paper.items():
        paper_chunks = chunks_by_id.get(arxiv_id)
        if not paper_chunks:
            paper_chunks = chunks_by_title.get(
                normalize_title(paper_records[0]["paper"]["title"])
            )

        if not paper_chunks:
            missing_papers.add(arxiv_id)
            for record in paper_records:
                qid = str(record["question_id"])
                retrieved_output[qid] = {
                    "question": record["question"],
                    "paper": record["paper"],
                    "retrieved_chunk_ids": [],
                    "retrieved_chunks": [],
                }
                answers_output[qid] = {
                    "question": record["question"],
                    "paper": record["paper"],
                    "answer": "",
                    "retrieved_chunk_ids": [],
                    "retrieved_chunks": [],
                }
            continue

        vector_store.reset()
        vector_store.add_chunks(paper_chunks)

        for record in paper_records:
            qid = str(record["question_id"])
            question = record["question"]

            # Retriever applies the optional reranker internally.
            results = retriever.retrieve(query=question, top_k=candidate_k)[:top_k]
            chunks = [result_record(result) for result in results]
            chunk_ids = [int(result.chunk.chunk_id) for result in results]

            retrieved_output[qid] = {
                "question": question,
                "paper": record["paper"],
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_chunks": chunks,
            }

            metric_rows.append({
                "question_id": qid,
                **retrieval_metrics(chunk_ids, relevance_grades(record), top_k),
            })

            answer = generate_answer(question, chunks, model, ollama_url)
            answers_output[qid] = {
                "question": question,
                "paper": record["paper"],
                "answer": answer,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_chunks": chunks,
            }

            completed += 1
            print(f"[{mode}] {completed}/{len(records)} — {qid}")

    prediction_dir = BASE_DIR / "predictions" / mode
    write_json(prediction_dir / "retrieved_chunks.json", retrieved_output)
    write_json(prediction_dir / "generated_answers.json", answers_output)
    save_metrics(mode, metric_rows, top_k)

    print(f"{mode} completed: {completed} questions.")
    if missing_papers:
        print("Missing paper chunks:", ", ".join(sorted(missing_papers)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline and reranked retrieval and generate Qwen answers."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be >= --top-k")

    records = load_dataset(args.dataset)
    chunks_by_id, chunks_by_title = load_chunks(args.chunks)
    vector_store = VectorStore()

    for mode in ("baseline", "reranked"):
        run_experiment(
            mode=mode,
            records=records,
            chunks_by_id=chunks_by_id,
            chunks_by_title=chunks_by_title,
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
