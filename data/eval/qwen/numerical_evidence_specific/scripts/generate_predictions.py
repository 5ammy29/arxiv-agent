"""Generate retrieval and answer predictions for numerical/evidence-specific eval.

Runs both baseline retrieval and cross-encoder reranking over the dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[5]
DATASET_DEFAULT = (
    ROOT
    / "data/eval/qwen/numerical_evidence_specific/datasets/"
    / "numerical_evidence_specific_retrieval_eval_dataset.json"
)
OUTPUT_DEFAULT = ROOT / "data/eval/qwen/numerical_evidence_specific/predictions"

# Ensure imports resolve when this script is run directly from the repository.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.retriever import Retriever
from project.nodes.vector_store import VectorStore
from project.processing.chunker import TextChunk


def load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Expected dataset JSON to be a dictionary keyed by question ID.")

    return dataset


def get_question_chunks(record: dict, corpus: dict[str, list[dict]]) -> list[dict]:
    """Return chunks only from the papers listed for this question."""
    paper_ids = [str(paper["arxiv_id"]) for paper in record.get("papers", [])]
    selected = []

    for arxiv_id in paper_ids:
        if arxiv_id not in corpus:
            raise KeyError(
                f"Dataset paper arXiv ID {arxiv_id!r} is absent from the loaded corpus."
            )
        selected.extend(corpus[arxiv_id])

    if not selected:
        raise ValueError("No chunks found for the papers attached to this question.")

    return selected


def load_corpus(dataset: dict) -> dict[str, list[dict]]:
    """Load each question's expected_chunks as corpus records, deduplicated per paper/chunk."""
    corpus: dict[str, dict[int, dict]] = {}

    for question_id, record in dataset.items():
        for chunk_record in record.get("expected_chunks", []):
            arxiv_id = str(chunk_record["arxiv_id"])
            chunk_id = int(chunk_record["chunk_id"])
            corpus.setdefault(arxiv_id, {})
            corpus[arxiv_id][chunk_id] = {
                "arxiv_id": arxiv_id,
                "title": chunk_record.get("title", ""),
                "chunk_id": chunk_id,
                "page_number": int(chunk_record.get("page_number", chunk_record.get("page", 0))),
                "chunk_text": str(chunk_record.get("chunk_text", chunk_record.get("text", ""))),
            }

    return {arxiv_id: list(chunks.values()) for arxiv_id, chunks in corpus.items()}


def load_chunks_csv(path: Path) -> dict[str, list[dict]]:
    """Load session_chunks.csv, which is the complete retrieval corpus."""
    import csv

    grouped: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"arxiv_id", "title", "chunk_id", "page_number", "chunk_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Corpus CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            arxiv_id = str(row["arxiv_id"]).strip()
            grouped.setdefault(arxiv_id, []).append(
                {
                    "arxiv_id": arxiv_id,
                    "title": row["title"],
                    "chunk_id": int(row["chunk_id"]),
                    "page_number": int(row["page_number"]),
                    "chunk_text": row["chunk_text"],
                }
            )

    return grouped


def build_retriever(chunks: list[dict], use_reranker: bool) -> tuple[Retriever, dict[int, dict]]:
    """Assign unique temporary IDs for indexing, since source chunk IDs may repeat across papers."""
    vector_store = VectorStore()
    metadata_by_internal_id = {}

    indexed_chunks = []
    for internal_id, chunk in enumerate(chunks):
        metadata_by_internal_id[internal_id] = chunk
        indexed_chunks.append(
            TextChunk(
                chunk_id=internal_id,
                page=int(chunk["page_number"]),
                text=str(chunk["chunk_text"]),
            )
        )

    vector_store.add_chunks(indexed_chunks)
    reranker = Reranker() if use_reranker else None
    retriever = Retriever(
        vector_store=vector_store,
        query_processor=QueryProcessor(),
        reranker=reranker,
    )
    return retriever, metadata_by_internal_id


def retrieve_chunks(
    retriever: Retriever,
    question: str,
    chunk_metadata: dict[int, dict],
    top_k: int,
    candidate_k: int,
) -> list[dict]:
    retrieval_k = candidate_k if retriever.reranker is not None else top_k
    results = retriever.retrieve(query=question, top_k=retrieval_k)[:top_k]

    retrieved = []
    for rank, result in enumerate(results, start=1):
        internal_id = int(result.chunk.chunk_id)
        metadata = chunk_metadata[internal_id]
        original_chunk_id = int(metadata["chunk_id"])
        retrieved.append(
            {
                "rank": rank,
                "arxiv_id": metadata["arxiv_id"],
                "title": metadata["title"],
                "chunk_id": original_chunk_id,
                "page_number": metadata["page_number"],
                "chunk_text": result.chunk.text,
                "retrieval_score": float(result.score),
                "rerank_score": (
                    float(result.rerank_score)
                    if result.rerank_score is not None
                    else None
                ),
            }
        )

    return retrieved


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            f"[Paper: {item['title']} | arXiv: {item['arxiv_id']} | "
            f"Page: {item['page_number']} | Chunk: {item['chunk_id']}]\n"
            f"{item['chunk_text']}"
        )

    context = "\n\n".join(context_parts)
    return f"""You are a research assistant answering numerical and evidence-specific questions about research papers.

Answer the question using only the evidence in the retrieved excerpts below.
Prioritize exact numerical values, quantities, percentages, dates, dimensions, parameter counts, dataset sizes, metric scores, and table/figure-specific details.
Preserve units, labels, experimental conditions, and comparisons exactly as stated in the excerpts.
For evidence-specific questions, ground each claim in the provided text and identify relevant paper title, arXiv ID, page, or chunk when useful.
Do not calculate, infer, or approximate values unless the question explicitly asks you to; if you calculate, show the operation.
Do not invent details. If the excerpts do not provide enough information, say what is missing.
When useful, identify the paper by title or arXiv ID.

Retrieved evidence:
{context}

Question:
{question}

Answer:"""


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model: str,
    ollama_url: str,
    timeout: int,
) -> str:
    prompt = build_prompt(question, retrieved_chunks)
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("response", "").strip()
    if not answer:
        raise RuntimeError(f"Ollama returned an empty answer for question: {question}")
    return answer


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def run_mode(
    mode: str,
    dataset: dict,
    corpus: dict[str, list[dict]],
    output_dir: Path,
    top_k: int,
    candidate_k: int,
    model: str,
    ollama_url: str,
    timeout: int,
) -> None:
    use_reranker = mode == "reranked"
    retrieved_output: dict[str, dict] = {}
    answers_output: dict[str, dict] = {}

    retrieved_path = output_dir / mode / "retrieved_chunks.json"
    answers_path = output_dir / mode / "generated_answers.json"

    # If outputs already exist, preserve them and skip completed question IDs.
    if retrieved_path.exists():
        with retrieved_path.open("r", encoding="utf-8") as file:
            retrieved_output = json.load(file)
    if answers_path.exists():
        with answers_path.open("r", encoding="utf-8") as file:
            answers_output = json.load(file)

    pending = [
        question_id
        for question_id in dataset
        if question_id not in retrieved_output or question_id not in answers_output
    ]

    print(f"\n{'=' * 60}\nMode: {mode} | Questions remaining: {len(pending)}\n{'=' * 60}")

    for index, question_id in enumerate(pending, start=1):
        record = dataset[question_id]
        question = record["question"]
        selected_chunks = get_question_chunks(record, corpus)
        # Build a fresh FAISS index from the papers relevant to this question.
        # Temporary IDs prevent chunk-ID collisions between different papers.
        retriever, metadata_by_internal_id = build_retriever(
            selected_chunks,
            use_reranker=use_reranker,
        )

        retrieved = retrieve_chunks(
            retriever=retriever,
            question=question,
            chunk_metadata=metadata_by_internal_id,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        answer = generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
        )

        retrieved_output[question_id] = {
            "question": question,
            "papers": record.get("papers", []),
            "retrieved_chunk_ids": [
                {"arxiv_id": item["arxiv_id"], "chunk_id": item["chunk_id"]}
                for item in retrieved
            ],
            "retrieved_chunks": retrieved,
        }
        answers_output[question_id] = {
            "question": question,
            "papers": record.get("papers", []),
            "answer": answer,
        }

        # Save after each question to support resuming if interrupted.
        write_json(retrieved_path, retrieved_output)
        write_json(answers_path, answers_output)
        print(f"[{index}/{len(pending)}] {question_id} completed")

    print(f"Retrieved chunks saved to: {retrieved_path}")
    print(f"Generated answers saved to: {answers_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate numerical/evidence-specific predictions with baseline and reranked retrieval."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data/eval/qwen/single_paper/corpus/session_chunks.csv",
        help="CSV corpus containing arxiv_id,title,chunk_id,page_number,chunk_text",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "reranked"],
        default=["baseline", "reranked"],
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be greater than or equal to --top-k")

    dataset = load_dataset(args.dataset)
    corpus = load_chunks_csv(args.corpus)

    print(f"Dataset: {args.dataset}")
    print(f"Questions: {len(dataset)}")
    print(f"Corpus papers: {len(corpus)}")
    print(f"Top K: {args.top_k} | Candidate K: {args.candidate_k}")
    print(f"Model: {args.model}")

    for mode in args.modes:
        run_mode(
            mode=mode,
            dataset=dataset,
            corpus=corpus,
            output_dir=args.output_dir,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            model=args.model,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
        )

    print("\nPrediction generation completed.")


if __name__ == "__main__":
    main()
