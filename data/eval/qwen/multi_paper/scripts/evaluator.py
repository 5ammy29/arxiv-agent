#!/usr/bin/env python3
"""Evaluate multi-paper retrieval only (no LLM-as-a-Judge)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = ROOT / "data/eval/qwen/multi_paper"
DEFAULT_DATASET = BASE_DIR / "datasets" / "multi_paper_retrieval_eval_dataset.json"
DEFAULT_PREDICTIONS = BASE_DIR / "predictions"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"


def normalize_chunk_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def normalize_arxiv_id(value: Any) -> str:
    return str(value).strip()


def chunk_key(arxiv_id: Any, chunk_id: Any) -> tuple[str, str]:
    """Uniquely identify a chunk across multiple papers."""
    return normalize_arxiv_id(arxiv_id), normalize_chunk_id(chunk_id)


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Multi-paper dataset JSON must be a dictionary.")

    for question_id, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {question_id} is not an object.")
        if not isinstance(record.get("papers"), list):
            raise ValueError(f"Dataset record {question_id} has no papers list.")
        if not isinstance(record.get("expected_chunk_ids"), list):
            raise ValueError(
                f"Dataset record {question_id} has no expected_chunk_ids list."
            )

    return {str(question_id): record for question_id, record in data.items()}


def load_retrieval_results(path: Path) -> dict[str, list[Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Retrieval results must map question IDs to records.")

    results: dict[str, list[Any]] = {}
    for question_id, record in data.items():
        if not isinstance(record, dict):
            raise ValueError(f"Retrieval record for {question_id} must be an object.")
        chunk_ids = record.get("retrieved_chunk_ids", [])
        if not isinstance(chunk_ids, list):
            raise ValueError(f"Retrieved chunk IDs for {question_id} must be a list.")
        results[str(question_id)] = chunk_ids
    return results


def normalize_retrieved_chunk(raw_chunk: Any) -> tuple[str, str]:
    if not isinstance(raw_chunk, dict):
        raise ValueError("Each retrieved chunk must contain arxiv_id and chunk_id.")
    if "arxiv_id" not in raw_chunk or "chunk_id" not in raw_chunk:
        raise ValueError("Retrieved chunk is missing arxiv_id or chunk_id.")
    return chunk_key(raw_chunk["arxiv_id"], raw_chunk["chunk_id"])


def dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def evaluate_question(
    question_id: str,
    record: dict[str, Any],
    retrieved_ids: list[Any],
    ks: list[int],
) -> dict[str, Any]:
    gold_grades: dict[tuple[str, str], int] = {}
    for item in record["expected_chunk_ids"]:
        key = chunk_key(item["arxiv_id"], item["chunk_id"])
        grade = max(0, int(item.get("relevance_candidate", 1)))
        gold_grades[key] = max(gold_grades.get(key, 0), grade)

    gold_ids = {key for key, grade in gold_grades.items() if grade > 0}

    # Deduplicate retrieved chunks while preserving rank order.
    retrieved: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_chunk in retrieved_ids:
        key = normalize_retrieved_chunk(raw_chunk)
        if key not in seen:
            retrieved.append(key)
            seen.add(key)

    first_relevant_rank = next(
        (rank for rank, key in enumerate(retrieved, start=1) if key in gold_ids),
        None,
    )

    row: dict[str, Any] = {
        "question_id": question_id,
        "question": record.get("question", ""),
        "gold_relevant_count": len(gold_ids),
        "retrieved_count": len(retrieved),
        "relevant_retrieved_count": sum(key in gold_ids for key in retrieved),
        "MRR": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }

    for k in ks:
        top_k = retrieved[:k]
        relevant_hits = sum(key in gold_ids for key in top_k)
        precision = relevant_hits / k
        recall = relevant_hits / len(gold_ids) if gold_ids else 0.0

        actual_grades = [gold_grades.get(key, 0) for key in top_k]
        ideal_grades = sorted(gold_grades.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_grades)
        ndcg = dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

        row[f"Precision@{k}"] = precision
        row[f"Recall@{k}"] = recall
        row[f"nDCG@{k}"] = ndcg

    return row


def evaluate_retrieval(
    dataset: dict[str, dict[str, Any]],
    results: dict[str, list[Any]],
    ks: list[int],
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for question_id, record in dataset.items():
        if question_id not in results:
            missing_ids.append(question_id)
        rows.append(
            evaluate_question(
                question_id=question_id,
                record=record,
                retrieved_ids=results.get(question_id, []),
                ks=ks,
            )
        )

    metric_keys = ["MRR"]
    for k in ks:
        metric_keys.extend([f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}"])

    aggregate = {
        key: mean(float(row[key]) for row in rows) if rows else 0.0
        for key in metric_keys
    }
    return aggregate, rows, missing_ids


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-paper retrieval metrics only; no LLM judge."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "reranked"],
        default=["baseline", "reranked"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ks = sorted(set(args.k))
    if not ks or any(k <= 0 for k in ks):
        print("Error: K values must be positive.", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(args.dataset)

        for mode in args.modes:
            print(f"\\n{'=' * 60}\\nEvaluating mode: {mode}\\n{'=' * 60}")
            retrieval_path = args.predictions_dir / mode / "retrieved_chunks.json"
            retrieval_results = load_retrieval_results(retrieval_path)

            aggregate, rows, missing_ids = evaluate_retrieval(dataset, retrieval_results, ks)
            output_dir = args.output_dir / mode
            report = {
                "mode": mode,
                "dataset_path": str(args.dataset),
                "retrieval_results_path": str(retrieval_path),
                "question_count": len(dataset),
                "questions_missing_retrieval_results": missing_ids,
                "metrics_macro_average": aggregate,
                "per_question_csv": str(output_dir / "retrieval_per_question.csv"),
            }

            write_json(output_dir / "retrieval_eval.json", report)
            write_csv(output_dir / "retrieval_per_question.csv", rows)

            print("\\nRetrieval macro-average metrics:")
            for metric, value in aggregate.items():
                print(f"  {metric:>12}: {value:.4f}")
            print(f"\\nResults saved to: {output_dir}")

    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("\\nRetrieval evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
