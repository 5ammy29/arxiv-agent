#!/usr/bin/env python3
"""Evaluate architecture/methodology retrieval metrics only (no LLM-as-a-Judge)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_DEFAULT = (
    ROOT / "data/eval/qwen/architecture_methodology/datasets/"
    "architecture_methodology_retrieval_eval_dataset.json"
)
PREDICTIONS_DEFAULT = ROOT / "data/eval/qwen/architecture_methodology/predictions"
RESULTS_DEFAULT = ROOT / "data/eval/qwen/architecture_methodology/results"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object/dictionary at {path}")
    return data

def normalize_arxiv_id(value: Any) -> str:
    return str(value).strip()

def normalize_chunk_id(value: Any) -> str:
    # Keep IDs as strings for stable matching regardless of CSV/JSON number type.
    return str(value).strip()

def chunk_key(item: dict) -> tuple[str, str]:
    return normalize_arxiv_id(item["arxiv_id"]), normalize_chunk_id(item["chunk_id"])

def get_expected(record: dict) -> dict[tuple[str, str], float]:
    expected = {}
    for item in record.get("expected_chunk_ids", []):
        key = chunk_key(item)
        grade = item.get("relevance_candidate", item.get("relevance", 1))
        try:
            grade = float(grade)
        except (TypeError, ValueError):
            grade = 1.0
        expected[key] = max(0.0, grade)
    return expected

def get_retrieved(record: dict) -> list[dict]:
    """Read the retrieval IDs from the saved predictions record."""
    ids = record.get("retrieved_chunk_ids", [])
    if ids:
        return ids
    # Fallback if the IDs list isn't present but full chunk objects are.
    return record.get("retrieved_chunks", [])

def retrieval_metrics(expected: dict[tuple[str, str], float], retrieved: list[dict], ks: list[int]) -> dict:
    ranked_keys = [chunk_key(item) for item in retrieved]
    # De-duplicate while preserving rank order.
    seen = set()
    unique_ranked = []
    for key in ranked_keys:
        if key not in seen:
            seen.add(key)
            unique_ranked.append(key)

    relevant_keys = {key for key, grade in expected.items() if grade > 0}
    first_relevant_rank = next(
        (rank for rank, key in enumerate(unique_ranked, start=1) if key in relevant_keys),
        None,
    )
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    output = {"MRR": mrr}
    for k in ks:
        top_k = unique_ranked[:k]
        hit_grades = [expected.get(key, 0.0) for key in top_k]
        hits = sum(1 for grade in hit_grades if grade > 0)
        precision = hits / k
        recall = hits / len(relevant_keys) if relevant_keys else 0.0

        dcg = sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(hit_grades, start=1)
            if grade > 0
        )
        ideal_grades = sorted(expected.values(), reverse=True)[:k]
        idcg = sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades, start=1)
            if grade > 0
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        output[f"Precision@{k}"] = precision
        output[f"Recall@{k}"] = recall
        output[f"nDCG@{k}"] = ndcg
    return output

def macro_average(rows: list[dict], keys: list[str]) -> dict:
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def evaluate_mode(
    mode: str,
    dataset: dict,
    predictions_dir: Path,
    results_dir: Path,
    ks: list[int],
) -> None:
    retrieval_path = predictions_dir / mode / "retrieved_chunks.json"
    retrieval_predictions = load_json(retrieval_path)

    rows = []
    missing_ids = []
    metric_keys = ["MRR"] + [
        metric for k in ks
        for metric in (f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}")
    ]

    print(f"\\n{'=' * 60}\\nEvaluating mode: {mode}\\n{'=' * 60}")

    for question_id, dataset_record in dataset.items():
        prediction_record = retrieval_predictions.get(question_id)
        if prediction_record is None:
            missing_ids.append(question_id)
            prediction_record = {}

        expected = get_expected(dataset_record)
        retrieved = get_retrieved(prediction_record)
        metrics = retrieval_metrics(expected, retrieved, ks)
        rows.append({
            "question_id": question_id,
            "question": dataset_record.get("question", ""),
            **metrics,
        })

    summary = macro_average(rows, metric_keys)
    print("\\nRetrieval macro-average metrics:")
    for key, value in summary.items():
        print(f"{key:>14}: {value:.4f}")

    output_dir = results_dir / mode
    write_json(
        output_dir / "retrieval_eval.json",
        {
            "mode": mode,
            "dataset_path": str(DATASET_DEFAULT),
            "retrieval_predictions_path": str(retrieval_path),
            "questions_evaluated": len(rows),
            "questions_missing_retrieval_predictions": missing_ids,
            "metrics_macro_average": summary,
            "ks": ks,
        },
    )
    write_csv(output_dir / "retrieval_per_question.csv", rows)
    print(f"\\nResults saved to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate architecture/methodology retrieval metrics only; no LLM judge."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument(
        "--modes", nargs="+", choices=["baseline", "reranked"],
        default=["baseline", "reranked"],
    )
    args = parser.parse_args()

    if not args.k or any(k <= 0 for k in args.k):
        parser.error("All K values must be positive.")
    ks = sorted(set(args.k))

    try:
        dataset = load_json(args.dataset)
        print(f"Dataset: {args.dataset}")
        print(f"Questions in dataset: {len(dataset)}")
        print(f"K values: {ks}")
        for mode in args.modes:
            evaluate_mode(
                mode=mode,
                dataset=dataset,
                predictions_dir=args.predictions_dir,
                results_dir=args.results_dir,
                ks=ks,
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("\\nRetrieval evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
