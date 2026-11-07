#!/usr/bin/env python3
"""Evaluate retrieval metrics for the numerical/evidence-specific dataset.

Supports expected_chunk_ids gold labels and retrieved_chunks.json produced by
generate_answer.py. Evaluates baseline and reranked retrieval separately.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
BASE_DIR = ROOT / "data/eval/qwen/numerical_evidence_specific"
DEFAULT_DATASET = BASE_DIR / "datasets/numerical_evidence_specific_retrieval_eval_dataset.json"
DEFAULT_PREDICTIONS = BASE_DIR / "predictions"
DEFAULT_OUTPUT = BASE_DIR / "evaluation"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dataset_records(data: Any) -> dict[str, dict]:
    """Normalize dataset JSON as either {qid: record} or [record, ...]."""
    if isinstance(data, dict):
        if "question_id" in data or "id" in data:
            data = [data]
        else:
            return {str(qid): record for qid, record in data.items()}
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list or a dictionary keyed by question ID.")

    records = {}
    for record in data:
        qid = str(record.get("question_id") or record.get("id") or "")
        if not qid:
            # Dataset variants may use a stable key-like field.
            qid = str(record.get("qid") or "")
        if not qid:
            raise ValueError("A dataset record is missing question_id/id/qid.")
        records[qid] = record
    return records


def chunk_key(item: Any) -> tuple[str, str]:
    """Return (paper ID, chunk ID) to avoid collisions across papers."""
    if isinstance(item, (int, str)):
        # Plain chunk ID accepted for single-paper output formats.
        return ("", str(item))
    if not isinstance(item, dict):
        raise TypeError(f"Unexpected chunk representation: {type(item).__name__}")

    chunk_id = item.get("chunk_id", item.get("id"))
    if chunk_id is None and isinstance(item.get("chunk"), dict):
        chunk_id = item["chunk"].get("chunk_id", item["chunk"].get("id"))
    if chunk_id is None:
        raise ValueError(f"Retrieved/gold chunk missing chunk_id: {item}")

    arxiv_id = item.get("arxiv_id", item.get("paper_id", ""))
    if isinstance(item.get("paper"), dict):
        arxiv_id = item["paper"].get("arxiv_id", arxiv_id)
    return (str(arxiv_id or ""), str(chunk_id))


def extract_gold(record: dict) -> dict[tuple[str, str], int]:
    raw_gold = record.get("expected_chunk_ids")
    if raw_gold is None:
        # Support a possible alternative naming.
        raw_gold = record.get("relevant_chunks", [])
    grades: dict[tuple[str, str], int] = {}
    for item in raw_gold:
        key = chunk_key(item)
        grade = max(0, int(item.get("relevance_candidate", item.get("relevance", 1)))
                    if isinstance(item, dict) else 1)
        grades[key] = max(grades.get(key, 0), grade)
    return grades


def normalize_retrieved_entry(entry: Any) -> list[Any]:
    """Unwrap common retrieved_chunks.json record layouts."""
    if isinstance(entry, list):
        return entry
    if not isinstance(entry, dict):
        raise TypeError(f"Unexpected retrieved entry type: {type(entry).__name__}")

    for field in ("retrieved_chunks", "chunks", "results", "retrieved"):
        value = entry.get(field)
        if isinstance(value, list):
            return value
    # Record itself may be a chunk.
    if "chunk_id" in entry or "id" in entry:
        return [entry]
    raise ValueError(f"Could not find chunk list in retrieved entry keys: {list(entry)}")


def load_retrieval_results(path: Path) -> dict[str, list[Any]]:
    data = load_json(path)

    if isinstance(data, dict):
        # Common direct mapping: {qid: [chunk objects]}
        results = {}
        for qid, entry in data.items():
            if isinstance(entry, dict) and not any(
                field in entry for field in ("retrieved_chunks", "chunks", "results", "retrieved", "chunk_id", "id")
            ):
                # A wrapped mapping with an outer results/data key.
                continue
            results[str(qid)] = normalize_retrieved_entry(entry)
        if results:
            return results
        for field in ("results", "retrieval_results", "predictions", "data"):
            if isinstance(data.get(field), (dict, list)):
                return load_retrieval_results_from_data(data[field])

    if isinstance(data, list):
        return load_retrieval_results_from_data(data)

    raise ValueError(f"Unsupported retrieval results format in {path}")


def load_retrieval_results_from_data(data: Any) -> dict[str, list[Any]]:
    if isinstance(data, dict):
        return {str(qid): normalize_retrieved_entry(entry) for qid, entry in data.items()}
    if isinstance(data, list):
        results = {}
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("List-form retrieval results must contain objects.")
            qid = str(row.get("question_id") or row.get("qid") or row.get("id") or "")
            if not qid:
                raise ValueError("Retrieval result row is missing question_id/qid/id.")
            results[qid] = normalize_retrieved_entry(row)
        return results
    raise ValueError("Unsupported nested retrieval results format.")


def dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def evaluate_one(qid: str, record: dict, raw_retrieved: list[Any], ks: list[int]) -> dict:
    gold_grades = extract_gold(record)
    gold_relevant = {key for key, grade in gold_grades.items() if grade > 0}

    # Deduplicate retrieved entries while retaining ranking order.
    retrieved_keys = []
    seen = set()
    for raw in raw_retrieved:
        key = chunk_key(raw)
        # If gold is single-paper and retrieved chunk omits arxiv_id, match by ID.
        if key[0] == "" and len({paper for paper, _ in gold_grades if paper}) == 1:
            key = (next(paper for paper, _ in gold_grades if paper), key[1])
        if key not in seen:
            retrieved_keys.append(key)
            seen.add(key)

    first_rank = next(
        (rank for rank, key in enumerate(retrieved_keys, start=1) if key in gold_relevant),
        None,
    )

    row = {
        "question_id": qid,
        "question": record.get("question", ""),
        "gold_relevant_count": len(gold_relevant),
        "retrieved_count": len(retrieved_keys),
        "relevant_retrieved_count": sum(key in gold_relevant for key in retrieved_keys),
        "MRR": 1.0 / first_rank if first_rank else 0.0,
    }

    for k in ks:
        ranked = retrieved_keys[:k]
        hits = sum(key in gold_relevant for key in ranked)
        row[f"Precision@{k}"] = hits / k if k else 0.0
        row[f"Recall@{k}"] = hits / len(gold_relevant) if gold_relevant else 0.0

        actual_grades = [gold_grades.get(key, 0) for key in ranked]
        ideal_grades = sorted(gold_grades.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_grades)
        row[f"nDCG@{k}"] = dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

    return row


def evaluate(dataset: dict[str, dict], results: dict[str, list[Any]], ks: list[int]):
    rows = []
    missing = []
    for qid, record in dataset.items():
        if qid not in results:
            missing.append(qid)
        rows.append(evaluate_one(qid, record, results.get(qid, []), ks))

    keys = ["MRR"]
    for k in ks:
        keys.extend([f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}"])
    macro = {key: mean(float(row[key]) for row in rows) if rows else 0.0 for key in keys}
    return macro, rows, missing


def evaluate_mode(dataset: dict[str, dict], retrieval_path: Path, output_dir: Path, mode: str, ks: list[int]):
    if not retrieval_path.exists():
        raise FileNotFoundError(f"{mode} retrieval output not found: {retrieval_path}")

    results = load_retrieval_results(retrieval_path)
    macro, rows, missing = evaluate(dataset, results, ks)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{mode}_retrieval_metrics.json"
    csv_path = output_dir / f"{mode}_retrieval_metrics_per_question.csv"

    report = {
        "mode": mode,
        "dataset_path": str(DEFAULT_DATASET),
        "retrieved_chunks_path": str(retrieval_path),
        "question_count": len(dataset),
        "questions_missing_retrieval_results": missing,
        "metrics_macro_average": macro,
        "evaluation_note": (
            "Metrics are macro-averaged over questions. Precision/Recall/MRR use binary relevance "
            "(any expected chunk with grade > 0). nDCG uses graded relevance_candidate values "
            "from expected_chunk_ids; missing grades default to 1."
        ),
        "per_question_csv": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else ["question_id", "question"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[{mode.upper()}]")
    print(f"Questions: {len(dataset)} | Missing retrieval results: {len(missing)}")
    for metric, value in macro.items():
        print(f"{metric}: {value:.4f}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="+", choices=["baseline", "reranked"], default=["baseline", "reranked"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()

    dataset = dataset_records(load_json(args.dataset))
    for mode in args.modes:
        retrieved_path = args.predictions_dir / mode / "retrieved_chunks.json"
        evaluate_mode(dataset, retrieved_path, args.output_dir, mode, args.k)


if __name__ == "__main__":
    main()
