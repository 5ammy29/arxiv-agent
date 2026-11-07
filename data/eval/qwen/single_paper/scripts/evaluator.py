from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import requests


BASE_DIR = Path("data/eval/qwen/single_paper")

DEFAULT_DATASET = (
    BASE_DIR / "datasets" / "single_paper_retrieval_eval_dataset.json"
)
DEFAULT_RETRIEVAL_RESULTS = (
    BASE_DIR
    / "predictions"
    / "retrieval"
    / "qwen_single_paper_retrieval_baseline_predictions.json"
)
DEFAULT_CORPUS = BASE_DIR / "corpus" / "session_chunks.csv"
DEFAULT_ANSWERS = (
    BASE_DIR / "predictions" / "answers" / "qwen_single_paper_baseline_answers.json"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
DEFAULT_RETRIEVED_CHUNKS = (
    BASE_DIR / "predictions" / "retrieval" / "retrieved_chunks.json"
)
DEFAULT_JUDGE_OUTPUT = (
    BASE_DIR
    / "predictions"
    / "answers"
    / "qwen_single_paper_baseline_judge_scores.json"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def normalize_chunk_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list.")

    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {index} is not an object.")

        if not (record.get("question_id") or record.get("id")):
            raise ValueError(f"Dataset record {index} has no question_id/id.")

        if not isinstance(record.get("relevant_chunks"), list):
            raise ValueError(
                f"Dataset record {index} has no relevant_chunks list."
            )

    return data


def load_retrieval_results(path: Path) -> dict[str, list[Any]]:
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError("Retrieval results must map question IDs to chunk IDs.")

    results = {}

    for question_id, chunk_ids in data.items():
        if not isinstance(chunk_ids, list):
            raise ValueError(
                f"Retrieval results for {question_id} must be a list."
            )

        results[str(question_id)] = chunk_ids

    return results


def load_corpus(path: Path) -> dict[str, str]:
    """Load chunk ID -> chunk text from CSV."""

    corpus = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        id_column = next(
            (
                name
                for name in ("chunk_id", "id")
                if name in reader.fieldnames
            ),
            None,
        )

        text_column = next(
            (
                name
                for name in ("text", "chunk_text", "content")
                if name in reader.fieldnames
            ),
            None,
        )

        if id_column is None or text_column is None:
            raise ValueError(
                "Corpus CSV must contain a chunk ID column "
                "(chunk_id/id) and text column (text/chunk_text/content)."
            )

        for row in reader:
            chunk_id = normalize_chunk_id(row[id_column])
            corpus[chunk_id] = row[text_column]

    return corpus


def load_answers(path: Path) -> dict[str, str]:
    """Accept either {question_id: answer} or a list of answer records."""

    data = load_json(path)
    answers = {}

    if isinstance(data, dict):
        for question_id, value in data.items():
            if isinstance(value, str):
                answers[str(question_id)] = value
            elif isinstance(value, dict):
                answer = value.get("answer") or value.get("generated_answer")
                if answer is not None:
                    answers[str(question_id)] = str(answer)

    elif isinstance(data, list):
        for record in data:
            question_id = str(
                record.get("question_id") or record.get("id")
            )
            answer = record.get("answer") or record.get("generated_answer")

            if answer is not None:
                answers[question_id] = str(answer)

    else:
        raise ValueError("Answers JSON must be an object or list.")

    return answers


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def dcg(grades: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def evaluate_question(
    record: dict[str, Any],
    retrieved_ids: list[Any],
    ks: list[int],
) -> dict[str, Any]:
    question_id = str(record.get("question_id") or record.get("id"))

    gold_ids = set()
    grades_by_id = {}

    for chunk in record["relevant_chunks"]:
        chunk_id = normalize_chunk_id(chunk["chunk_id"])
        grade = max(0, int(chunk.get("relevance_candidate", 1)))

        gold_ids.add(chunk_id)
        grades_by_id[chunk_id] = max(
            grades_by_id.get(chunk_id, 0),
            grade,
        )

    # Deduplicate while preserving rank order.
    retrieved = []
    seen = set()

    for value in retrieved_ids:
        chunk_id = normalize_chunk_id(value)

        if chunk_id not in seen:
            retrieved.append(chunk_id)
            seen.add(chunk_id)

    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )

    row = {
        "question_id": question_id,
        "question": record.get("question", ""),
        "gold_relevant_count": len(gold_ids),
        "retrieved_count": len(retrieved),
        "relevant_retrieved_count": sum(
            chunk_id in gold_ids for chunk_id in retrieved
        ),
        "MRR": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }

    for k in ks:
        top_k = retrieved[:k]
        relevant_hits = sum(chunk_id in gold_ids for chunk_id in top_k)

        precision = relevant_hits / k
        recall = relevant_hits / len(gold_ids) if gold_ids else 0.0

        actual_grades = [
            grades_by_id.get(chunk_id, 0)
            for chunk_id in top_k
        ]
        ideal_grades = sorted(
            grades_by_id.values(),
            reverse=True,
        )[:k]

        ideal_dcg = dcg(ideal_grades)
        ndcg = dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

        row[f"Precision@{k}"] = precision
        row[f"Recall@{k}"] = recall
        row[f"nDCG@{k}"] = ndcg

    return row


def evaluate_retrieval(
    dataset: list[dict[str, Any]],
    results: dict[str, list[Any]],
    ks: list[int],
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    rows = []
    missing_ids = []

    for record in dataset:
        question_id = str(record.get("question_id") or record.get("id"))

        if question_id not in results:
            missing_ids.append(question_id)

        rows.append(
            evaluate_question(
                record,
                results.get(question_id, []),
                ks,
            )
        )

    metric_keys = ["MRR"]

    for k in ks:
        metric_keys.extend(
            [f"Precision@{k}", f"Recall@{k}", f"nDCG@{k}"]
        )

    aggregate = {
        key: mean(float(row[key]) for row in rows) if rows else 0.0
        for key in metric_keys
    }

    return aggregate, rows, missing_ids


# ---------------------------------------------------------------------------
# Retrieved chunk persistence
# ---------------------------------------------------------------------------

def save_retrieved_chunks(
    dataset: list[dict[str, Any]],
    results: dict[str, list[Any]],
    corpus: dict[str, str],
    output_path: Path,
) -> None:
    output = []

    for record in dataset:
        question_id = str(record.get("question_id") or record.get("id"))
        retrieved_ids = results.get(question_id, [])

        chunks = []

        for rank, raw_chunk_id in enumerate(retrieved_ids, start=1):
            chunk_id = normalize_chunk_id(raw_chunk_id)

            chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "text": corpus.get(chunk_id),
                    "text_found": chunk_id in corpus,
                }
            )

        output.append(
            {
                "question_id": question_id,
                "question": record.get("question", ""),
                "retrieved_chunks": chunks,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    retrieved_context: str,
    model: str,
    ollama_url: str,
) -> dict[str, Any]:
    prompt = f"""
You are an impartial evaluator of answers produced by a research assistant.

Evaluate the generated answer using the question, reference answer, and
retrieved context.

Score each criterion from 1 to 5:
- correctness: factual correctness compared with the reference answer
- faithfulness: whether claims are supported by the retrieved context
- relevance: whether the answer directly addresses the question

Use these score meanings:
1 = very poor
2 = poor
3 = partially acceptable
4 = good
5 = excellent

Do not reward unsupported claims. If the context does not contain enough
information, an answer should acknowledge that limitation.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{generated_answer}

Return ONLY valid JSON in this format:
{{
  "correctness": 1,
  "faithfulness": 1,
  "relevance": 1,
  "rationale": "Brief explanation"
}}
""".strip()

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=300,
    )

    response.raise_for_status()
    raw_response = response.json()["response"]
    scores = json.loads(raw_response)

    for key in ("correctness", "faithfulness", "relevance"):
        score = int(scores[key])

        if not 1 <= score <= 5:
            raise ValueError(f"Judge returned invalid {key} score: {score}")

        scores[key] = score

    scores["overall"] = mean(
        scores[key]
        for key in ("correctness", "faithfulness", "relevance")
    )

    return scores


def evaluate_answers(
    dataset: list[dict[str, Any]],
    answers: dict[str, str],
    retrieved_chunks: dict[str, list[Any]],
    corpus: dict[str, str],
    model: str,
    ollama_url: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    rows = []
    missing_ids = []

    for record in dataset:
        question_id = str(record.get("question_id") or record.get("id"))

        generated_answer = answers.get(question_id)
        reference_answer = record.get("reference_answer")

        if not generated_answer or not reference_answer:
            missing_ids.append(question_id)
            continue

        context_chunks = []

        for raw_chunk_id in retrieved_chunks.get(question_id, []):
            chunk_id = normalize_chunk_id(raw_chunk_id)
            text = corpus.get(chunk_id)

            if text:
                context_chunks.append(text)

        context = "\n\n".join(context_chunks)

        scores = judge_answer(
            question=record.get("question", ""),
            reference_answer=str(reference_answer),
            generated_answer=generated_answer,
            retrieved_context=context,
            model=model,
            ollama_url=ollama_url,
        )

        rows.append(
            {
                "question_id": question_id,
                "question": record.get("question", ""),
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                **scores,
            }
        )

        print(f"Judged answer: {question_id}")

    metric_keys = [
        "correctness",
        "faithfulness",
        "relevance",
        "overall",
    ]

    aggregate = {
        key: mean(float(row[key]) for row in rows) if rows else 0.0
        for key in metric_keys
    }

    return aggregate, rows, missing_ids


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval metrics and LLM-as-a-Judge answer quality."
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RETRIEVAL_RESULTS,
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--retrieved-chunks-output",
        type=Path,
        default=DEFAULT_RETRIEVED_CHUNKS,
    )
    parser.add_argument(
        "--judge-output",
        type=Path,
        default=DEFAULT_JUDGE_OUTPUT,
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
    )
    parser.add_argument("--judge-model", default="qwen2.5:3b")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
    )
    parser.add_argument("--name", default="retrieval_eval")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ks = sorted(set(args.k))

    if not ks or any(k <= 0 for k in ks):
        print("Error: K values must be positive.", file=sys.stderr)
        return 2

    try:
        dataset = load_dataset(args.dataset)
        retrieval_results = load_retrieval_results(args.results)
        corpus = load_corpus(args.corpus)

        retrieval_aggregate, retrieval_rows, missing_retrieval_ids = (
            evaluate_retrieval(dataset, retrieval_results, ks)
        )

        save_retrieved_chunks(
            dataset,
            retrieval_results,
            corpus,
            args.retrieved_chunks_output,
        )

        retrieval_report = {
            "dataset_path": str(args.dataset),
            "retrieval_results_path": str(args.results),
            "question_count": len(dataset),
            "questions_missing_retrieval_results": missing_retrieval_ids,
            "metrics_macro_average": retrieval_aggregate,
            "per_question_csv": str(
                args.output_dir / f"{args.name}_per_question.csv"
            ),
        }

        write_json(
            args.output_dir / f"{args.name}.json",
            retrieval_report,
        )
        write_csv(
            args.output_dir / f"{args.name}_per_question.csv",
            retrieval_rows,
        )

        print("\nRetrieval macro-average metrics:")
        for metric, value in retrieval_aggregate.items():
            print(f"  {metric:>12}: {value:.4f}")

        print(f"\nRetrieved chunks saved: {args.retrieved_chunks_output}")

        # Answer judging is run when the generated answers file is available.
        if args.answers.exists():
            answers = load_answers(args.answers)

            judge_aggregate, judge_rows, missing_answer_ids = evaluate_answers(
                dataset=dataset,
                answers=answers,
                retrieved_chunks=retrieval_results,
                corpus=corpus,
                model=args.judge_model,
                ollama_url=args.ollama_url,
            )

            write_json(
                args.judge_output,
                {
                    "judge_model": args.judge_model,
                    "question_count": len(dataset),
                    "questions_judged": len(judge_rows),
                    "questions_missing_answers_or_references": missing_answer_ids,
                    "metrics_macro_average": judge_aggregate,
                    "per_question": judge_rows,
                },
            )

            write_json(
                args.output_dir / "answer_eval.json",
                {
                    "judge_model": args.judge_model,
                    "questions_judged": len(judge_rows),
                    "questions_missing_answers_or_references": missing_answer_ids,
                    "metrics_macro_average": judge_aggregate,
                },
            )

            write_csv(
                args.output_dir / "answer_eval_per_question.csv",
                judge_rows,
            )

            write_json(
                args.output_dir / "combined_eval.json",
                {
                    "retrieval_metrics": retrieval_aggregate,
                    "answer_judge_metrics": judge_aggregate,
                    "retrieval_question_count": len(dataset),
                    "answer_question_count": len(judge_rows),
                },
            )

            print("\nLLM-as-a-Judge macro-average scores (1–5):")
            for metric, value in judge_aggregate.items():
                print(f"  {metric:>12}: {value:.4f}")

            print(f"\nJudge scores saved: {args.judge_output}")

        else:
            print(
                "\nAnswers file not found. Retrieval evaluation completed; "
                "answer judging skipped."
            )
            print(f"Expected answers file: {args.answers}")

    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
