import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag_answer import answer_question
from retrieval import native_hybrid_search


def load_eval_questions(path: Path) -> List[Dict[str, str]]:
    """
    Charge un fichier JSON ou JSONL contenant au minimum :
    - question
    - ground_truth

    Exemple JSONL :
    {"question": "Quelle est la date de la facture pour le part number 822-1821-001 ?", "ground_truth": "La date est ..."}
    """
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return data["examples"]

    raise ValueError("Le fichier d'évaluation doit être une liste JSON ou un JSONL.")


def retrieved_contexts(question: str, k: int) -> List[str]:
    hits = native_hybrid_search(query=question, k=k)
    return [
        hit.get("_source", {}).get("text", "")
        for hit in hits
        if hit.get("_source", {}).get("text")
    ]

def build_ragas_dataset(eval_rows: List[Dict[str, str]], k: int) -> Dataset:
    ragas_rows: List[Dict[str, Any]] = []

    for row in eval_rows:
        question = row["question"]
        ground_truth = row["ground_truth"]

        contexts = retrieved_contexts(question, k=k)
        answer = answer_question(question)

        ragas_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return Dataset.from_list(ragas_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS du pipeline OpenSearch existant."
    )
    parser.add_argument(
        "--eval-file",
        required=True,
        help="Fichier JSON/JSONL avec les champs question et ground_truth.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default="ragas_report.csv")
    args = parser.parse_args()

    eval_rows = load_eval_questions(Path(args.eval_file))
    dataset = build_ragas_dataset(eval_rows, k=args.k)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print(result)

    output_path = Path(args.output)
    result.to_pandas().to_csv(output_path, index=False)
    print(f"Rapport RAGAS enregistré dans : {output_path}")


if __name__ == "__main__":
    main()
