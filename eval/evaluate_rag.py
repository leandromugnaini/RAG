from datasets import load_dataset, Dataset
from tqdm import tqdm
import requests
import json

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy
)


def main():
    # 1. Load only the first 50 test examples
    TEST_SPLIT = "test"
    N_SAMPLES = 50

    ds_full = load_dataset("neural-bridge/rag-dataset-1200", split=TEST_SPLIT)
    ds_50 = ds_full.select(range(N_SAMPLES))

    # 2. Ask your local RAG system for answers
    answers = []
    for question in tqdm(ds_50["question"], desc="Querying RAG"):
        response = requests.post(
            "http://localhost:8000/question",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"question": question}),
            timeout=30
        )
        answers.append(response.json()["answer"])

    # 3. Assemble a dataset for RAGAS
    rag_eval_ds = Dataset.from_dict({
        "question": ds_50["question"],
        "contexts": [[c] for c in ds_50["context"]],
        "answer": answers,
        "ground_truth": ds_50["answer"]
    })

    # 4. Evaluate
    report = evaluate(
        rag_eval_ds,
        metrics=[context_precision, faithfulness, answer_relevancy],
        column_map={
            "question": "question",
            "contexts": "contexts",
            "answer": "answer",
            "ground_truth": "ground_truth",
        },
    )

    print(report)

    df = report.to_pandas()
    df.to_csv("ragas_scores.csv", index=False)

    print("✅  Results saved to ragas_scores.csv")
    print(df)


if __name__ == "__main__":
    main()
