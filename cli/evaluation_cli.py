import argparse
import json

from lib.hybrid_search import HybridSearch


def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def load_golden_dataset() -> list[dict]:
    with open("data/golden_dataset.json", "r") as file:
        data = json.load(file)

    return data["test_cases"]


def precision_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    if not retrieved_titles:
        return 0.0

    relevant_set = set(relevant_titles)

    hits = sum(
        1 for title in retrieved_titles if title in relevant_set
    )

    return hits / len(retrieved_titles)


def recall_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    if not relevant_titles:
        return 0.0

    retrieved_set = set(retrieved_titles)

    hits = sum(
        1 for title in relevant_titles if title in retrieved_set
    )

    return hits / len(relevant_titles)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    documents = load_movies()
    hybrid_search = HybridSearch(documents)

    golden_dataset = load_golden_dataset()

    print(f"k={limit}\n")

    for case in golden_dataset:
        query = case["query"]
        relevant_titles = case["relevant_docs"]

        results = hybrid_search.rrf_search(
            query,
            60,
            limit
        )

        retrieved_titles = [result["title"] for result in results]

        precision = precision_at_k(retrieved_titles, relevant_titles)
        recall = recall_at_k(retrieved_titles, relevant_titles)
        f1 = f1_score(precision, recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()