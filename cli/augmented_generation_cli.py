import argparse
import json

from lib.hybrid_search import HybridSearch
from lib.llm import (
    answer_question,
    answer_with_citations,
    generate_answer,
    summarize_results,
)


def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Augmented Generation CLI"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize results)"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query to summarize"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + cite sources in answer)"
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for citation-aware answer"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to use as sources",
    )

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + conversational answer)"
    )
    question_parser.add_argument(
        "question", type=str, help="Question to answer"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to use as context",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query

            documents = load_movies()
            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                query,
                60,
                5
            )

            answer = generate_answer(query, results)

            print("Search Results:")
            for result in results:
                print(f"- {result['title']}")

            print()
            print("RAG Response:")
            print(answer)

        case "summarize":
            query = args.query
            limit = args.limit

            documents = load_movies()
            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                query,
                60,
                limit
            )

            summary = summarize_results(query, results)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print()
            print("LLM Summary:")
            print(summary)

        case "citations":
            query = args.query
            limit = args.limit

            documents = load_movies()
            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                query,
                60,
                limit
            )

            answer = answer_with_citations(query, results)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print()
            print("LLM Answer:")
            print(answer)

        case "question":
            question = args.question
            limit = args.limit

            documents = load_movies()
            hybrid_search = HybridSearch(documents)

            results = hybrid_search.rrf_search(
                question,
                60,
                limit
            )

            answer = answer_question(question, results)

            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")

            print()
            print("Answer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()