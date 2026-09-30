import argparse
import json

from lib.hybrid_search import (
    HybridSearch,
    normalize_scores,
    rerank_cross_encoder,
)

from lib.llm import (
    evaluate_results,
    expand_query,
    fix_spelling,
    rerank_batch,
    rerank_individual,
    rewrite_query,
)

def load_movies() -> list[dict]:
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Min-max normalize a list of scores"
    )

    normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="*",
        help="Scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Search using weighted hybrid search"
    )

    weighted_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight given to BM25 scores"
    )

    weighted_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search",
        help="Search using RRF hybrid search"
    )

    rrf_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF constant controlling rank weighting"
    )

    rrf_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method to apply to RRF results",
    )

    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate RRF search results using an LLM",
    )
        

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)

            for score in normalized:
                print(f"* {score:.4f}")

        case "weighted-search":
            documents = load_movies()

            hybrid_search = HybridSearch(documents)

            results = hybrid_search.weighted_search(
                args.query,
                args.alpha,
                args.limit
            )

            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                print(
                    f"  Hybrid Score: "
                    f"{result['score']:.3f}"
                )
                print(
                    f"  BM25: {result['bm25']:.3f}, "
                    f"Semantic: {result['semantic']:.3f}"
                )
                print(
                    f"  {result['document']}"
                )

        case "rrf-search":
            documents = load_movies()

            hybrid_search = HybridSearch(documents)

            query = args.query

            if args.enhance == "spell":
                enhanced_query = fix_spelling(query)
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "rewrite":
                enhanced_query = rewrite_query(query)
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query
            elif args.enhance == "expand":
                enhanced_query = expand_query(query)
                print(
                    f"Enhanced query ({args.enhance}): "
                    f"'{query}' -> '{enhanced_query}'\n"
                )
                query = enhanced_query

            rrf_limit = args.limit

            if args.rerank_method in ("individual", "batch", "cross_encoder"):
                rrf_limit = args.limit * 5

            results = hybrid_search.rrf_search(
                query,
                args.k,
                rrf_limit
            )

            if args.rerank_method == "individual":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using individual method...\n"
                )
                print(
                    f"Reciprocal Rank Fusion Results for "
                    f"'{query}' (k={args.k}):\n"
                )

                results = rerank_individual(query, results)
                results = results[:args.limit]

                for i, result in enumerate(results, start=1):
                    bm25_rank = (
                        result["bm25_rank"]
                        if result["bm25_rank"] is not None
                        else "N/A"
                    )
                    semantic_rank = (
                        result["semantic_rank"]
                        if result["semantic_rank"] is not None
                        else "N/A"
                    )

                    print(f"{i}. {result['title']}")
                    print(
                        f"   Re-rank Score: "
                        f"{result['rerank_score']:.3f}/10"
                    )
                    print(f"   RRF Score: {result['score']:.3f}")
                    print(
                        f"   BM25 Rank: {bm25_rank}, "
                        f"Semantic Rank: {semantic_rank}"
                    )
                    print(f"   {result['document'][:100]}...")

            elif args.rerank_method == "batch":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using batch method...\n"
                )
                print(
                    f"Reciprocal Rank Fusion Results for "
                    f"'{query}' (k={args.k}):\n"
                )

                results = rerank_batch(query, results)
                results = results[:args.limit]

                for i, result in enumerate(results, start=1):
                    bm25_rank = (
                        result["bm25_rank"]
                        if result["bm25_rank"] is not None
                        else "N/A"
                    )
                    semantic_rank = (
                        result["semantic_rank"]
                        if result["semantic_rank"] is not None
                        else "N/A"
                    )

                    print(f"{i}. {result['title']}")
                    print(f"   Re-rank Rank: {result['rerank_rank']}")
                    print(f"   RRF Score: {result['score']:.3f}")
                    print(
                        f"   BM25 Rank: {bm25_rank}, "
                        f"Semantic Rank: {semantic_rank}"
                    )
                    print(f"   {result['document'][:100]}...")

            elif args.rerank_method == "cross_encoder":
                print(
                    f"Re-ranking top {args.limit} results "
                    f"using cross_encoder method...\n"
                )
                print(
                    f"Reciprocal Rank Fusion Results for "
                    f"'{query}' (k={args.k}):\n"
                )

                results = rerank_cross_encoder(query, results)
                results = results[:args.limit]

                for i, result in enumerate(results, start=1):
                    bm25_rank = (
                        result["bm25_rank"]
                        if result["bm25_rank"] is not None
                        else "N/A"
                    )
                    semantic_rank = (
                        result["semantic_rank"]
                        if result["semantic_rank"] is not None
                        else "N/A"
                    )

                    print(f"{i}. {result['title']}")
                    print(
                        f"   Cross Encoder Score: "
                        f"{result['cross_encoder_score']:.3f}"
                    )
                    print(f"   RRF Score: {result['score']:.3f}")
                    print(
                        f"   BM25 Rank: {bm25_rank}, "
                        f"Semantic Rank: {semantic_rank}"
                    )
                    print(f"   {result['document'][:100]}...")

            else:
                for i, result in enumerate(results, start=1):
                    bm25_rank = (
                        result["bm25_rank"]
                        if result["bm25_rank"] is not None
                        else "N/A"
                    )
                    semantic_rank = (
                        result["semantic_rank"]
                        if result["semantic_rank"] is not None
                        else "N/A"
                    )

                    print(f"{i}. {result['title']}")
                    print(f"  RRF Score: {result['score']:.3f}")
                    print(
                        f"  BM25 Rank: {bm25_rank}, "
                        f"Semantic Rank: {semantic_rank}"
                    )
                    print(f"  {result['document'][:100]}...")
            if args.evaluate:
                scores = evaluate_results(query, results)
                print()
                for i, (result, score) in enumerate(zip(results, scores), start=1):
                    print(f"{i}. {result['title']}: {score}/3")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()