import argparse
import json

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    semantic_chunk,
    verify_embeddings,
    verify_model,
)


def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    subparsers.add_parser(
        "verify",
        help="Verify the semantic search model"
    )

    embed_parser = subparsers.add_parser(
        "embed_text",
        help="Generate an embedding for text"
    )
    embed_parser.add_argument(
        "text",
        type=str,
        help="Text to embed"
    )

    subparsers.add_parser(
        "verify_embeddings",
        help="Verify movie embeddings"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query",
        help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument(
        "query",
        type=str,
        help="Search query to embed"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies semantically"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Split text into fixed-size chunks"
    )
    chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk"
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words shared between chunks"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="Split text into sentence-based chunks"
    )
    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences shared between chunks"
    )

    subparsers.add_parser(
        "embed_chunks",
        help="Generate embeddings for semantic chunks"
    )

    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="Search movies using chunk embeddings"
    )
    search_chunked_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            semantic_search = SemanticSearch()
            documents = load_movies()
            semantic_search.load_or_create_embeddings(documents)

            results = semantic_search.search(
                args.query,
                args.limit
            )

            for position, result in enumerate(results, start=1):
                print(
                    f"{position}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"  {result['description']}")
                print()

        case "chunk":
            words = args.text.split()

            if args.overlap >= args.chunk_size:
                raise ValueError(
                    "Overlap must be smaller than chunk size"
                )

            print(f"Chunking {len(args.text)} characters")

            step = args.chunk_size - args.overlap

            for i in range(0, len(words), step):
                chunk = " ".join(
                    words[i:i + args.chunk_size]
                )
                print(f"{(i // step) + 1}. {chunk}")

        case "semantic_chunk":
            # Now delegates to the shared, hardened implementation
            chunks = semantic_chunk(
                args.text,
                max_chunk_size=args.max_chunk_size,
                overlap=args.overlap
            )

            print(
                f"Semantically chunking {len(args.text)} characters"
            )

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed_chunks":
            documents = load_movies()

            semantic_search = ChunkedSemanticSearch()

            embeddings = (
                semantic_search.load_or_create_chunk_embeddings(
                    documents
                )
            )

            print(
                f"Generated {len(embeddings)} chunked embeddings"
            )

        case "search_chunked":
            documents = load_movies()

            semantic_search = ChunkedSemanticSearch()

            semantic_search.load_or_create_chunk_embeddings(
                documents
            )

            results = semantic_search.search_chunks(
                args.query,
                args.limit
            )

            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"   {result['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()