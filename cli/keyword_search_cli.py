import argparse
import math

from inverted_index import (
    BM25_B,
    BM25_K1,
    InvertedIndex,
    tokenize_term,
    tokenize_text,
)


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query):
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files do not exist. Run the build command first.")
        return

    query_tokens = tokenize_text(query)

    results = []

    for query_token in query_tokens:
        documents = index.get_documents(query_token)

        for doc_id in documents:
            if doc_id not in results:
                results.append(doc_id)

            if len(results) == 5:
                break

        if len(results) == 5:
            break

    print(f"Searching for: {query}")

    for doc_id in results:
        movie = index.docmap[doc_id]
        print(f"{movie['title']} ({doc_id})")


def tf_command(doc_id, term):
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files do not exist. Run the build command first.")
        return

    token = tokenize_term(term)
    frequency = index.get_tf(doc_id, token)

    print(frequency)


def idf_command(term):
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files do not exist. Run the build command first.")
        return

    token = tokenize_term(term)

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(token))

    idf = math.log(
        (total_doc_count + 1) / (term_match_doc_count + 1)
    )

    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(doc_id, term):
    index = InvertedIndex()

    try:
        index.load()
    except FileNotFoundError:
        print("Error: index files do not exist. Run the build command first.")
        return

    token = tokenize_term(term)

    tf = index.get_tf(doc_id, token)

    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(token))

    idf = math.log(
        (total_doc_count + 1) / (term_match_doc_count + 1)
    )

    tf_idf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf_command(term):
    index = InvertedIndex()

    index.load()

    token = tokenize_term(term)

    return index.get_bm25_idf(token)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()

    index.load()

    token = tokenize_term(term)

    return index.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(query, limit):
    index = InvertedIndex()

    index.load()

    results = index.bm25_search(query, limit)

    for position, (doc_id, score) in enumerate(results, start=1):
        movie = index.docmap[doc_id]

        print(
            f"{position}. ({doc_id}) {movie['title']} - Score: {score:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    tf_parser = subparsers.add_parser(
        "tf",
        help="Get term frequency"
    )
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    tf_parser.add_argument(
        "term",
        type=str,
        help="Term"
    )

    idf_parser = subparsers.add_parser(
        "idf",
        help="Get inverse document frequency"
    )
    idf_parser.add_argument(
        "term",
        type=str,
        help="Term"
    )

    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Get TF-IDF score"
    )
    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    tfidf_parser.add_argument(
        "term",
        type=str,
        help="Term"
    )

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get the BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    bm25_tf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 TF score for"
    )
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=BM25_B,
        help="Tunable BM25 b parameter"
    )

    bm25_search_parser = subparsers.add_parser(
        "bm25search",
        help="Search movies using full BM25 scoring"
    )
    bm25_search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    bm25_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results"
    )

    subparsers.add_parser(
        "build",
        help="Build the inverted index"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            build_command()

        case "search":
            search_command(args.query)

        case "tf":
            tf_command(args.doc_id, args.term)

        case "idf":
            idf_command(args.term)

        case "tfidf":
            tfidf_command(args.doc_id, args.term)

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(
                f"BM25 IDF score of '{args.term}': {bm25idf:.2f}"
            )

        case "bm25tf":
            bm25tf = bm25_tf_command(
                args.doc_id,
                args.term,
                args.k1,
                args.b
            )
            print(
                f"BM25 TF score of '{args.term}' "
                f"in document '{args.doc_id}': {bm25tf:.2f}"
            )

        case "bm25search":
            bm25_search_command(
                args.query,
                args.limit
            )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()