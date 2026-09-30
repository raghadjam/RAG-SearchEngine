import os

from inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def hybrid_score(
    bm25_score: float,
    semantic_score: float,
    alpha: float
) -> float:
    return (
        alpha * bm25_score
        + (1 - alpha) * semantic_score
    )


def rrf_score(rank: int, k: int) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(
            documents
        )

        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(
        self,
        query: str,
        limit: int
    ) -> list[tuple]:
        self.idx.load()

        return self.idx.bm25_search(
            query,
            limit
        )

    def weighted_search(
        self,
        query: str,
        alpha: float,
        limit: int = 5
    ) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(
            query, search_limit
        )

        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [
            result["score"] for result in semantic_results
        ]

        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_semantic = normalize_scores(semantic_scores)

        results = {}

        for i, (doc_id, _) in enumerate(bm25_results):
            movie = self.idx.docmap[doc_id]

            results[doc_id] = {
                "id": doc_id,
                "title": movie["title"],
                "document": movie["description"],
                "bm25": normalized_bm25[i],
                "semantic": 0.0
            }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]

            if doc_id not in results:
                results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25": 0.0,
                    "semantic": normalized_semantic[i]
                }
            else:
                results[doc_id]["semantic"] = normalized_semantic[i]

        final_results = []

        for result in results.values():
            score = hybrid_score(
                result["bm25"],
                result["semantic"],
                alpha
            )

            final_results.append(
                {
                    "id": result["id"],
                    "title": result["title"],
                    "document": result["document"],
                    "score": score,
                    "bm25": result["bm25"],
                    "semantic": result["semantic"]
                }
            )

        final_results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return final_results[:limit]

    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 5
    ) -> list[dict]:
        search_limit = limit * 500

        bm25_results = self._bm25_search(query, search_limit)
        semantic_results = self.semantic_search.search_chunks(
            query, search_limit
        )

        combined = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            movie = self.idx.docmap[doc_id]

            combined[doc_id] = {
                "id": doc_id,
                "title": movie["title"],
                "document": movie["description"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "score": rrf_score(rank, k)
            }

        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]

            if doc_id not in combined:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "score": rrf_score(rank, k)
                }
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["score"] += rrf_score(rank, k)

        final_results = list(combined.values())

        final_results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return final_results[:limit]
from sentence_transformers import CrossEncoder


def rerank_cross_encoder(query: str, results: list[dict]) -> list[dict]:
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L2-v2",
        device="cpu"
    )

    pairs = []

    for doc in results:
        pairs.append(
            [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        )

    scores = cross_encoder.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        reranked.append(
            {
                **result,
                "cross_encoder_score": float(score),
            }
        )

    reranked.sort(
        key=lambda result: result["cross_encoder_score"],
        reverse=True
    )

    return reranked