import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_chunk(
    text: str,
    max_chunk_size: int = 4,
    overlap: int = 0
) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError(
            "Overlap must be smaller than max chunk size"
        )

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not sentences[0].endswith(
        (".", "!", "?")
    ):
        sentences = [text]

    step = max_chunk_size - overlap
    chunks = []

    for i in range(0, len(sentences), step):
        chunk_sentences = [
            sentence.strip()
            for sentence in sentences[i:i + max_chunk_size]
        ]
        chunk_sentences = [s for s in chunk_sentences if s]

        if not chunk_sentences:
            continue

        chunks.append(" ".join(chunk_sentences))

    return chunks


class SemanticSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace")

        embedding = self.model.encode([text])

        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        texts = []

        for doc in documents:
            texts.append(
                f"{doc['title']}: {doc['description']}"
            )

        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        np.save(
            "cache/movie_embeddings.npy",
            self.embeddings
        )

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load(
                "cache/movie_embeddings.npy"
            )

            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        results = []

        for i, document_embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(
                query_embedding,
                document_embedding
            )

            document = self.documents[i]

            results.append(
                (similarity_score, document)
            )

        results.sort(
            key=lambda result: result[0],
            reverse=True
        )

        return [
            {
                "score": score,
                "title": document["title"],
                "description": document["description"]
            }
            for score, document in results[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(
        self,
        documents: list[dict]
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for movie_idx, document in enumerate(self.documents):
            description = document["description"]

            if not description.strip():
                continue

            chunks = semantic_chunk(
                description,
                max_chunk_size=4,
                overlap=1
            )

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)

                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks
                    }
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True
        )

        self.chunk_metadata = chunk_metadata

        np.save(
            "cache/chunk_embeddings.npy",
            self.chunk_embeddings
        )

        with open(
            "cache/chunk_metadata.json",
            "w"
        ) as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(
        self,
        documents: list[dict]
    ) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if (
            os.path.exists(embeddings_path)
            and os.path.exists(metadata_path)
        ):
            self.chunk_embeddings = np.load(
                embeddings_path
            )

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.chunk_metadata = metadata["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(
        self,
        query: str,
        limit: int = 10
    ) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        chunk_scores = []

        for chunk_index, chunk_embedding in enumerate(
            self.chunk_embeddings
        ):
            metadata = self.chunk_metadata[chunk_index]

            score = cosine_similarity(
                query_embedding,
                chunk_embedding
            )

            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": score
                }
            )

        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]

            if (
                movie_idx not in movie_scores
                or score > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = score

        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for movie_idx, score in sorted_movies[:limit]:
            document = self.documents[movie_idx]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"][:100],
                    "score": round(score, 4),
                    "metadata": {}
                }
            )

        return results


def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(
        f"Max sequence length: "
        f"{semantic_search.model.max_seq_length}"
    )


def embed_text(text):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()

    with open("data/movies.json", "r") as file:
        data = json.load(file)

    documents = data["movies"]

    embeddings = semantic_search.load_or_create_embeddings(
        documents
    )

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: "
        f"{embeddings.shape[0]} vectors in "
        f"{embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")