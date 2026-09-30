from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        self.texts = [
            f"{doc['title']}: {doc['description']}"
            for doc in documents
        ]

        self.text_embeddings = self.model.encode(
            self.texts,
            show_progress_bar=True
        )

    def embed_image(self, image_path):
        image = Image.open(image_path)

        embedding = self.model.encode([image])

        return embedding[0]

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)

        results = []

        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(
                image_embedding,
                text_embedding
            )

            document = self.documents[i]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "description": document["description"],
                    "score": similarity
                }
            )

        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return results[:5]


def load_movies():
    import json

    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch([])

    embedding = multimodal_search.embed_image(image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    documents = load_movies()

    multimodal_search = MultimodalSearch(documents)

    results = multimodal_search.search_with_image(image_path)

    return results