import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = "cache"

stemmer = PorterStemmer()


def preprocess(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()


def load_stop_words():
    with open("data/stopwords.txt") as f:
        stop_words = f.read().splitlines()

    translator = str.maketrans("", "", string.punctuation)

    return [
        word.lower().translate(translator)
        for word in stop_words
    ]


def tokenize_text(text):
    tokens = preprocess(text)
    stop_words = load_stop_words()

    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    tokens = [
        stemmer.stem(token)
        for token in tokens
    ]

    return tokens


def tokenize_term(term):
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise Exception("Term must contain exactly one token")

    return tokens[0]


def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_DIR, "term_frequencies.pkl"
        )
        self.doc_lengths_path = os.path.join(
            CACHE_DIR, "doc_lengths.pkl"
        )

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        if term not in self.index:
            return []

        return sorted(self.index[term])

    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequencies:
            return 0

        return self.term_frequencies[doc_id].get(term, 0)

    def get_bm25_idf(self, term):
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))

        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def __get_avg_doc_length(self):
        if len(self.doc_lengths) == 0:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        if tf == 0:
            return 0.0

        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        query_tokens = tokenize_text(query)

        scores = {}

        for doc_id in self.docmap:
            score = 0.0

            for token in query_tokens:
                score += self.bm25(doc_id, token)

            scores[doc_id] = score

        sorted_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_scores[:limit]

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)