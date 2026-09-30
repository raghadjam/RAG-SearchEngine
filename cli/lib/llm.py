import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def fix_spelling(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()

def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    additional_terms = response.choices[0].message.content.strip()

    return f"{query} {additional_terms}"

import time

MAX_SCORE_ATTEMPTS = 5
SLEEP_SECONDS = 3


def score_document(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    last_error = None

    for attempt in range(MAX_SCORE_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )

            text = response.choices[0].message.content.strip()

            return float(text)
        except (ValueError, Exception) as error:
            last_error = error

    raise RuntimeError(
        f"Failed to score document after {MAX_SCORE_ATTEMPTS} attempts"
    ) from last_error


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    reranked = []

    for i, result in enumerate(results):
        score = score_document(query, result)

        reranked.append(
            {
                **result,
                "rerank_score": score,
            }
        )

        if i < len(results) - 1:
            time.sleep(SLEEP_SECONDS)

    reranked.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return reranked

import json

MAX_BATCH_ATTEMPTS = 5


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    doc_list_str = "\n".join(
        f"{result['id']}: {result['title']} - {result['document']}"
        for result in results
    )

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    last_error = None

    for _ in range(MAX_BATCH_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )

            text = response.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[len("json"):].strip()

            ranked_ids = json.loads(text)

            by_id = {result["id"]: result for result in results}

            ranked_ids = [
                doc_id for doc_id in ranked_ids if doc_id in by_id
            ]

            reranked = []

            for rank, doc_id in enumerate(ranked_ids, start=1):
                reranked.append(
                    {
                        **by_id[doc_id],
                        "rerank_rank": rank,
                    }
                )

            reranked.sort(key=lambda result: result["rerank_rank"])

            return reranked
        except (ValueError, KeyError, Exception) as error:
            last_error = error

    raise RuntimeError(
        f"Failed to batch rerank after {MAX_BATCH_ATTEMPTS} attempts"
    ) from last_error

def evaluate_results(query: str, results: list[dict]) -> list[int]:
    formatted_results = [
        f"{i}. {result.get('title', '')} - {result.get('document', '')}"
        for i, result in enumerate(results, start=1)
    ]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    last_error = None

    for _ in range(MAX_BATCH_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                messages=messages,
            )

            text = response.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[len("json"):].strip()

            scores = json.loads(text)

            if len(scores) != len(results):
                raise ValueError(
                    "Number of scores does not match number of results"
                )

            return [int(score) for score in scores]
        except (ValueError, Exception) as error:
            last_error = error

    raise RuntimeError(
        f"Failed to evaluate results after {MAX_BATCH_ATTEMPTS} attempts"
    ) from last_error

def generate_answer(query: str, results: list[dict]) -> str:
    docs = "\n".join(
        f"- {result.get('title', '')}: {result.get('document', '')}"
        for result in results
    )

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()

def summarize_results(query: str, results: list[dict]) -> str:
    formatted_results = "\n".join(
        f"- {result.get('title', '')}: {result.get('document', '')}"
        for result in results
    )

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{formatted_results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()
def answer_with_citations(query: str, results: list[dict]) -> str:
    documents = "\n".join(
        f"[{i}] {result.get('title', '')}: {result.get('document', '')}"
        for i, result in enumerate(results, start=1)
    )

    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()

def answer_question(question: str, results: list[dict]) -> str:
    context = "\n".join(
        f"- {result.get('title', '')}: {result.get('document', '')}"
        for result in results
    )

    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    return response.choices[0].message.content.strip()