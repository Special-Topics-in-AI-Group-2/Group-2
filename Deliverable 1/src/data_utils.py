
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random


TOPIC_VOCAB = {
    f"topic_{i}": {
        "keywords": [f"keyword_{i}_{j}" for j in range(6)],
        "semantic": [f"semantic_{i}_{j}" for j in range(4)],
        "templates": [
            "{keyword_1} systems use {semantic_1} techniques for optimization.",
            "{keyword_1} and {keyword_2} improve {semantic_1} workflows.",
            "{semantic_1} methods support {keyword_1} applications.",
            "{keyword_1} architectures rely on {semantic_2} representations."
        ]
    }
    for i in range(8)
}


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    topic_id: str
    page: int
    text: str
    keywords: List[str]
    semantic_tags: List[str]


@dataclass
class Query:
    query_id: str
    topic_id: str
    query_text: str
    relevant_chunk_ids: List[str]
    query_type: str
    timestamp: int


def build_corpus(
    n_papers: int = 80,
    chunks_per_paper: int = 5,
    seed: int = 42
) -> Tuple[List[Chunk], List[Query]]:

    random.seed(seed)

    assert n_papers % len(TOPIC_VOCAB) == 0, (
        "n_papers must be divisible by number of topics"
    )

    chunks: List[Chunk] = []
    queries: List[Query] = []

    topic_names = list(TOPIC_VOCAB.keys())
    papers_per_topic = n_papers // len(topic_names)

    chunk_lookup = {}

    chunk_counter = 0
    query_counter = 0

    for topic_id in topic_names:

        vocab = TOPIC_VOCAB[topic_id]

        keywords = vocab["keywords"]
        semantic_terms = vocab["semantic"]
        templates = vocab["templates"]

        for paper_idx in range(papers_per_topic):

            paper_id = f"paper_{topic_id}_{paper_idx:02d}"

            for _ in range(chunks_per_paper):

                chunk_id = f"chunk_{chunk_counter:04d}"
                chunk_counter += 1

                selected_keywords = random.sample(keywords, k=3)
                selected_semantic = random.sample(semantic_terms, k=2)

                template = random.choice(templates)

                text = template.format(
                    keyword_1=selected_keywords[0],
                    keyword_2=selected_keywords[1],
                    keyword_3=selected_keywords[2],
                    semantic_1=selected_semantic[0],
                    semantic_2=selected_semantic[1]
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    topic_id=topic_id,
                    page=random.randint(1, 20),
                    text=text,
                    keywords=selected_keywords,
                    semantic_tags=selected_semantic
                )

                chunks.append(chunk)

                if topic_id not in chunk_lookup:
                    chunk_lookup[topic_id] = []

                chunk_lookup[topic_id].append(chunk_id)

    query_types = [
        "keyword",
        "paraphrase",
        "hybrid",
        "ambiguous",
        "semantic"
    ]

    queries_per_topic = 5

    for topic_id in topic_names:

        vocab = TOPIC_VOCAB[topic_id]

        keywords = vocab["keywords"]
        semantic_terms = vocab["semantic"]

        topic_chunk_ids = chunk_lookup[topic_id]

        for q_idx in range(queries_per_topic):

            query_id = f"query_{query_counter:03d}"
            query_counter += 1

            query_type = query_types[q_idx % len(query_types)]

            if query_type == "keyword":

                query_text = (
                    f"{random.choice(keywords)} "
                    f"{random.choice(keywords)} methods"
                )

            elif query_type == "paraphrase":

                query_text = (
                    f"{random.choice(semantic_terms)} "
                    f"for intelligent systems"
                )

            elif query_type == "hybrid":

                query_text = (
                    f"{random.choice(keywords)} and "
                    f"{random.choice(semantic_terms)} approaches"
                )

            elif query_type == "ambiguous":

                query_text = (
                    f"optimization techniques for "
                    f"{random.choice(keywords)}"
                )

            else:

                query_text = (
                    f"{random.choice(semantic_terms)} "
                    f"representation learning"
                )

            relevant_chunk_ids = random.sample(topic_chunk_ids, k=3)

            query = Query(
                query_id=query_id,
                topic_id=topic_id,
                query_text=query_text,
                relevant_chunk_ids=relevant_chunk_ids,
                query_type=query_type,
                timestamp=query_counter
            )

            queries.append(query)

    return chunks, queries


def build_query_stream(
    queries: List[Query],
    n_stream: int = 400,
    drift_at: int = 200,
    seed: int = 42
) -> List[Query]:

    random.seed(seed)

    all_topics = sorted(
        set(q.topic_id for q in queries),
        key=lambda x: int(x.split("_")[-1])
    )

    drift_topics = [f"topic_{i}" for i in range(4)]

    queries_by_topic = {
        topic_id: [q for q in queries if q.topic_id == topic_id]
        for topic_id in all_topics
    }

    stream: List[Query] = []

    for t in range(n_stream):

        if t < drift_at:
            selected_topic = random.choice(all_topics)
        else:
            selected_topic = random.choice(drift_topics)

        base_query = random.choice(queries_by_topic[selected_topic])

        streamed_query = Query(
            query_id=f"stream_{t:04d}_{base_query.query_id}",
            topic_id=base_query.topic_id,
            query_text=base_query.query_text,
            relevant_chunk_ids=base_query.relevant_chunk_ids,
            query_type=base_query.query_type,
            timestamp=t
        )

        stream.append(streamed_query)

    return stream


def get_topic_distribution(queries: List[Query]) -> Dict[str, int]:

    distribution = {}

    for query in queries:
        distribution[query.topic_id] = (
            distribution.get(query.topic_id, 0) + 1
        )

    return distribution


def validate_corpus(
    chunks: List[Chunk],
    queries: List[Query]
) -> None:

    chunk_ids = {chunk.chunk_id for chunk in chunks}

    assert len(chunk_ids) == len(chunks), "Duplicate chunk IDs found."

    query_ids = {query.query_id for query in queries}

    assert len(query_ids) == len(queries), "Duplicate query IDs found."

    for query in queries:

        assert len(query.relevant_chunk_ids) > 0, (
            f"{query.query_id} has no relevant chunks."
        )

        for chunk_id in query.relevant_chunk_ids:

            assert chunk_id in chunk_ids, (
                f"Missing chunk ID: {chunk_id}"
            )
