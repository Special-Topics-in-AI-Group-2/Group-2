
from src.data_utils import (
    build_corpus,
    build_query_stream
)


def test_corpus_shape():

    chunks, queries = build_corpus(
        n_papers=80,
        chunks_per_paper=5,
        seed=42
    )

    assert len(chunks) == 400
    assert len(queries) == 40


def test_relevant_chunk_ids_exist():

    chunks, queries = build_corpus(seed=42)

    valid_chunk_ids = {
        chunk.chunk_id
        for chunk in chunks
    }

    for query in queries:

        for chunk_id in query.relevant_chunk_ids:

            assert chunk_id in valid_chunk_ids


def test_query_stream_size():

    _, queries = build_corpus(seed=42)

    stream = build_query_stream(
        queries,
        n_stream=400,
        drift_at=200,
        seed=42
    )

    assert len(stream) == 400
