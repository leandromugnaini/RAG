import json
import logging
import os
import uuid
from typing import Any, Dict, List

import chromadb
from langchain_openai import OpenAIEmbeddings

try:
    import tiktoken

    _ENCODER = tiktoken.encoding_for_model("text-embedding-3-small")
except ModuleNotFoundError:
    _ENCODER = None

try:
    from app.core.config import settings

    _OPENAI_KEY = getattr(settings, "OPENAI_API_KEY", None)
except (ImportError, AttributeError):
    _OPENAI_KEY = None

_OPENAI_KEY = _OPENAI_KEY or os.getenv("OPENAI_API_KEY")

if not _OPENAI_KEY:
    raise RuntimeError("Missing OpenAI API key – set OPENAI_API_KEY env var or app.core.config.settings")  # noqa: E501

logger = logging.getLogger(__name__)


def embed_json_file(
    chunk_json_path,
    persist_dir="data/chroma_db",
    collection_name="documents",
    batch_size=64,
):
    """Embed and store all chunks present in *chunk_json_path*.

    Parameters
    ----------
    chunk_json_path
        Path to the JSON file produced by `chunker.py` (list of dicts).
    persist_dir
        Directory where Chroma will persist vectors & metadata.
    collection_name
        Name of (or alias to) the collection inside Chroma.
    batch_size
        Number of chunks to embed per API call.

    Returns
    -------
    dict with counters: `n_chunks`, `n_vectors`, `collection_name`.
    """
    logger.info("Loading chunk list from %s", chunk_json_path)
    chunks: List[Dict[str, Any]] = json.loads(chunk_json_path.read_text())
    if not chunks:
        logger.warning("No chunks found – nothing to embed")
        return {"n_chunks": 0, "n_vectors": 0,
                "collection_name": collection_name}

    # Instantiate embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=_OPENAI_KEY)

    # Prepare Chroma client + collection
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Use external embeddings (we compute first, then
    # add with `embeddings=` param)
    collection = client.get_or_create_collection(name=collection_name)

    # Process in batches
    texts = []
    metadatas = []
    ids = []
    total_vectors = 0

    for chunk in chunks:
        text = chunk["text"]
        metadata = {
            "page_index": chunk.get("page_index"),
            "chunk_index": chunk.get("chunk_index"),
            "filename": chunk.get("filename")
        }
        if _ENCODER is not None:
            metadata["tokens"] = len(_ENCODER.encode(text))

        texts.append(text)
        metadatas.append(metadata)
        ids.append(str(uuid.uuid4()))

        # If batch is full, embed & upsert
        if len(texts) >= batch_size:
            _upsert_batch(collection, texts, metadatas, ids, embeddings)
            total_vectors += len(texts)
            texts, metadatas, ids = [], [], []

    # final tail
    if texts:
        _upsert_batch(collection, texts, metadatas, ids, embeddings)
        total_vectors += len(texts)

    logger.info("Stored %s vectors in collection '%s' (persist dir = %s)",
                total_vectors, collection_name, persist_dir)
    return {"n_chunks": len(chunks), "n_vectors": total_vectors,
            "collection_name": collection_name}


def _upsert_batch(
    collection,
    texts,
    metadatas,
    ids,
    embeddings,
):
    """Embed *texts* into vector space and upsert into *collection*."""
    logger.debug("Embedding batch of %s chunks", len(texts))
    vectors = embeddings.embed_documents(texts)
    collection.add(documents=texts, embeddings=vectors,
                   metadatas=metadatas, ids=ids)
