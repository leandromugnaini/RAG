import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
import openai
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings

try:
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover
    tiktoken = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
try:
    _OPENAI_KEY: Optional[str] = getattr(settings, "OPENAI_API_KEY", None)
except (ImportError, AttributeError):
    _OPENAI_KEY = None

_OPENAI_KEY = _OPENAI_KEY or os.getenv("OPENAI_API_KEY")
if not _OPENAI_KEY:
    raise RuntimeError("Missing OpenAI API key – set OPENAI_API_KEY or populate app.core.config.settings")  # noqa: E501

openai.api_key = _OPENAI_KEY


try:
    _ROUTER_API_KEY: Optional[str] = getattr(settings, "ROUTER_API_KEY", None)
except (ImportError, AttributeError):
    _ROUTER_API_KEY = None

_ROUTER_API_KEY = _ROUTER_API_KEY or os.getenv("ROUTER_API_KEY")
if not _ROUTER_API_KEY:
    raise RuntimeError("Missing Router API key (Requesty) – set ROUTER_API_KEY or populate app.core.config.settings")  # noqa: E501

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS_CONTEXT = 10000  # plenty of room for prompt and answer


def _get_collection(persist_dir: Path | str, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        return client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in Chroma at "
            f"'{persist_dir}'. Have you run the embed step?"
        ) from exc


def _similar_chunks(
    question,
    collection,
    embeddings_model,
    top_k=4,
):
    """Return *top_k* most similar chunks from Chroma (with score)."""
    q_vector = embeddings_model.embed_query(question)

    res = collection.query(
        query_embeddings=[q_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    mets = res["metadatas"][0]
    dists = res["distances"][0]

    out = []

    for text, meta, dist in zip(docs, mets, dists):
        out.append({
            "text": text,
            "filename": meta.get("filename"),
            "page_index": meta.get("page_index"),
            "chunk_index": meta.get("chunk_index"),
            "score": dist
        })
    return out


def _encode_len(text, model="gpt-4o-mini"):
    if tiktoken is None:
        return len(text.split())
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def _build_context(chunks, max_tokens_context):
    """Construct OpenAI chat messages with context trimmed to *max_tokens_context*."""  # noqa: E501
    # Merge chunk texts separated by two newlines, order by
    # proximity (ascending score)
    # Lower distance => higher similarity, so sort ascending.
    chunks_sorted = sorted(chunks, key=lambda c: c["score"])
    context_texts = []
    total_tokens = 0
    for ch in chunks_sorted:
        token_len = _encode_len(ch["text"])
        if total_tokens + token_len > max_tokens_context:
            break
        context_texts.append(ch["text"])
        total_tokens += token_len

    context_block = "\n---\n".join(context_texts)

    return context_block


def answer_question(
    question,
    persist_dir="data/chroma_db",
    collection_name="documents",
    top_k=4,
    max_tokens_context=_DEFAULT_MAX_TOKENS_CONTEXT
):
    """Retrieve similar chunks and ask GPT‑4o to answer.

    Returns a dict with keys: `answer`, `sources`.
    """
    collection = _get_collection(persist_dir, collection_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=_OPENAI_KEY)

    chunks = _similar_chunks(question, collection=collection,
                             embeddings_model=embeddings, top_k=top_k)
    if not chunks:
        return {
            "answer": "I couldn't find relevant information.", "sources": []
        }

    context = _build_context(chunks, max_tokens_context)

    logger.info("Querying GPT‑4o with context (tokens≈%s)",
                _encode_len(context))

    instruction = """
        You are a meticulous assistant. Use the provided CONTEXT to answer the USER question.
        If the CONTEXT is insufficient to answer confidently, say so instead of inventing information.
        Make sure the answer to the question is properly formatted for the USER, but do not change the content of the answer (do not invent new information).
    """  # noqa: E501

    input_str = f"""
    QUESTION:\n {question}
    CONTEXT:\n {context}
    """

    try:
        # Initialize OpenAI client
        client = openai.OpenAI(
            api_key=_ROUTER_API_KEY,
            base_url="https://router.requesty.ai/v1",
            default_headers={"Authorization": f"Bearer {_ROUTER_API_KEY}"}
        )
        # Example request
        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_str}
            ]
        )
        print(response)
        # Check if the response is successful
        if not response.choices:
            raise Exception("No response choices found.")
        # Print the result
        print(response.choices[0].message.content)
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }
