import json
from pathlib import Path
from typing import Any, Dict

from langchain_text_splitters import MarkdownTextSplitter


def load_ocr_json(path: Path) -> Dict[str, Any]:
    """Load the JSON produced by extractor.py (e.g.: Mistral OCR)."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def chunk_markdown_pages(
    data, filename, chunk_size=1000, chunk_overlap=200
):
    """Iterate over each markdown field inside pages and create chunks.

    Parameters
    ----------
    data
        Dict shaped exactly like the example provided by *extractor.py*.
    filename
        Name of the original pdf file.
    chunk_size
        Maximum characters for each chunk.
    chunk_overlap
        Desired character overlap between consecutive chunks.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains: *page_index*, *chunk_index*, and *text*.
    """
    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    for page in data.get("pages", []):
        markdown_text: str = page.get("markdown", "")
        if not markdown_text:
            continue  # skip empty pages

        page_chunks = splitter.split_text(markdown_text)
        page_index = page.get("index", None)
        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append(
                {
                    "page_index": page_index+1,
                    "chunk_index": idx,
                    "text": chunk_text.strip(),
                    "filename": filename
                }
            )
    return all_chunks


def save_chunks(chunks, output_path) -> None:
    """Persist *chunks* as JSON so that *embedder.py* can ingest them later."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(chunks, fp, ensure_ascii=False, indent=2)
