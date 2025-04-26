import tempfile
from pathlib import Path
from typing import List

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.schemas.document import DocumentsResponse
from app.services import chunker, embedder, extractor

documents_bp = Blueprint("documents", __name__, url_prefix="/documents")

ALLOWED_MIMETYPE = "application/pdf"


@documents_bp.route("", methods=["POST"])
def upload_documents():
    if "files" not in request.files:
        return jsonify({"detail": "multipart field 'files' missing"}), 400
    files: List = request.files.getlist("files")
    if not files:
        return jsonify({"detail": "no files sent"}), 400

    docs_indexed = 0
    total_chunks = 0
    uploads_dir = Path(current_app.config.get("UPLOAD_DIR", "data/uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for uf in files:
            if uf.mimetype != ALLOWED_MIMETYPE:
                return jsonify({"detail": f"{uf.filename} is not a PDF"}), 415

            safe_name = secure_filename(uf.filename)

            pdf_path = uploads_dir / safe_name
            uf.save(pdf_path)

            # -- OCR ➜ markdown -------------------------------------------------
            ocr_json = extractor.extract_pdf(pdf_path)
            # -- markdown ➜ overlapping chunks -------------------------------
            chunks = chunker.chunk_markdown_pages(ocr_json, uf.filename)
            chunk_file = pdf_path.with_suffix(".chunks.json")
            chunker.save_chunks(chunks, chunk_file)
            total_chunks += len(chunks)
            # -- chunks ➜ Chroma vectors --------------------------------------
            embedder.embed_json_file(chunk_file)

            docs_indexed += 1

    body = DocumentsResponse(
        message="Documents processed successfully",
        documents_indexed=docs_indexed,
        total_chunks=total_chunks,
    ).model_dump()

    return jsonify(body), 201
