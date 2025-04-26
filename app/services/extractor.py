import json
import logging
import os

from mistralai import DocumentURLChunk, Mistral

from app.core.config import settings

try:

    _API_KEY = getattr(settings, "MISTRAL_API_KEY", None)
except (ImportError, AttributeError):
    _API_KEY = None

_API_KEY = _API_KEY or os.getenv("MISTRAL_API_KEY")

if not _API_KEY:
    raise RuntimeError("Missing Mistral API key. Set MISTRAL_API_KEY or add it to .env file")  # noqa: E501

logger = logging.getLogger(__name__)


def _upload_and_ocr(client, file_path):
    """Upload *file_path* to Mistral, run OCR, return (ocr_dict, file_id)."""
    logger.debug("Uploading %s (%s bytes) to Mistral",
                 file_path.name, file_path.stat().st_size)

    # Upload file to client
    upload_resp = client.files.upload(
        file={
            "file_name": file_path.name,
            "content": file_path.read_bytes(),
        },
        purpose="ocr",
    )

    # Get URL from uploaded file
    signed = client.files.get_signed_url(file_id=upload_resp.id, expiry=1)

    # OCR process
    ocr_resp = client.ocr.process(
        document=DocumentURLChunk(document_url=signed.url),
        model="mistral-ocr-latest",
        include_image_base64=True,
    )

    # Dump response to json
    ocr_dict = json.loads(ocr_resp.model_dump_json())
    return ocr_dict, upload_resp.id


def extract_pdf(file_path, cleanup_remote=True):
    """Run OCR on a PDF located at *file_path* and return the JSON result.

    Parameters
    ----------
    file_path
        Local path to the PDF document.
    cleanup_remote
        If *True* (default) the temporary file stored on Mistral's side is
        deleted after OCR completes.
    """
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    with Mistral(api_key=_API_KEY) as client:
        ocr_dict, file_id = _upload_and_ocr(client, file_path)

        if cleanup_remote:
            try:
                client.files.delete(file_id=file_id)
            except Exception as exc:
                logger.warning(
                    "Failed to delete remote file %s – %r", file_id, exc
                )

    return ocr_dict
