import os
import uuid
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/vnd.pdf",
}
ALLOWED_EXTENSIONS = {".pdf"}


def validate_pdf_upload(file: UploadFile, max_size_mb: int) -> None:
    """Validate uploaded file is a valid PDF within size limits."""
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{extension}'. Only PDF files are accepted.",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES and content_type != "application/octet-stream":
        # Be lenient — some clients send generic content types
        logger.warning("Unexpected content-type '%s' for PDF upload", content_type)


def check_file_size(file_bytes: bytes, max_size_mb: int) -> None:
    """Raise HTTPException if file exceeds max size."""
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_size_mb} MB.",
        )


async def save_upload_to_disk(file: UploadFile, upload_dir: str) -> tuple[str, bytes]:
    """Save uploaded file to a temp location and return the path."""
    os.makedirs(upload_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{Path(file.filename or 'upload').name}"
    dest_path = os.path.join(upload_dir, unique_name)

    contents = await file.read()
    check_file_size(contents, max_size_mb=50)

    with open(dest_path, "wb") as f:
        f.write(contents)

    logger.info("Saved upload to %s (%d bytes)", dest_path, len(contents))
    return dest_path, contents


def cleanup_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        logger.warning("Could not delete temp file: %s", path)
