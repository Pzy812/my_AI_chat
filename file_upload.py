"""上传文件保存、类型判断与解析调度。"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import os

from document_parse_local import DOC_FILE_TYPES, read_document_local
from document_loader_mcp import read_document_aws_mcp
from vision_parse import IMAGE_MIME, describe_image_glm4v

IMAGE_EXT = set(IMAGE_MIME.keys())
DOC_EXT = set(DOC_FILE_TYPES.keys())
ALLOWED_EXT = IMAGE_EXT | DOC_EXT

MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024)


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip(" .")
    return base[:180] or "file"


def detect_kind(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in DOC_EXT:
        return "document"
    return "unknown"


def _use_document_mcp() -> bool:
    return os.getenv("USE_DOCUMENT_MCP", "1").strip().lower() in ("0", "true", "yes")


async def parse_uploaded_file(file_path: Path, uploads_root: Path, kind: str) -> str:
    if kind == "image":
        return describe_image_glm4v(file_path)
    if kind == "document":
        ext = file_path.suffix.lower()
        file_type = DOC_FILE_TYPES.get(ext)
        if not file_type:
            raise ValueError(f"不支持的文档类型: {ext}")
        if _use_document_mcp():
            return await read_document_aws_mcp(file_path, uploads_root)
        return await read_document_local(file_path, file_type)
    raise ValueError("不支持的文件类型")


def new_file_id() -> str:
    return uuid.uuid4().hex[:12]
