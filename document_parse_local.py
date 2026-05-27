"""进程内解析 PDF / Office，避免 Windows 上 MCP stdio 子进程 TaskGroup 失败。"""
from __future__ import annotations

import asyncio
from pathlib import Path

DOC_FILE_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
    ".ppt": "ppt",
}


def _extract_pdf_text(file_path: Path) -> str:
    import pdfplumber

    text_content = ""
    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text_content += f"\n--- Page {page_num} ---\n"
            page_text = page.extract_text()
            if page_text:
                text_content += page_text
    return text_content.strip()


def _convert_with_markitdown(file_path: Path) -> str:
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(file_path))
    return (result.text_content or "").strip()


def parse_document_sync(file_path: Path, file_type: str) -> str:
    ft = file_type.lower()
    if ft == "pdf":
        text = _extract_pdf_text(file_path)
    elif ft in {"docx", "doc", "xlsx", "xls", "pptx", "ppt"}:
        text = _convert_with_markitdown(file_path)
    else:
        raise ValueError(f"不支持的文档类型: {file_type}")
    return text or "（未提取到文本内容）"


async def read_document_local(file_path: Path, file_type: str) -> str:
    return await asyncio.to_thread(parse_document_sync, file_path, file_type)
