"""
通过 AWS Document Loader MCP（stdio）解析 Office / PDF 等文档。
需安装：pip install awslabs-document-loader-mcp-server
或系统有 uvx：uvx awslabs.document-loader-mcp-server@latest
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DOC_FILE_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xls": "xls",
    ".pptx": "pptx",
    ".ppt": "ppt",
}


def _tool_result_text(result) -> str:
    parts: list[str] = []
    for block in result.content or []:
        if hasattr(block, "text") and block.text:
            parts.append(block.text)
        elif isinstance(block, dict):
            if block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
    if result.isError and not parts:
        return "文档解析失败（MCP 返回错误）"
    return "\n".join(parts).strip() or "（未提取到文本内容）"


def _stdio_server_params(document_base_dir: Path) -> StdioServerParameters:
    base = document_base_dir.resolve()
    env = {
        **os.environ,
        "DOCUMENT_BASE_DIR": str(base),
        "FASTMCP_LOG_LEVEL": os.getenv("FASTMCP_LOG_LEVEL", "ERROR"),
        "MAX_FILE_SIZE_MB": os.getenv("MAX_FILE_SIZE_MB", "50"),
    }
    custom_cmd = os.getenv("DOCUMENT_LOADER_MCP_CMD", "").strip()
    if custom_cmd:
        custom_args = os.getenv("DOCUMENT_LOADER_MCP_ARGS", "").strip()
        return StdioServerParameters(
            command=custom_cmd,
            args=custom_args.split() if custom_args else [],
            env=env,
        )
    script = shutil.which("awslabs.document-loader-mcp-server")
    if script:
        return StdioServerParameters(command=script, args=[], env=env)
    if shutil.which("uvx"):
        return StdioServerParameters(
            command="uvx",
            args=["awslabs.document-loader-mcp-server@latest"],
            env=env,
        )
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", "from awslabs.document_loader_mcp_server.server import main; main()"],
        env=env,
    )


async def read_document_aws_mcp(file_path: Path, document_base_dir: Path) -> str:
    """
    file_path 必须在 document_base_dir 之下。
    向 MCP 传相对路径（相对 DOCUMENT_BASE_DIR）。
    """
    base = document_base_dir.resolve()
    fp = file_path.resolve()
    try:
        rel = fp.relative_to(base)
    except ValueError as e:
        raise ValueError("文件不在允许的上传目录内") from e

    ext = fp.suffix.lower()
    file_type = DOC_FILE_TYPES.get(ext)
    if not file_type:
        raise ValueError(f"不支持的文档类型: {ext}")

    params = _stdio_server_params(base)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "read_document",
                arguments={
                    "file_path": rel.as_posix(),
                    "file_type": file_type,
                },
            )
    return _tool_result_text(result)
