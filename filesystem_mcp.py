"""本地文件系统 MCP 工具：列目录、读文件、按模式查找。"""
from __future__ import annotations

import fnmatch
import os
from datetime import datetime
from pathlib import Path

from document_parse_local import DOC_FILE_TYPES, parse_document_sync

# 可直接按文本读取的扩展名
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".py",
    ".pyw",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".sql",
    ".csv",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".env",
    ".bat",
    ".ps1",
    ".sh",
    ".vue",
    ".rb",
    ".php",
    ".swift",
    ".kt",
}

MAX_LIST_ITEMS = int(os.getenv("MCP_FS_MAX_LIST_ITEMS", "300"))
MAX_GLOB_ITEMS = int(os.getenv("MCP_FS_MAX_GLOB_ITEMS", "200"))
MAX_READ_CHARS = int(os.getenv("MCP_FS_MAX_READ_CHARS", "80000"))
MAX_READ_BYTES = int(os.getenv("MCP_FS_MAX_READ_BYTES", str(5 * 1024 * 1024)))


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def is_path_restricted() -> bool:
    """默认不限制路径；设置 MCP_FS_RESTRICT_PATHS=1 或显式配置 MCP_FS_ALLOWED_ROOTS 时启用白名单。"""
    if _env_truthy("MCP_FS_RESTRICT_PATHS"):
        return True
    return bool(os.getenv("MCP_FS_ALLOWED_ROOTS", "").strip())


def _default_allowed_roots() -> list[Path]:
    roots: list[Path] = []
    agent_root = Path(os.getenv("MCP_FS_DEFAULT_ROOT", "E:/agent"))
    if agent_root.exists():
        roots.append(agent_root.resolve())
    project_root = Path(__file__).resolve().parent
    roots.append(project_root)
    home = Path.home()
    if home.exists():
        roots.append(home.resolve())
    # 去重并保持顺序
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def allowed_roots() -> list[Path]:
    if not is_path_restricted():
        return []
    raw = os.getenv("MCP_FS_ALLOWED_ROOTS", "").strip()
    if not raw:
        return _default_allowed_roots()
    roots: list[Path] = []
    for part in raw.replace(",", ";").split(";"):
        p = part.strip()
        if not p:
            continue
        path = Path(p).expanduser()
        if path.exists():
            roots.append(path.resolve())
    return roots or _default_allowed_roots()


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_path(path_str: str, *, must_exist: bool = True) -> Path:
    if not (path_str or "").strip():
        raise ValueError("路径不能为空")
    path = Path(path_str.strip()).expanduser()
    path = path.resolve()
    if is_path_restricted():
        roots = allowed_roots()
        if roots and not any(_is_under_root(path, root) for root in roots):
            allowed = "\n".join(f"  - {r}" for r in roots)
            raise PermissionError(
                f"路径不在允许访问范围内：{path}\n允许的根目录：\n{allowed}\n"
                "当前已启用路径白名单（MCP_FS_RESTRICT_PATHS 或 MCP_FS_ALLOWED_ROOTS）。"
                "若要访问任意路径，请关闭 MCP_FS_RESTRICT_PATHS 并清空 MCP_FS_ALLOWED_ROOTS。"
            )
    if must_exist and not path.exists():
        raise FileNotFoundError(f"路径不存在：{path}")
    return path


def _entry_info(path: Path) -> dict:
    stat = path.stat()
    is_dir = path.is_dir()
    return {
        "name": path.name,
        "path": str(path.resolve()),
        "type": "directory" if is_dir else "file",
        "size": stat.st_size if not is_dir else None,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _list_directory(dir_path: Path, *, recursive: bool, max_items: int) -> dict:
    items: list[dict] = []
    truncated = False

    if recursive:
        for root, dirnames, filenames in os.walk(dir_path):
            root_path = Path(root)
            # 跳过隐藏目录
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in sorted(dirnames):
                items.append(_entry_info(root_path / name))
                if len(items) >= max_items:
                    truncated = True
                    break
            if truncated:
                break
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                items.append(_entry_info(root_path / name))
                if len(items) >= max_items:
                    truncated = True
                    break
            if truncated:
                break
    else:
        for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            items.append(_entry_info(child))
            if len(items) >= max_items:
                truncated = True
                break

    return {
        "directory": str(dir_path),
        "count": len(items),
        "truncated": truncated,
        "items": items,
    }


def _read_text_file(path: Path, max_chars: int) -> dict:
    data = path.read_bytes()
    if len(data) > MAX_READ_BYTES:
        raise ValueError(
            f"文件过大（{len(data)} 字节），上限 {MAX_READ_BYTES}。"
            "请缩小文件或调大 MCP_FS_MAX_READ_BYTES。"
        )
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
        try:
            text = data.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法解码为文本：{path}（可能是二进制文件）")

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n…(内容已截断)"
    return {
        "path": str(path),
        "kind": "text",
        "encoding": encoding,
        "size": len(data),
        "truncated": truncated,
        "content": text,
    }


def _read_document_file(path: Path, max_chars: int) -> dict:
    file_type = DOC_FILE_TYPES[path.suffix.lower()]
    text = parse_document_sync(path, file_type)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n…(内容已截断)"
    return {
        "path": str(path),
        "kind": "document",
        "extension": path.suffix.lower(),
        "truncated": truncated,
        "content": text,
    }


def register_filesystem_tools(mcp) -> None:
    """向 FastMCP 实例注册本地文件系统工具。"""

    @mcp.tool(
        name="list_local_directory",
        description=(
            "列出本机目录下的文件和子文件夹（返回绝对路径）。"
            "用户询问某文件夹里有什么、要把文件夹内文件发微信/邮件前，应先用此工具获取真实路径。"
        ),
    )
    def list_local_directory(
        directory: str,
        recursive: bool = False,
        max_items: int = MAX_LIST_ITEMS,
    ):
        try:
            dir_path = resolve_allowed_path(directory, must_exist=True)
            if not dir_path.is_dir():
                return f"❌ 不是目录：{dir_path}"
            cap = max(1, min(int(max_items or MAX_LIST_ITEMS), MAX_LIST_ITEMS))
            result = _list_directory(dir_path, recursive=bool(recursive), max_items=cap)
            return result
        except Exception as e:
            return f"❌ 列目录失败：{e}"

    @mcp.tool(
        name="glob_local_files",
        description=(
            "在指定目录下按文件名模式查找文件（如 *、*.py、*.pdf），返回绝对路径列表。"
            "用户说「把某文件夹里所有文件发给某人」时，先用此工具列出路径，再调用 send_wechat_files。"
        ),
    )
    def glob_local_files(
        directory: str,
        pattern: str = "*",
        recursive: bool = False,
        max_items: int = MAX_GLOB_ITEMS,
    ):
        try:
            dir_path = resolve_allowed_path(directory, must_exist=True)
            if not dir_path.is_dir():
                return f"❌ 不是目录：{dir_path}"
            cap = max(1, min(int(max_items or MAX_GLOB_ITEMS), MAX_GLOB_ITEMS))
            pat = (pattern or "*").strip()
            matches: list[dict] = []

            if recursive:
                for root, _, filenames in os.walk(dir_path):
                    for name in filenames:
                        if name.startswith("."):
                            continue
                        if fnmatch.fnmatch(name, pat):
                            fp = Path(root) / name
                            matches.append(_entry_info(fp))
                            if len(matches) >= cap:
                                break
                    if len(matches) >= cap:
                        break
            else:
                for child in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
                    if not child.is_file() or child.name.startswith("."):
                        continue
                    if fnmatch.fnmatch(child.name, pat):
                        matches.append(_entry_info(child))
                        if len(matches) >= cap:
                            break

            return {
                "directory": str(dir_path),
                "pattern": pat,
                "recursive": bool(recursive),
                "count": len(matches),
                "truncated": len(matches) >= cap,
                "files": matches,
            }
        except Exception as e:
            return f"❌ 查找文件失败：{e}"

    @mcp.tool(
        name="read_local_file",
        description=(
            "读取本机文件内容。支持 txt/md/py/json 等文本，以及 pdf/docx/xlsx/pptx 等办公文档（自动提取正文）。"
            "file_path 必须是绝对路径；路径须先通过 list_local_directory 或 glob_local_files 确认存在。"
        ),
    )
    def read_local_file(file_path: str, max_chars: int = MAX_READ_CHARS):
        try:
            path = resolve_allowed_path(file_path, must_exist=True)
            if path.is_dir():
                return f"❌ 这是目录不是文件，请用 list_local_directory：{path}"
            cap = max(1000, min(int(max_chars or MAX_READ_CHARS), MAX_READ_CHARS))
            ext = path.suffix.lower()
            if ext in DOC_FILE_TYPES:
                return _read_document_file(path, cap)
            if ext in TEXT_EXTENSIONS or ext == "":
                return _read_text_file(path, cap)
            # 未知扩展名：尝试按文本读，失败则提示
            try:
                return _read_text_file(path, cap)
            except ValueError:
                return (
                    f"❌ 不支持直接读取该类型：{ext or '(无扩展名)'}。"
                    f" 可用 list_local_directory 查看文件；"
                    f"办公文档支持 {', '.join(sorted(DOC_FILE_TYPES))}；"
                    f"或通过聊天页面上传后解析。"
                )
        except Exception as e:
            return f"❌ 读取文件失败：{e}"

    @mcp.tool(
        name="get_filesystem_roots",
        description="查看当前 MCP 文件访问策略（是否限制根目录）。",
    )
    def get_filesystem_roots():
        if is_path_restricted():
            roots = allowed_roots()
            return {
                "restricted": True,
                "roots": [str(r) for r in roots],
                "hint": "当前为白名单模式；要访问任意路径请关闭 MCP_FS_RESTRICT_PATHS",
            }
        return {
            "restricted": False,
            "roots": [],
            "hint": "当前可访问本机任意存在的路径（未启用白名单）",
        }
