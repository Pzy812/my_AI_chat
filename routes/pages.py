"""页面与静态导出下载。"""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config.app_config import BASE_DIR, EXPORTS_DIR

router = APIRouter(tags=["pages"])

_INDEX_HTML = BASE_DIR / "template" / "1.html"


@router.get("/")
async def index():
    if not _INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index not found")
    return FileResponse(_INDEX_HTML, media_type="text/html; charset=utf-8")


@router.get("/exports/{filename}")
async def export_file_download(filename: str):
    """下载 MCP export_to_excel 写入 exports/ 目录下的文件（仅允许单层文件名）。"""
    if not filename or "/" in filename or "\\" in filename or filename.strip() in (".", ".."):
        raise HTTPException(status_code=404, detail="not found")
    safe = Path(filename).name
    if safe != filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    if not safe.lower().endswith(".xlsx"):
        raise HTTPException(status_code=404, detail="not found")
    fp = (EXPORTS_DIR / safe).resolve()
    try:
        fp.relative_to(EXPORTS_DIR)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found") from None
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        path=str(fp),
        filename=safe,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
