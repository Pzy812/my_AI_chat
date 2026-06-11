"""页面与静态导出下载。"""
from pathlib import Path

from flask import Blueprint, abort, render_template, send_from_directory

from config.app_config import EXPORTS_DIR

bp = Blueprint("pages", __name__)


@bp.route("/")
def index():
    return render_template("1.html")


@bp.route("/exports/<filename>")
def export_file_download(filename: str):
    """下载 MCP export_to_excel 写入 exports/ 目录下的文件（仅允许单层文件名）。"""
    if not filename or "/" in filename or "\\" in filename or filename.strip() in (".", ".."):
        abort(404)
    safe = Path(filename).name
    if safe != filename:
        abort(400)
    if not safe.lower().endswith(".xlsx"):
        abort(404)
    fp = (EXPORTS_DIR / safe).resolve()
    try:
        fp.relative_to(EXPORTS_DIR)
    except ValueError:
        abort(404)
    if not fp.is_file():
        abort(404)
    return send_from_directory(str(EXPORTS_DIR), safe, as_attachment=True, download_name=safe)
