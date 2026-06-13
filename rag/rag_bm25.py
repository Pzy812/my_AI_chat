"""BM25 稀疏检索：与 Milvus 向量检索配合做 Hybrid RRF。"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from config.app_config import UPLOADS_DIR
from rag.milvus_naming import norm_session_id

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|\w+", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    body = (text or "").strip().lower()
    if not body:
        return []
    tokens = _TOKEN_RE.findall(body)
    return tokens or [body[:64]]


def _bm25_dir(session_id: str) -> Path:
    sid = norm_session_id(session_id)
    path = UPLOADS_DIR / sid / ".rag_bm25"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bm25_path(session_id: str, file_id: str) -> Path:
    fid = (file_id or "").strip()
    return _bm25_dir(session_id) / f"{fid}.pkl"


def bm25_index_exists(session_id: str, file_id: str) -> bool:
    fid = (file_id or "").strip()
    if not fid:
        return False
    return _bm25_path(session_id, fid).is_file()


def save_bm25_index(
    session_id: str,
    file_id: str,
    records: list[dict[str, Any]],
) -> None:
    """records: split_text_parent_child 输出，按 child_text 建 BM25。"""
    fid = (file_id or "").strip()
    if not fid or not records:
        return
    child_texts = [str(r.get("child_text") or "") for r in records]
    tokenized = [tokenize_for_bm25(t) for t in child_texts]
    if not any(tokenized):
        return
    payload = {
        "records": [
            {
                "child_text": r.get("child_text") or "",
                "parent_text": r.get("parent_text") or "",
                "parent_index": int(r.get("parent_index", 0)),
                "child_index": int(r.get("child_index", 0)),
            }
            for r in records
        ],
        "tokenized": tokenized,
    }
    path = _bm25_path(session_id, fid)
    path.write_bytes(pickle.dumps(payload))


def delete_bm25_index(session_id: str, file_id: str) -> None:
    fid = (file_id or "").strip()
    if not fid:
        return
    path = _bm25_path(session_id, fid)
    path.unlink(missing_ok=True)
    try:
        parent = _bm25_dir(session_id)
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def delete_session_bm25_indexes(session_id: str) -> None:
    path = _bm25_dir(session_id)
    if not path.exists():
        return
    for item in path.glob("*.pkl"):
        item.unlink(missing_ok=True)
    try:
        path.rmdir()
    except OSError:
        pass


def _load_bm25(session_id: str, file_id: str) -> tuple[BM25Okapi | None, list[dict[str, Any]]]:
    fid = (file_id or "").strip()
    if not fid:
        return None, []
    path = _bm25_path(session_id, fid)
    if not path.is_file():
        return None, []
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None, []
    records = payload.get("records") or []
    tokenized = payload.get("tokenized") or []
    if not records or not tokenized:
        return None, []
    return BM25Okapi(tokenized), records


def search_bm25(
    session_id: str,
    file_id: str,
    query: str,
    *,
    limit: int,
    source_name: str = "",
    collection: str = "",
) -> list[dict[str, Any]]:
    bm25, records = _load_bm25(session_id, file_id)
    if bm25 is None or not records:
        return []
    q_tokens = tokenize_for_bm25(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(
        enumerate(scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )[: max(limit, 1)]
    hits: list[dict[str, Any]] = []
    for idx, score in ranked:
        if score <= 0:
            continue
        rec = records[idx]
        hits.append(
            {
                "text": rec.get("child_text") or "",
                "parent_text": rec.get("parent_text") or "",
                "parent_index": rec.get("parent_index"),
                "child_index": rec.get("child_index"),
                "source": source_name,
                "file_id": file_id,
                "collection": collection,
                "score": float(score),
                "retriever": "bm25",
            }
        )
    return hits
