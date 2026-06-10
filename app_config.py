"""应用路径、环境变量与日志配置。"""
import logging
import os
from pathlib import Path

from env_config import ensure_zhipuai_api_key_in_environ

BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = (BASE_DIR / "exports").resolve()
UPLOADS_DIR = (BASE_DIR / "uploads").resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MCP_TABLE_ATTACH_MAX = 120_000

LOG_LLM_PROMPT = os.getenv("LOG_LLM_PROMPT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
LOG_LLM_PROMPT_MAX = int(os.getenv("LOG_LLM_PROMPT_MAX", "12000"))

logger = logging.getLogger("ai_chat")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

ensure_zhipuai_api_key_in_environ()
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5001"))

# PostgreSQL：聊天权威存储 + LangGraph Checkpointer（可共用同一库）
POSTGRES_URI = (
    os.getenv("POSTGRES_URI", "").strip()
    or os.getenv("DATABASE_URL", "").strip()
)
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "500"))
CHAT_AGENT_CONTEXT_MESSAGES = int(
    os.getenv(
        "CHAT_AGENT_CONTEXT_MESSAGES",
        os.getenv("REDIS_CHAT_MAX_MESSAGES", "80"),
    )
)
# 超过 N 轮用户发言时，对更早对话做小模型摘要（保留最近 CHAT_SUMMARY_KEEP_ROUNDS 轮原文）
CHAT_SUMMARY_ENABLED = os.getenv("CHAT_SUMMARY_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
CHAT_SUMMARY_ROUNDS = int(os.getenv("CHAT_SUMMARY_ROUNDS", "20"))
CHAT_SUMMARY_KEEP_ROUNDS = int(
    os.getenv("CHAT_SUMMARY_KEEP_ROUNDS", os.getenv("CHAT_SUMMARY_ROUNDS", "20"))
)
CHAT_SUMMARY_MAX_INPUT_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_INPUT_CHARS", "48000"))
CHAT_SUMMARY_MAX_MSG_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_MSG_CHARS", "6000"))
AGENT_CHECKPOINT_ENABLED = os.getenv("AGENT_CHECKPOINT_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
UPLOAD_META_TTL_SEC = int(os.getenv("UPLOAD_META_TTL_SEC", str(7 * 24 * 3600)))
HITL_ENABLED = os.getenv("HITL_ENABLED", "1").strip().lower() not in ("0", "false", "no")
