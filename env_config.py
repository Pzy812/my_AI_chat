"""从项目根目录 .env 加载环境变量（各模块统一从此读取，勿在业务代码里写密钥）。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# 优先加载项目根 .env；Docker Compose 注入的环境变量不会被覆盖（override=False）
load_dotenv(BASE_DIR / ".env", override=False)


def get_zhipuai_api_key() -> str:
    key = os.getenv("ZHIPUAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "未配置 ZHIPUAI_API_KEY。请在项目根目录创建 .env 文件并写入：\n"
            "ZHIPUAI_API_KEY=你的密钥"
        )
    return key


def ensure_zhipuai_api_key_in_environ() -> str:
    """确保 os.environ 中有 ZHIPUAI_API_KEY（供 LangChain 等库读取）。"""
    key = get_zhipuai_api_key()
    os.environ["ZHIPUAI_API_KEY"] = key
    return key
