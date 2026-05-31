"""使用智谱 GLM-4V 解析用户上传的图片。"""
from __future__ import annotations

import base64
from pathlib import Path

import env_config  # noqa: F401 — 加载 .env
from env_config import get_zhipuai_api_key

from llm_zhipu import make_chat_llm
from langchain_core.messages import HumanMessage

IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def describe_image_glm4v(image_path: Path, hint: str = "") -> str:
    api_key = get_zhipuai_api_key()

    ext = image_path.suffix.lower()
    mime = IMAGE_MIME.get(ext, "image/jpeg")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    prompt = hint.strip() or (
        "请详细描述这张图片的内容，包括可见文字、图表、界面元素和关键信息。"
        "若图片主要是文字截图，请尽量逐段转写。"
    )
    llm = make_chat_llm(model="glm-4v", temperature=0.0, api_key=api_key)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ]
    )
    resp = llm.invoke([msg])
    content = resp.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts).strip()
    return str(content).strip()
