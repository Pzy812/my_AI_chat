"""wxauto4 兼容补丁：修复未读消息数 '2条未读' 无法 int() 的问题。"""

from __future__ import annotations

import re

from wxauto4.utils import tools as _tools

_ORIG_PARSE_SESSION_FORMAT = _tools.parse_session_format
_ORIG_PARSE_OLD_FORMAT = _tools.parse_old_format


def _parse_unread_count(value) -> int:
    if value in (None, "", 0):
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def _wrap_parse(parser):
    def safe_parse(text):
        info = parser(text)
        if info and info.get("unread_count") not in (None, "", 0):
            info["unread_count"] = _parse_unread_count(info["unread_count"])
        return info

    return safe_parse


_tools.parse_session_format = _wrap_parse(_ORIG_PARSE_SESSION_FORMAT)
_tools.parse_old_format = _wrap_parse(_ORIG_PARSE_OLD_FORMAT)
