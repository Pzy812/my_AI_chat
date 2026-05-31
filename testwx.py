#!/usr/bin/env python3
"""
本地微信连接诊断（配合 mcp_server.py 使用的免费 wxauto4）。

用法（请先登录 PC 微信并保持主窗口可见，不要最小化到托盘）:
    conda activate agent_env
    cd E:\\agent\\前端AI发消息
    python testwx.py

可选:
    python testwx.py --no-banner    # 初始化时关闭 wxauto4 广告横幅
    python testwx.py --probe-x4     # 顺带探测 wxautox4（wxauto-mcp 依赖，收费版）
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import time
import traceback
from typing import Any

# wxauto4 官方说明：客户端需为微信 4.0.5，其它 4.x 小版本可能无法挂载
WXAUTO4_SUPPORTED_CLIENT = "4.0.5"
WXAUTO4_CLIENT_DOWNLOAD = (
    "https://github.com/SiverKing/wechat4.0-windows-versions/releases"
)


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _ok(msg: str) -> None:
    _print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    _print(f"[FAIL] {msg}")


def _warn(msg: str) -> None:
    _print(f"[WARN] {msg}")


def _info(msg: str) -> None:
    _print(f"[INFO] {msg}")


def _section(title: str) -> None:
    _print()
    _print("=" * 60)
    _print(title)
    _print("=" * 60)


def _wechat_processes() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        _warn("未安装 psutil，跳过进程检测（pip install psutil）")
        return []

    # 经典 PC 微信 + 4.x (xwechat) 子进程
    names = {"wechat.exe", "wechatapp.exe", "wechatappex.exe"}
    found: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in names:
                exe = proc.info.get("exe") or ""
                found.append(
                    {
                        "pid": proc.info.get("pid"),
                        "name": proc.info.get("name"),
                        "exe": exe,
                        "is_xwechat": "xwechat" in exe.replace("\\", "/").lower(),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _analyze_wechat_procs(procs: list[dict[str, Any]]) -> None:
    has_classic = any(
        (p.get("name") or "").lower() == "wechat.exe" for p in procs
    )
    has_xwechat = any(p.get("is_xwechat") for p in procs)
    if has_xwechat and not has_classic:
        _warn(
            "仅检测到微信 4.x (xwechat / WeChatAppEx) 子进程，未见经典 WeChat.exe。"
            " wxauto4 需能识别「已登录主窗口」；请把聊天主界面点开在前台后再测。"
        )
    if has_classic:
        _ok("存在经典 WeChat.exe 进程")


def _common_wechat_exe_paths() -> list[str]:
    candidates = [
        os.path.expandvars(r"%ProgramFiles%\Tencent\WeChat\WeChat.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Tencent\WeChat\WeChat.exe"),
        os.path.expandvars(
            r"%LocalAppData%\Tencent\WeChat\WeChat.exe"
        ),
    ]
    return [p for p in candidates if p and os.path.isfile(p)]


def check_environment() -> bool:
    _section("1. 运行环境")
    _info(f"Python: {sys.executable}")
    _info(f"版本:   {sys.version.split()[0]}")
    _info(f"平台:   {platform.platform()}")

    if "agent_env" in sys.executable.replace("/", "\\").lower():
        _ok("当前解释器路径包含 agent_env（与 pip install wxauto-mcp 环境一致）")
    else:
        _warn(
            "当前可能不是 agent_env；mcp_server 若用别的 Python 会找不到 wxauto4。"
            " 建议: conda activate agent_env 后再运行本脚本与 mcp_server.py"
        )

    if sys.platform != "win32":
        _fail("微信 UI 自动化仅支持 Windows")
        return False
    _ok("Windows 平台")
    return True


def check_import_wxauto4() -> tuple[bool, Any]:
    _section("2. wxauto4（免费，mcp_server.py 使用）")
    try:
        from wxauto4 import WeChat  # noqa: F401

        import wxauto4
        from wxauto4.param import VERSION as lib_ver

        _ok(f"已安装 wxauto4 @ {wxauto4.__file__}")
        _info(f"wxauto4 库版本: {lib_ver}（≠ 微信客户端版本）")
        _info(
            f"官方要求 PC 微信客户端为 {WXAUTO4_SUPPORTED_CLIENT}；"
            f"其它 4.x 构建号常出现「未找到已登录主窗口」"
        )
        return True, WeChat
    except ImportError as e:
        _fail(f"无法 import wxauto4: {e}")
        _info("修复: conda activate agent_env && pip install wxauto4")
        return False, None


def check_import_wxautox4(probe: bool) -> bool:
    _section("3. wxautox4（收费，仅 wxauto-mcp 官方包需要）")
    if not probe:
        _info("未加 --probe-x4，跳过（与 mcp_server 的 wxauto4 无关）")
        return True
    try:
        import wxautox4

        _ok(f"已安装 wxautox4 @ {wxautox4.__file__}")
        _warn("wxautox4 需激活密钥；本项目的 send_message 走的是 wxauto4，不是 x4")
        return True
    except ImportError as e:
        _warn(f"未安装 wxautox4: {e}（不影响仅用 wxauto4 的 mcp_server）")
        return True


def check_wechat_running() -> bool:
    _section("4. 本机微信进程")
    procs = _wechat_processes()
    if procs:
        for p in procs:
            tag = " [xwechat]" if p.get("is_xwechat") else ""
            _ok(f"发现进程 {p['name']} pid={p['pid']}{tag}")
            _info(f"       exe={p.get('exe') or '?'}")
        _analyze_wechat_procs(procs)
        return True

    _fail("未发现 WeChat.exe / WeChatApp.exe 等进程")
    paths = _common_wechat_exe_paths()
    if paths:
        _info("本机可能的微信安装路径:")
        for p in paths:
            _info(f"  {p}")
        _info("可先手动启动微信并登录，再重新运行: python testwx.py")
    else:
        _info("未在常见路径找到 WeChat.exe，请确认已安装 PC 版微信")
    return False


def _guess_xwechat_build(procs: list[dict[str, Any]]) -> str | None:
    for p in procs:
        exe = (p.get("exe") or "").replace("\\", "/")
        m = re.search(r"/(\d{4,6})/extracted/", exe)
        if m:
            return m.group(1)
    return None


def _find_wechat_top_windows() -> list[tuple[int, str, str]]:
    from wxauto4.utils import GetAllWindows

    hits: list[tuple[int, str, str]] = []
    for hwnd, cls, title in GetAllWindows():
        if cls == "Qt51514QWindowIcon" and title in ("微信", "Weixin"):
            hits.append((hwnd, cls, title))
    return hits


def _uia_tree_stats(hwnd: int) -> dict[str, Any]:
    from wxauto4.uia import uiautomation as uia

    root = uia.ControlFromHandle(hwnd)
    if not root.Exists(0):
        return {"exists": False, "nodes": 0, "has_mmui": False, "mmui_children": 0}

    nodes = 0
    has_mmui = False
    mmui_children = 0

    def walk(ctrl, depth: int = 0) -> None:
        nonlocal nodes, has_mmui, mmui_children
        nodes += 1
        if (ctrl.ClassName or "") == "MMUIRenderSubWindow":
            has_mmui = True
            try:
                mmui_children = max(mmui_children, len(ctrl.GetChildren()))
            except Exception:
                pass
        if depth >= 8:
            return
        try:
            children = ctrl.GetChildren()
        except Exception:
            return
        for c in children[:40]:
            walk(c, depth + 1)

    walk(root)
    return {
        "exists": True,
        "nodes": nodes,
        "has_mmui": has_mmui,
        "mmui_children": mmui_children,
    }


def check_wechat_windows_ui(*, foreground: bool) -> bool:
    """检查能否看到 wxauto4 需要的 Qt 主窗口，以及 UIA 子控件是否足够深。"""
    _section("4b. 微信主窗口与无障碍(UIA)树")
    wins = _find_wechat_top_windows()
    if not wins:
        _fail("未找到标题为「微信」的 Qt51514QWindowIcon 顶层窗口")
        _info("请打开微信主界面（不要只在托盘），再运行本脚本")
        return False

    for hwnd, cls, title in wins:
        _ok(f"顶层窗口 hwnd=0x{hwnd:X} class={cls!r} title={title!r}")

    main = next((w for w in wins if w[2] == "微信"), wins[0])
    hwnd = main[0]

    if foreground:
        try:
            import win32con
            import win32gui

            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            win32gui.MoveWindow(hwnd, 80, 80, 1200, 900, True)
            time.sleep(2)
            _info("已尝试将微信主窗口置前并放大到 1200x900（wxauto4 仅对可见区域注册 UIA）")
        except Exception as e:
            _warn(f"无法自动置前微信窗口: {e}")

    stats = _uia_tree_stats(hwnd)
    if not stats["exists"]:
        _fail("ControlFromHandle 无法访问该窗口")
        return False

    _info(
        f"UIA 节点数(深度≤8): {stats['nodes']}；"
        f"MMUIRenderSubWindow 子节点: {stats['mmui_children']}"
    )

    # 实测：能连上时通常 nodes>>5 且 mmui_children>0；仅 3 个节点多为版本不兼容
    if stats["nodes"] <= 5 and stats["mmui_children"] == 0:
        _fail(
            "微信窗口在，但内部聊天控件未暴露给 UIA（树过浅）。"
            " wxauto4 会报「未找到已登录的客户端主窗口」。"
        )
        return False

    _ok("UIA 树深度正常，可继续尝试 WeChat()")
    return True


def _print_failure_recommendations(procs: list[dict[str, Any]]) -> None:
    _section("诊断结论与建议")
    build = _guess_xwechat_build(procs)
    if build:
        _warn(f"检测到 xwechat 插件构建号约 {build}（路径中的数字），可能不是 {WXAUTO4_SUPPORTED_CLIENT}")

    _info("你的环境：Python/agent_env 正常，微信进程在跑，但 wxauto4 挂不上主界面。")
    _info("高概率原因：微信 4.x 小版本与 wxauto4 不匹配（官方仅保证 4.0.5）。")
    _info("建议按顺序尝试：")
    _info(f"  1) 安装/改用微信 PC 客户端 {WXAUTO4_SUPPORTED_CLIENT}")
    _info(f"     下载: {WXAUTO4_CLIENT_DOWNLOAD}")
    _info("  2) 退出当前微信，用 4.0.5 登录后，主窗口置前，再跑:")
    _info("     python testwx.py --no-banner --foreground")
    _info("  3) 关闭多余「Weixin」窗口，只保留一个已登录主号")
    _info("  4) mcp_server / app 必须用同一解释器:")
    _info("     C:\\Users\\HP\\.conda\\envs\\agent_env\\python.exe")
    _warn(
        "wxautox4（wxauto-mcp 自带）需 plus 授权，与免费 wxauto4 不是同一条链路；"
        " 未授权时会提示「未授权设备」。"
    )


def connect_wechat(WeChat: Any, *, ads: bool) -> Any | None:
    _section("5. 连接微信窗口（wxauto4.WeChat）")
    _info("正在 attach 已登录的微信主窗口（约 5–15 秒）…")
    _info("请确保: 已扫码登录、主界面未仅缩在托盘、未被其它自动化占用")

    try:
        wx = WeChat(ads=ads)
    except Exception as e:
        _fail(f"WeChat() 初始化失败: {e}")
        _info("常见原因:")
        _info("  1) 微信未登录或只有登录小窗 → 完成登录并打开主界面")
        _info("  2) 微信在托盘 → 点开主窗口到前台")
        _info("  3) 多开/版本过新 → 关闭多余实例后重试")
        _info("  4) 用了非 agent_env 的 Python 跑 mcp_server → 与上面 pip 环境不一致")
        _info("  5) 微信版本不是 4.0.5 → 见本脚本末尾「诊断结论」")
        _print()
        traceback.print_exc()
        return None

    _ok("WeChat() 初始化成功")
    return wx


def exercise_wechat(wx: Any) -> bool:
    _section("6. 基础能力探测")
    ok = True

    if hasattr(wx, "IsOnline"):
        try:
            online = wx.IsOnline()
            if online:
                _ok(f"IsOnline() = {online}")
            else:
                _warn(f"IsOnline() = {online}（已连接但状态为离线？）")
        except Exception as e:
            _warn(f"IsOnline() 调用异常: {e}")

    try:
        info = wx.GetMyInfo()
        _ok(f"GetMyInfo() = {info!r}")
    except Exception as e:
        _fail(f"GetMyInfo() 失败: {e}")
        ok = False

    try:
        sessions = wx.GetSession()
        n = len(sessions) if sessions is not None else 0
        _ok(f"GetSession() 返回 {n} 个会话")
        if sessions and n > 0:
            preview = sessions[:3]
            _info(f"前几条会话样例: {preview!r}")
    except Exception as e:
        _warn(f"GetSession() 失败（有时仍可调 SendMsg）: {e}")

    try:
        chat = wx.ChatInfo()
        _ok(f"ChatInfo() = {chat!r}")
    except Exception as e:
        _warn(f"ChatInfo() 失败: {e}")

    return ok


def dry_run_sendmsg(wx: Any) -> None:
    _section("7. SendMsg 干跑（默认不真发）")
    _info("未执行真实发送。若需自测发送，请自行在交互里调用:")
    _info('  wx.SendMsg("测试", "文件传输助手")')
    _info("（请勿对陌生人误发；建议仅发给「文件传输助手」）")


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="wxauto4 微信连接诊断")
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="WeChat(ads=False)，关闭 wxauto4 启动时的推广输出",
    )
    parser.add_argument(
        "--probe-x4",
        action="store_true",
        help="额外检查 wxautox4 是否已安装（收费版，非 mcp_server 必需）",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="检测前将微信主窗口置前并放大（有助于 UIA 注册）",
    )
    args = parser.parse_args()
    procs: list[dict[str, Any]] = []

    _print("wxauto4 微信连接测试 — 与 mcp_server.py 使用同一套库")
    _print(f"工作目录: {os.getcwd()}")

    if not check_environment():
        return 1

    ok_imp, WeChat = check_import_wxauto4()
    if not ok_imp or WeChat is None:
        return 1

    check_import_wxautox4(args.probe_x4)

    if not check_wechat_running():
        return 2

    procs = _wechat_processes()
    ui_ok = check_wechat_windows_ui(foreground=args.foreground)
    if not ui_ok:
        _print_failure_recommendations(procs)
        return 3

    wx = connect_wechat(WeChat, ads=not args.no_banner)
    if wx is None:
        _print_failure_recommendations(procs)
        return 3

    if not exercise_wechat(wx):
        return 4

    dry_run_sendmsg(wx)

    _section("结果")
    _ok("wxauto4 已能连接本机微信；mcp_server 的 send_message 在相同 Python 下应可用")
    _info("下一步: 用同一解释器启动 MCP")
    py = sys.executable
    _info(f'  "{py}" mcp_server.py')
    _info(f'  "{py}" app.py')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _print("\n已中断")
        raise SystemExit(130)
