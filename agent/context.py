"""上下文文件加载 —— 从项目和用户目录自动加载 AGENTS.md / CLAUDE.md。"""

from __future__ import annotations
from pathlib import Path


def load_context_files(
    cwd: Path | None = None,
    user_dir: Path | None = None,
) -> str:
    """从项目目录链和用户配置目录加载上下文文件。

    加载顺序（后面的覆盖前面的语义）：
    1. 用户全局配置 (~/.miniagent/AGENTS.md)
    2. 从根目录到 cwd 沿途的 AGENTS.md / CLAUDE.md

    返回拼接后的上下文文本；无文件时返回空字符串。
    """
    cwd = (cwd or Path.cwd()).resolve()
    collected: list[str] = []

    # ── 1. 用户全局上下文 ──
    if user_dir:
        global_file = user_dir / "AGENTS.md"
        if global_file.exists():
            try:
                collected.append(global_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    # ── 2. 从根目录到 cwd 逐级加载 ──
    # 收集路径链（从根到 cwd）
    chain: list[Path] = []
    cur = cwd
    while True:
        chain.append(cur)
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    # 从根向下遍历（后面的追加，可覆盖前面的约定）
    for d in reversed(chain):
        for name in ("AGENTS.md", "CLAUDE.md"):
            f = d / name
            if f.exists():
                try:
                    text = f.read_text(encoding="utf-8")
                    if text.strip():
                        collected.append(
                            f"# 项目上下文 ({f.relative_to(cwd) if cwd in f.parents else f})\n\n{text}"
                        )
                except Exception:
                    pass
                break  # 每级目录只加载第一个匹配的文件

    return "\n\n---\n\n".join(collected) if collected else ""
