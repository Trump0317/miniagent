"""TUI 渲染工具 — Markdown 转 prompt_toolkit 格式化文本。"""

from __future__ import annotations
import re

from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer, TextLexer
from pygments.formatters import Terminal256Formatter


# ── 正则 ──
_RE_CODE_BLOCK = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_RE_H1 = re.compile(r"^# (.+)$", re.MULTILINE)
_RE_H2 = re.compile(r"^## (.+)$", re.MULTILINE)
_RE_H3 = re.compile(r"^### (.+)$", re.MULTILINE)
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _highlight_code(code: str, language: str | None = None) -> str:
    """对代码块做语法高亮，返回 ANSI 转义字符串。"""
    try:
        if language:
            lexer = get_lexer_by_name(language, stripall=True)
        else:
            lexer = guess_lexer(code) or TextLexer()
    except Exception:
        lexer = TextLexer()
    formatter = Terminal256Formatter(style="monokai")
    return highlight(code, lexer, formatter).rstrip("\n")


def _parse_ansi_to_styled(ansi_text: str) -> list[tuple[str, str]]:
    """将 ANSI 转义序列转换为 prompt_toolkit 的 (style, text) 格式。

    简化处理：提取 ANSI 颜色码并映射到 style class。
    """
    result: list[tuple[str, str]] = []
    # 移除 ANSI 序列，仅保留纯文本（代码块的样式通过 class 统一设置）
    clean = re.sub(r"\x1b\[[0-9;]*m", "", ansi_text)
    for line in clean.split("\n"):
        if result:
            result.append(("", "\n"))
        result.append(("class:md-code-block", line))
    return result


def _render_inline_markdown(text: str) -> list[tuple[str, str]]:
    """渲染行内 Markdown（code / bold / italic / link）。"""
    result: list[tuple[str, str]] = []
    pos = 0

    # 统一用正则扫描
    pattern = re.compile(
        r"(`[^`]+`)"           # inline code
        r"|(\*\*[^*\n]+\*\*)"  # bold
        r"|(\*[^*\n]+\*)"      # italic (simplified)
        r"|(\[([^\]]+)\]\(([^)]+)\))"  # link
    )

    for m in pattern.finditer(text):
        start, end = m.span()
        # 前置普通文本
        if start > pos:
            result.append(("", text[pos:start]))

        if m.group(1):  # inline code
            code = m.group(1)[1:-1]  # strip backticks
            result.append(("class:md-inline-code", code))
        elif m.group(2):  # bold
            bold = m.group(2)[2:-2]
            result.append(("class:md-bold", bold))
        elif m.group(3):  # italic
            italic = m.group(3)[1:-1]
            result.append(("class:md-italic", italic))
        elif m.group(4):  # link [text](url)
            link_text = m.group(5)
            result.append(("class:md-link", link_text))

        pos = end

    # 剩余文本
    if pos < len(text):
        result.append(("", text[pos:]))

    return result


def render_markdown(md_text: str) -> list[tuple[str, str]]:
    """将 Markdown 文本渲染为 prompt_toolkit 格式化文本列表。

    支持: #/##/### 标题, **粗体**, *斜体*, `行内代码`,
           ```代码块``` (pygments 高亮), [链接](url).
    """
    if not md_text:
        return [("", "")]

    # 预处理：用占位符替换代码块，避免代码块内容被行内正则误匹配
    code_blocks: list[tuple[str, str | None, str]] = []  # (placeholder, lang, code)

    def _save_block(m: re.Match) -> str:
        lang = m.group(1)
        code = m.group(2)
        placeholder = f"\x00CODEBLOCK{len(code_blocks)}\x00"
        code_blocks.append((placeholder, lang, code))
        return placeholder

    processed = _RE_CODE_BLOCK.sub(_save_block, md_text)

    # 按行处理
    lines = processed.split("\n")
    result: list[tuple[str, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 检查是否包含代码块占位符
        cb_match = re.match(r"^(.*?)\x00CODEBLOCK(\d+)\x00(.*)$", line)
        if cb_match:
            prefix = cb_match.group(1)
            idx = int(cb_match.group(2))
            suffix = cb_match.group(3)

            if prefix:
                result.extend(_render_inline_markdown(prefix))
                result.append(("", "\n"))

            _, lang, code = code_blocks[idx]
            highlighted = _highlight_code(code, lang)
            result.extend(_parse_ansi_to_styled(highlighted))

            if suffix:
                if result and result[-1][0] == "" and result[-1][1] == "":
                    pass
                else:
                    result.append(("", "\n"))
                result.extend(_render_inline_markdown(suffix))

            i += 1
            continue

        # 标题
        h1 = _RE_H1.match(line)
        h2 = _RE_H2.match(line)
        h3 = _RE_H3.match(line)

        if h1:
            if result:
                result.append(("", "\n"))
            result.append(("class:md-h1", h1.group(1)))
        elif h2:
            if result:
                result.append(("", "\n"))
            result.append(("class:md-h2", h2.group(1)))
        elif h3:
            if result:
                result.append(("", "\n"))
            result.append(("class:md-h3", h3.group(1)))
        else:
            # 普通行：渲染行内 markdown
            if i > 0:
                result.append(("", "\n"))
            result.extend(_render_inline_markdown(line))

        i += 1

    return result
