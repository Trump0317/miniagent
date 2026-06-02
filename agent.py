#!/usr/bin/env python3
"""miniagent — 基于 LLM 的智能助手

用法:
    python agent.py                        # 交互模式 (CLI)
    python agent.py --tui                  # TUI 模式
    python agent.py -p "列出文件"          # print 模式
    python agent.py --tui --thinking high  # TUI + 思考级别
"""

import sys

if __name__ == "__main__":
    if "--tui" in sys.argv:
        sys.argv.remove("--tui")
        from agent.tui import main
        main()
    else:
        from agent.cli import main
        main()
