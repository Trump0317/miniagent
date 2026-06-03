from .core.agent import Agent
from .core.events import EventBus, Event
from .ai.config import AppConfig
from .ai.llm import LLMClient
from .cli.helpers import handle_tree, handle_fork, handle_back

__all__ = ["Agent", "EventBus", "Event", "AppConfig", "LLMClient",
           "handle_tree", "handle_fork", "handle_back"]


def main():
    """入口函数，由 pyproject.toml [project.scripts] 调用。"""
    import sys

    if "--web" in sys.argv:
        sys.argv.remove("--web")
        from agent.web.server import main as web_main
        web_main()
    elif "--tui" in sys.argv:
        sys.argv.remove("--tui")
        from agent.tui import main as tui_main
        tui_main()
    else:
        from agent.cli import main as cli_main
        cli_main()
