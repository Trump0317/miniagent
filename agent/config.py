"""应用配置 —— 集中管理所有可配置项，消除上帝方法。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from openai import OpenAI


@dataclass
class AppConfig:
    """Agent 应用的集中配置。

    使用 dataclass 而非散落的构造参数，好处：
    1. 一个地方看所有配置
    2. 可以用 AppConfig.load() 从文件/环境变量加载
    3. 测试时轻松替换
    """

    # ── 模型 ──
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    api_base_url: str = "https://api.deepseek.com/v1"

    # ── 路径 ──
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    memory_dir: Path | None = None
    skills_dir: Path | None = None

    # ── 运行时参数 ──
    max_turns: int | None = None
    max_tokens: int = 20_000
    max_context: int = 200_000
    compact_threshold: float = 0.7

    # ── 子代理 ──
    subagent_model: str = "deepseek-v4-flash"
    subagent_max_turns: int = 15

    def __post_init__(self):
        if self.memory_dir is None:
            self.memory_dir = self.root / "agent" / ".memory"
        if self.skills_dir is None:
            self.skills_dir = self.root / "skills"

    @classmethod
    def from_env(cls, **overrides) -> AppConfig:
        """从 .env 文件和环境变量加载配置。

        读取 DEEPSEEK_API_KEY 和 DEEPSEEK_API_BASE_URL，
        允许通过 overrides 覆盖任意字段。
        """
        import os
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            api_base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
            **overrides,
        )

    def create_client(self) -> OpenAI:
        """根据配置创建 OpenAI 兼容客户端"""
        return OpenAI(api_key=self.api_key, base_url=self.api_base_url)
