"""应用配置 —— 集中管理所有可配置项，支持多 Provider。"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
from openai import OpenAI

# ── Provider 预设 ──
# 每个 Provider 定义了：环境变量名（API Key / Base URL）、默认模型和子代理模型。
PROVIDER_PRESETS: dict[str, dict] = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "env_base": "DEEPSEEK_API_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-v4-flash",
        "subagent_model": "deepseek-v4-flash",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "env_base": "OPENAI_API_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "subagent_model": "gpt-4o-mini",
    },
    "custom": {
        "env_key": "API_KEY",
        "env_base": "API_BASE_URL",
        "default_base_url": "",
        "default_model": "deepseek-v4-flash",
        "subagent_model": "deepseek-v4-flash",
    },
}


@dataclass
class AppConfig:
    """Agent 应用的集中配置。

    用法:
        # 自动检测 .env 中的 Provider
        config = AppConfig.from_env()

        # 显式指定 Provider
        config = AppConfig.from_env(provider="openai")

        # 完全手动
        config = AppConfig(
            api_key="sk-xxx",
            api_base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
    """

    # ── Provider ──
    provider: str = "deepseek"
    model: str = ""
    api_key: str = ""
    api_base_url: str = ""

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
    subagent_model: str = ""
    subagent_max_turns: int = 15

    # ── 会话 ──
    restore_session: bool = True  # 启动时恢复上次对话

    # ── 上下文文件 ──
    context_files: str = ""  # AGENTS.md 等上下文文件内容

    def __post_init__(self):
        # 路径默认值
        if self.memory_dir is None:
            self.memory_dir = self.root / "agent" / ".memory"
        if self.skills_dir is None:
            self.skills_dir = self.root / "skills"

        # Provider 预设 → 填充默认 model / base_url
        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS["custom"])
        if not self.model:
            self.model = preset["default_model"]
        if not self.api_base_url:
            self.api_base_url = preset["default_base_url"]
        if not self.subagent_model:
            self.subagent_model = preset.get("subagent_model", self.model)

    @classmethod
    def from_env(cls, **overrides) -> AppConfig:
        """从 .env 和系统环境变量智能加载配置。

        检测顺序：DEEPSEEK_API_KEY → OPENAI_API_KEY → API_KEY（custom）。
        用户可通过 overrides 覆盖任意字段（如 provider="openai"）。
        """
        from dotenv import load_dotenv

        load_dotenv()

        # 确定 provider：显式指定 > 环境变量自动检测
        provider = overrides.pop("provider", None)
        if not provider:
            for name, preset in PROVIDER_PRESETS.items():
                if name == "custom":
                    continue
                if os.getenv(preset["env_key"]):
                    provider = name
                    break
            else:
                provider = "custom"

        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])

        return cls(
            provider=provider,
            api_key=overrides.pop("api_key", None)
            or os.getenv(preset["env_key"], ""),
            api_base_url=overrides.pop("api_base_url", None)
            or os.getenv(preset["env_base"], "")
            or preset["default_base_url"],
            model=overrides.pop("model", None) or preset["default_model"],
            subagent_model=overrides.pop("subagent_model", None)
            or preset.get("subagent_model", preset["default_model"]),
            **overrides,
        )

    def create_client(self) -> OpenAI:
        """根据配置创建 OpenAI 兼容客户端"""
        return OpenAI(api_key=self.api_key, base_url=self.api_base_url)
