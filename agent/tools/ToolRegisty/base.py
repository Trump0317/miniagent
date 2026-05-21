from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Type, Dict
from pydantic import BaseModel, ValidationError


class Tool(ABC):
    """
    所有的工具类必须继承此类。
    使用 Pydantic 进行参数校验和转换。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        ...

    @property
    @abstractmethod
    def args_model(self) -> Type[BaseModel]:
        """Pydantic 模型，用于定义和校验参数"""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """展示符合 JSON Schema 标准的参数定义，供 LLM 使用"""
        return self.args_model.model_json_schema()

    def cast_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        利用 Pydantic 模型将原始字典进行类型转换和校验。
        返回转换后的字典，如果失败则抛出 ValueError。
        """
        try:
            return self.args_model.model_validate(params).model_dump()
        except ValidationError as e:
            raise ValueError(f"Parameter validation failed for {self.name}: {e}") from e

    def validate_params(self, params: Dict[str, Any]) -> None:
        """校验参数是否合法"""
        self.cast_params(params)

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具的核心逻辑"""
        ...


def tool(parameters: Type[BaseModel], name: str | None = None, description: str | None = None):
    """
    类装饰器：将 Pydantic 模型绑定到 Tool 子类上。
    同时支持通过装饰器快速定义工具名称和描述。
    """
    def wrap(cls):
        cls._args_model = parameters
        cls.args_model = property(lambda self: type(self)._args_model)
        if name:
            cls.name = property(lambda self: name)
        if description:
            cls.description = property(lambda self: description)
        elif cls.__doc__ and "description" in getattr(cls, "__abstractmethods__", set()):
            cls.description = property(lambda self: cls.__doc__.strip())

        implemented = ["args_model"]
        if name: implemented.append("name")
        if description or (cls.__doc__ and "description" in getattr(cls, "__abstractmethods__", set())):
            implemented.append("description")

        if hasattr(cls, "__abstractmethods__"):
            cls.__abstractmethods__ = frozenset(
                m for m in cls.__abstractmethods__ if m not in implemented
            )
        return cls
    return wrap