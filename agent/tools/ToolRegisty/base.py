from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Type, Dict
from pydantic import BaseModel, ValidationError


class Tool(ABC):
    """所有工具的基类。

    name / description / args_model 由 @tool 装饰器自动注入，子类无需手动维护。
    只有 execute() 是必须实现的核心方法。
    """

    # 声明该工具是否可在多线程环境下安全并发执行
    parallel_safe: bool = True

    @property
    def name(self) -> str:
        """工具名称 —— 由 @tool 装饰器注入到 _tool_name"""
        return self.__class__._tool_name

    @property
    def description(self) -> str:
        """工具描述 —— 由 @tool 装饰器注入到 _tool_description"""
        return self.__class__._tool_description

    @property
    def args_model(self) -> Type[BaseModel]:
        """Pydantic 参数模型 —— 由 @tool 装饰器注入到 _args_model"""
        return self.__class__._args_model

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema 参数定义，供 LLM function calling 使用"""
        return self.args_model.model_json_schema()

    def cast_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """利用 Pydantic 模型对原始参数做类型转换和校验"""
        try:
            return self.args_model.model_validate(params).model_dump()
        except ValidationError as e:
            raise ValueError(f"参数校验失败 [{self.name}]: {e}") from e

    def validate_params(self, params: Dict[str, Any]) -> None:
        """仅校验，不返回转换结果"""
        self.cast_params(params)

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具的核心逻辑 —— 子类必须实现"""
        ...


def tool(
    parameters: Type[BaseModel],
    name: str | None = None,
    description: str | None = None,
):
    """类装饰器：将 name / description / args_model 注入到 Tool 子类。

    用法:
        @tool(name="my_tool", description="...", parameters=MyArgs)
        class MyTool(Tool):
            def execute(self, **kwargs) -> str: ...

    注入的类属性:
        _tool_name       → Tool.name 属性读这里
        _tool_description → Tool.description 属性读这里
        _args_model      → Tool.args_model 属性读这里
    """
    def wrap(cls):
        cls._args_model = parameters
        cls._tool_name = name or cls.__name__
        cls._tool_description = description or (cls.__doc__ or "").strip()
        return cls
    return wrap
