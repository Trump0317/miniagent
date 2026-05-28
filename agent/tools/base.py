from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Type, Dict
from pydantic import BaseModel, ValidationError


def _minify_schema(schema: dict) -> dict:
    """瘦身 JSON Schema：去除 LLM 不需要的冗余字段，缩减 token 消耗。

    处理：
    - 删除 title / default / additionalProperties
    - anyOf[{type:X},{type:null}] → 直接保留 non-null 类型
    - 递归处理嵌套对象和数组
    """
    schema = deepcopy(schema)
    _walk(schema)
    return schema


def _walk(node: dict) -> None:
    """递归遍历并精简 schema 节点。"""
    # 移除冗余字段
    for key in ("title", "default", "additionalProperties"):
        node.pop(key, None)

    # anyOf[non-null, null] → 展开为 non-null 类型
    if "anyOf" in node and isinstance(node["anyOf"], list):
        non_null = [o for o in node["anyOf"] if o.get("type") != "null"]
        if len(non_null) == 1:
            merged = {k: v for k, v in node.items() if k != "anyOf"}
            merged.update(non_null[0])
            node.clear()
            node.update(merged)

    # 递归：properties
    for prop in node.get("properties", {}).values():
        if isinstance(prop, dict):
            _walk(prop)

    # 递归：items (array)
    if isinstance(node.get("items"), dict):
        _walk(node["items"])

    # 递归：$defs
    for d in node.get("$defs", {}).values():
        if isinstance(d, dict):
            _walk(d)

    # 递归：anyOf（多个类型的情况）
    for o in node.get("anyOf", []):
        if isinstance(o, dict) and o.get("type") != "null":
            _walk(o)


class Tool(ABC):
    """所有工具的基类。

    name / description / args_model 由 @tool 装饰器自动注入，子类无需手动维护。
    只有 execute() 是必须实现的核心方法。
    """

    # 声明该工具是否可在多线程环境下安全并发执行
    parallel_safe: bool = True

    # 声明该工具是否支持流式输出（边执行边产出中间结果）
    supports_streaming: bool = False

    def stream_execute(self, **kwargs):
        """流式执行工具，边执行边产出中间文本。

        重写此方法的工具必须同时设置 supports_streaming = True。
        子类不需要重写时直接调用 self.execute(**kwargs)。
        """
        yield self.execute(**kwargs)

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
        """JSON Schema 参数定义，供 LLM function calling 使用（已瘦身）"""
        return _minify_schema(self.args_model.model_json_schema())

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
