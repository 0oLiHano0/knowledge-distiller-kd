"""
接口与契约测试的辅助函数。
"""
from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, ForwardRef, Type, Union, get_args, get_origin

# ---------------------------------------------------------------------------
# 公共 API 提取
# ---------------------------------------------------------------------------

def get_public_methods_from_class(klass: Type) -> dict[str, Callable]:
    """获取类的公共方法（不以下划线开头）。"""
    return {
        name: member
        for name, member in inspect.getmembers(klass)
        if not name.startswith("_") and (inspect.isfunction(member) or inspect.ismethod(member))
    }

# ---------------------------------------------------------------------------
# 类型归一化工具
# ---------------------------------------------------------------------------

_ALIAS_MAP = {
    typing.List: list,
    typing.Dict: dict,
    typing.Set: set,
    typing.Tuple: tuple,
    list: list,
    dict: dict,
    set: set,
    tuple: tuple,
}


def _strip_module(path: str) -> str:
    """删除模块前缀，仅保留最终标识符 (a.b.C → C)。"""
    return path.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# 解析字符串 / ForwardRef 注解
# ---------------------------------------------------------------------------

def _parse_str_annotation(s: str) -> Any:  # noqa: C901 复杂度无伤大雅
    """将字符串形式的类型注解解析为归一化结构。"""
    s = s.strip("'\"")

    # 1. None / NoneType → None
    if s in {"None", "NoneType"}:
        return None

    # 2. 若包含泛型 [ ... ]
    if "[" in s and s.endswith("]"):
        outer, inner = s.split("[", 1)
        inner = inner[:-1]  # 去掉末尾 ]
        outer = _strip_module(outer)

        # 分割多参数泛型，外层逗号分割，但内部可能还有嵌套 → 简化假设：无嵌套逗号
        # 若需更严谨，可用 typing 模块解析，但当前足够
        inner_parts = [p.strip() for p in inner.split(",") if p.strip()]
        inner_norm = tuple(_normalize_type(p) for p in inner_parts)

        if outer in {"Optional", "typing.Optional"}:
            # Optional[T] 等价于 Union[T, None]
            return ("Union", frozenset({None, inner_norm[0]}))
        if outer in {"Union", "typing.Union"}:
            return ("Union", frozenset(inner_norm))
        if outer in {"List", "typing.List", "list"}:
            return (list, inner_norm)
        if outer in {"Set", "typing.Set", "set"}:
            return (set, inner_norm)
        if outer in {"Dict", "typing.Dict", "dict"}:
            return (dict, inner_norm)
        if outer in {"Tuple", "typing.Tuple", "tuple"}:
            return (tuple, inner_norm)

        # 其他未知泛型，保留 outer 名称
        return (outer, inner_norm)

    # 3. 非泛型前向引用：保留末尾标识符
    return _strip_module(s)


# ---------------------------------------------------------------------------
# 归一化与比较
# ---------------------------------------------------------------------------

def _normalize_type(tp: Any) -> Any:  # noqa: C901 复杂度 acceptable
    """将类型注解归一化为可哈希结构，使语义等价注解得到相同表示。"""

    # sentinel
    if tp in (inspect._empty, typing.Any):
        return tp

    # 字符串 / ForwardRef
    if isinstance(tp, (str, ForwardRef)):
        return _parse_str_annotation(str(tp))

    # NoneType → None （inspect 返回 type(None)）
    if tp is type(None):  # noqa: E721
        return None

    origin = get_origin(tp)

    # 纯类型（非泛型）
    if origin is None:
        if hasattr(tp, "__qualname__"):
            return _strip_module(tp.__qualname__)
        return tp

    # Union / Optional → 使用不可变集合强化忽略顺序
    if origin is Union:
        args_norm = frozenset(_normalize_type(a) for a in get_args(tp))
        return ("Union", args_norm)

    # 其他泛型
    origin_norm = _ALIAS_MAP.get(origin, origin)
    args_norm = tuple(_normalize_type(a) for a in get_args(tp))
    return (origin_norm, args_norm)


def _types_equal(a: Any, b: Any) -> bool:
    """深度比较注解等价性，额外兜底到字符串 repr。"""
    norm_a = _normalize_type(a)
    norm_b = _normalize_type(b)
    if norm_a == norm_b:
        return True
    # 退级比较：若人类可读 repr 相同也认为等价（极端情况）
    return _type_repr(a) == _type_repr(b)


def _type_repr(tp: Any) -> str:  # noqa: D401
    """生成简洁友好的注解字符串。"""
    if tp is inspect._empty:
        return "<未注解>"
    if tp is None or tp is type(None):  # noqa: E721
        return "None"
    if tp is typing.Any:
        return "typing.Any"
    try:
        return typing.get_type_hints({"_": tp})["_"].__repr__()
    except Exception:
        return repr(tp)

# ---------------------------------------------------------------------------
# 核心：方法签名比较
# ---------------------------------------------------------------------------

def compare_method_signatures(
    interface_method: Callable,
    implementation_method: Callable,
    class_name: str,
    method_name: str,
) -> None:
    """严格比较接口与实现方法签名，忽略等价注解书写差异。"""

    iface_sig = inspect.signature(interface_method)
    impl_sig = inspect.signature(implementation_method)

    # 1. 参数数量
    if len(iface_sig.parameters) != len(impl_sig.parameters):
        raise AssertionError(
            f"契约错误({class_name}.{method_name}): 参数数量不符。\n"
            f"  接口: {list(iface_sig.parameters.keys())}\n"
            f"  实现: {list(impl_sig.parameters.keys())}"
        )

    # 2. 参数详细比较
    for p_name, iface_p in iface_sig.parameters.items():
        if p_name not in impl_sig.parameters:
            raise AssertionError(
                f"契约错误({class_name}.{method_name}): 实现缺少参数 '{p_name}'."
            )
        impl_p = impl_sig.parameters[p_name]

        # 2a. kind
        if iface_p.kind != impl_p.kind:
            raise AssertionError(
                f"契约错误({class_name}.{method_name}): 参数 '{p_name}' 种类不符。\n"
                f"  接口: {iface_p.kind}  实现: {impl_p.kind}"
            )

        # 2b. 注解
        if iface_p.annotation not in (inspect._empty, typing.Any):
            if not _types_equal(iface_p.annotation, impl_p.annotation):
                raise AssertionError(
                    f"契约错误({class_name}.{method_name}): 参数 '{p_name}' 类型注解不符。\n"
                    f"  接口: {_type_repr(iface_p.annotation)}\n  实现: {_type_repr(impl_p.annotation)}"
                )

        # 2c. 默认值
        if iface_p.default != impl_p.default:
            raise AssertionError(
                f"契约错误({class_name}.{method_name}): 参数 '{p_name}' 默认值不符。\n"
                f"  接口: {iface_p.default}  实现: {impl_p.default}"
            )

    # 3. 返回注解
    iface_ret, impl_ret = iface_sig.return_annotation, impl_sig.return_annotation

    if iface_ret not in (inspect._empty, typing.Any):
        if iface_ret is None:
            if impl_ret is not None:
                raise AssertionError(
                    f"契约错误({class_name}.{method_name}): 返回类型注解不符。\n"
                    f"  接口期望 None (-> None)  实现: {_type_repr(impl_ret)}"
                )
        elif not _types_equal(iface_ret, impl_ret):
            raise AssertionError(
                f"契约错误({class_name}.{method_name}): 返回类型注解不符。\n"
                f"  接口: {_type_repr(iface_ret)}  实现: {_type_repr(impl_ret)}"
            )

    # 通过所有检查 → 签名兼容
