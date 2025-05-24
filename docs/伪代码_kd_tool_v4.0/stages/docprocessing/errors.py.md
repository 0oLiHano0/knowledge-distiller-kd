```python
# ------------------------------------------------------------------------------
# 文件名: knowledge_distiller_kd/stages/docprocessing/errors.py
# 模块: P03 - 文档处理阶段 (DocumentProcessingStage) - 自定义异常
# 描述:
#   此模块定义了 P03 DocumentProcessingStage (原始提取阶段)
#   及其相关操作可能抛出的特定异常。
#   所有在此定义的异常都必须继承自核心异常类 KDToolError。
# 架构约束:
#   - 所有异常必须继承自 KDToolError (或其子类)。
#   - 异常应携带足够上下文信息 (通过 **kwargs 传递给 KDToolError)。
#   - 异常命名应清晰反映错误类型。
#   - 'module' 上下文信息在此基类中固定为 'DocumentProcessingStageP03'。
# ------------------------------------------------------------------------------

from typing import Optional, Any
from pathlib import Path # <-- 确保导入 Path

# 导入核心基础异常类 (必须继承)
# 假设其定义在 knowledge_distiller_kd/core/errors.py
from knowledge_distiller_kd.core.errors import KDToolError

# ==============================================================================
# P03 - 文档处理阶段基础异常 (DocumentProcessingError)
# ==============================================================================

class DocumentProcessingError(KDToolError):
    """
    P03 - 文档处理阶段 (原始提取) 中发生的错误的基类。

    架构说明:
        - 这是所有 P03 阶段特定错误的父类。
        - 在构造时，固定 'module' 上下文信息为 'DocumentProcessingStageP03'，
          以区分于其他可能的文档处理相关错误（例如未来更高级的处理阶段）。
    """
    def __init__(
        self,
        message: str,
        original_exception: Optional[Exception] = None,
        file_path: Optional[Path] = None, # 标准化，期望所有子类都可能提供
        **kwargs: Any
    ):
        """
        构造 DocumentProcessingError。

        参数:
            message (str): 描述错误的主要信息。
            original_exception (Optional[Exception]): 触发此异常的原始异常 (如果有)。
            file_path (Optional[Path]): 发生错误时正在处理的文件路径 (如果适用)。
            **kwargs: 任何需要传递给 KDToolError 以增强上下文的额外键值对。
        """
        # 将 file_path (如果提供) 转换为字符串并添加到 kwargs，以便 KDToolError 处理
        if file_path:
            kwargs['file_path'] = str(file_path)

        super().__init__(
            message,
            original_exception=original_exception,
            module="DocumentProcessingStageP03",  # <-- 强制模块上下文
            **kwargs
        )

# ==============================================================================
# P03 - 文档处理阶段特定异常
# ==============================================================================

class FileReadError(DocumentProcessingError):
    """
    当尝试读取输入文件内容，但在文件系统层面发生I/O错误时抛出。
    例如：文件不存在、权限不足等。

    架构说明:
        - **coding 阶段要求**: 在进行任何文件内容读取操作（如 `open(file_path, 'rb')`）
          之前或期间，捕获标准的 `FileNotFoundError`, `PermissionError`, `IOError` 等，
          并包装为此异常后重新抛出。
        - **必须包含** `file_path` 作为上下文信息（已由基类处理）。
    """
    def __init__(
        self,
        file_path: Path,
        original_exception: Exception # 通常由标准 I/O 异常触发
    ):
        message = f"读取文件失败: {file_path}"
        # file_path 会被基类的 __init__ 处理并放入 kwargs
        super().__init__(message, original_exception=original_exception, file_path=file_path)

class UnsupportedFileTypeError(DocumentProcessingError):
    """
    当尝试处理一个未在 `DocumentProcessingStageSettings.supported_extensions`
    中列出，或底层解析库明确表示不支持的文件类型时抛出。

    架构说明:
        - **coding 阶段要求**: 在调用底层解析库之前，应首先根据
          `settings.supported_extensions` 检查文件扩展名。如果不匹配，则抛出此异常。
          此外，如果底层解析库（如 unstructured）在尝试解析时能识别出无法处理的格式，
          也应捕获其特定异常并包装为此错误。
        - **必须包含** `file_path` 和检测到的 `file_type` (通常是扩展名) 作为上下文。
    """
    def __init__(
        self,
        file_path: Path,
        detected_file_type: str, # 例如 ".xyz" 或解析库返回的 MIME 类型
        original_exception: Optional[Exception] = None
    ):
        message = f"不支持的文件类型 '{detected_file_type}' (文件: {file_path})。"
        super().__init__(
            message,
            original_exception=original_exception,
            file_path=file_path,
            detected_file_type=detected_file_type # <-- 添加 'detected_file_type' 到上下文
        )

class ParsingError(DocumentProcessingError):
    """
    当调用底层库 (如 unstructured) 解析文件内容时，该库内部发生错误时抛出。

    架构说明:
        - **coding 阶段要求**: 在调用例如 `unstructured.partition.auto.partition()`
          或类似功能的代码块中，必须捕获其可能抛出的所有相关异常
          (例如，来自 `unstructured` 内部的特定解析错误、依赖库错误等)，
          并包装为此异常后重新抛出。
        - **必须包含** `file_path` 和实际使用的 `parser_name` (例如 "unstructured")
          以及原始异常作为上下文信息。
    """
    def __init__(
        self,
        file_path: Path,
        parser_name: str, # 例如 "unstructured" 或更具体的解析器名称
        original_exception: Exception # 通常由底层解析库的异常触发
    ):
        message = f"使用解析器 '{parser_name}' 解析文件 '{file_path}' 时失败。"
        super().__init__(
            message,
            original_exception=original_exception,
            file_path=file_path,
            parser_name=parser_name # <-- 添加 'parser_name' 到上下文
        )

class DTOConversionError(DocumentProcessingError):
    """
    当将底层解析库返回的原始解析元素转换为我们自定义的 `ContentBlockDTO` 时发生错误时抛出。
    例如：原始元素缺少必要字段、类型不匹配无法转换等。

    架构说明:
        - **coding 阶段要求**: 在 `_convert_raw_elements_to_preliminary_dtos` 或类似
          方法中，当尝试从原始元素创建 `ContentBlockDTO` 实例时，如果遇到
          例如 `KeyError` (缺少字段), `ValueError` (值不合法), `TypeError` (类型不匹配)
          等问题，必须捕获这些错误并包装为此异常后重新抛出。
        - **必须包含** `file_path` 和一些关于导致错误的 `element_info` (例如元素索引或类型)
          作为上下文信息。
    """
    def __init__(
        self,
        file_path: Path,
        element_info: str, # 例如 "原始元素在索引 5 (类型: Title)"
        original_exception: Exception # 通常由数据访问或类型转换错误触发
    ):
        message = f"在文件 '{file_path}' 中，转换元素 '{element_info}' 为 ContentBlockDTO 时失败。"
        super().__init__(
            message,
            original_exception=original_exception,
            file_path=file_path,
            element_info=element_info # <-- 添加 'element_info' 到上下文
        )

# 架构说明:
#   P03 阶段的错误主要集中在文件本身的处理和初步内容的提取。
#   与后续阶段（如块合并 P04）相关的逻辑错误应在各自阶段定义。
```