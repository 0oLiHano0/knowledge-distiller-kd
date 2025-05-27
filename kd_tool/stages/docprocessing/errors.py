from typing import Optional, Any
from pathlib import Path
from kd_tool.core.errors import KDToolError


class DocumentProcessingError(KDToolError):
    """
    P03 - 文档处理阶段 (原始提取) 中发生的错误的基类。

    架构说明:
        - 这是所有 P03 阶段特定错误的父类。
        - 在构造时，固定 'module' 上下文信息为 'DocumentProcessingStageP03'，
          以区分于其他可能的文档处理相关错误（例如未来更高级的处理阶段）。
    """

    def __init__(self, message: str, original_exception: Optional[Exception
        ]=None, file_path: Optional[Path]=None, **kwargs: Any):
        """
        构造 DocumentProcessingError。

        参数:
            message (str): 描述错误的主要信息。
            original_exception (Optional[Exception]): 触发此异常的原始异常 (如果有)。
            file_path (Optional[Path]): 发生错误时正在处理的文件路径 (如果适用)。
            **kwargs: 任何需要传递给 KDToolError 以增强上下文的额外键值对。
        """
        if file_path:
            kwargs['file_path'] = str(file_path)
        super().__init__(message, original_exception=original_exception,
            module='DocumentProcessingStageP03', **kwargs)


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

    def __init__(self, file_path: Path, original_exception: Exception):
        message = f'读取文件失败: {file_path}'
        super().__init__(message, original_exception=original_exception,
            file_path=file_path)


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

    def __init__(self, file_path: Path, detected_file_type: str,
        original_exception: Optional[Exception]=None):
        message = f"不支持的文件类型 '{detected_file_type}' (文件: {file_path})。"
        super().__init__(message, original_exception=original_exception,
            file_path=file_path, detected_file_type=detected_file_type)


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

    def __init__(self, file_path: Path, parser_name: str,
        original_exception: Exception):
        message = f"使用解析器 '{parser_name}' 解析文件 '{file_path}' 时失败。"
        super().__init__(message, original_exception=original_exception,
            file_path=file_path, parser_name=parser_name)


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

    def __init__(self, file_path: Path, element_info: str,
        original_exception: Exception):
        message = (
            f"在文件 '{file_path}' 中，转换元素 '{element_info}' 为 ContentBlockDTO 时失败。"
            )
        super().__init__(message, original_exception=original_exception,
            file_path=file_path, element_info=element_info)
