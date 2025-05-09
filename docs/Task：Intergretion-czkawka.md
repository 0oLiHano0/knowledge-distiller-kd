原子任务 1：实现 CzkawkaAdapter.filter_unique_files()
- 上下文
    - 路径：knowledge_distiller_kd/prefilter/czkawka_adapter.py
    - 目标：扫描给定目录，过滤掉重复文件，只保留 .md、.doc、.docx 三种扩展名
    - 二进制：vendor/czkawka/macos-arm64/czkawka
- 输入
    ```python
    input_dir: Path  # 例如 Path("input/")
    extensions: List[str] = [".md", ".doc", ".docx"]
    recursive: bool = True
    max_depth: Optional[int] = None
    ```
- 输出
    ```python
    unique_files: List[Path]        # 未重复的文件列表
    duplicate_groups: List[List[Path]]  # 每组重复文件的绝对路径列表
    ```
- 验收标准
    - 在 input/ 目录下运行时，返回 2 个 unique + 1 组 duplicate (2 个文件)
    - 对非 .md、.docx 文件不参与扫描
    - 提供单元测试：手动构造临时目录并断言输出