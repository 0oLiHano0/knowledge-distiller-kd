### 原子任务 1：实现 CzkawkaAdapter.filter_unique_files()
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

### 原子任务 2：添加 CLI 参数 `--pre-filter` 与 `--skip-prefilter`

**文件位置**  
    `knowledge_distiller_kd/ui/cli_interface.py`

**目标**  
    - 实现 `--pre-filter`：仅执行预过滤阶段（调用 `filter_unique_files()`），打印统计后退出  
    - 实现 `--skip-prefilter`：跳过预过滤阶段，直接进入后续分析流程  
    - 对同时使用两个冲突参数的场景，给出错误提示并退出  

**输入示例**  
    ```bash
    kd_tool scan-files --pre-filter --dir input/
    kd_tool scan-files --skip-prefilter --dir input/
    ```

输出要求

    - --pre-filter 模式下，打印：

    ```csharp
    [Prefilter] Scanned 4 files, filtered 2 duplicates → 2 files remain.
    ```
    然后退出，不做后续处理。

- --skip-prefilter：不调用过滤逻辑，直接继续后续分析。

- 验收标准
    --pre-filter 时只跑预过滤并退出，统计数字正确。
    --skip-prefilter 时跳过预过滤。
    同时传入两者时报参数冲突错误并退出。
    增加对应单元测试验证三种场景。

### 原子任务 2.1：修复 CLI 接口参数逻辑和属性缺失

**文件位置**  
- `knowledge_distiller_kd/ui/cli_interface.py`  
- `knowledge_distiller_kd/core/engine.py`

**目标**  
1. 在 `KnowledgeDistillerEngine` 中暴露并初始化 `input_dir` 属性  
2. 调整 CLI 中的参数冲突检测顺序，确保：  
   - 同时传入 `--pre-filter` 与 `--skip-prefilter` 时，报错并退出  
3. 在处理 `--pre-filter` 时，调用 `engine.run_prefilter_only()` 并正常退出（`sys.exit(0)`）  
4. 在处理 `--skip-prefilter` 时，仅设置 `engine.skip_prefilter = True`，不提前退出，继续后续流程  

**实现细节**  
1. **Engine 修改**  
   ```python
   class KnowledgeDistillerEngine:
       def __init__(self, input_dir: Optional[str] = None, skip_prefilter: bool = False):
           self.input_dir = input_dir
           self.skip_prefilter = skip_prefilter
           …
   ```
   或者在 CLI 初始化后，调用 `engine.set_input_dir(args.input_dir)`。

2. **CLI 参数处理**  
   ```python
   # 冲突检测放在最前
   if args.pre_filter and args.skip_prefilter:
       print("Error: --pre-filter and --skip-prefilter are mutually exclusive", file=sys.stderr)
       sys.exit(1)

   # 单独预过滤
   if args.pre_filter:
       engine.input_dir = args.input_dir
       engine.run_prefilter_only()
       sys.exit(0)

   # 跳过预过滤，继续后续流程
   if args.skip_prefilter:
       engine.input_dir = args.input_dir
       engine.skip_prefilter = True
   ```

3. **确保在后续 `run_analysis()` 中使用 `engine.input_dir`、`engine.skip_prefilter`**  

**验收标准**  
1. **属性初始化**：`engine.input_dir` 在 CLI 解析后不再触发 `AttributeError`  
2. **冲突场景**：同时传入 `--pre-filter` 与 `--skip-prefilter` 时，CLI 以状态码 1 退出并输出错误信息  
3. **预过滤场景**：仅 `--pre-filter` 的命令调用中，`run_prefilter_only()` 被调用一次，并以状态码 0 正常退出  
4. **跳过预过滤场景**：仅 `--skip-prefilter` 时，不调用 `run_prefilter_only()`，也不提前退出，后续流程正常进行  
5. **单元测试**：新增/修正测试用例，覆盖上述四种场景，并在 CI 中全部通过


### 原子任务 3 — 在 engine.run_analysis() 集成预过滤

1. 上下文

    - 文件：knowledge_distiller_kd/core/engine.py

    - 目标：在分析流程最前端，根据 engine.skip_prefilter 和 args.pre_filter 标志，调用 filter_unique_files()

    - 将过滤后得到的 unique_files 列表，替换原先的 input_path 文件集合，作为后续 Processing 层的输入

2. 主要工作

    - 在 run_analysis() 方法开头：
        ```python
        if not self.skip_prefilter:
            unique_files, duplicate_groups = filter_unique_files(self.input_dir, extensions, recursive, max_depth)
            log_info(f"[Prefilter] Scanned {total} files, filtered {filtered} duplicates → {len(unique_files)} remain.")
            # 如果是仅预过滤模式（来自 CLI 的 pre_filter_only），则在 run_prefilter_only() 中已退出
        else:
            unique_files = self._gather_input_files(self.input_dir)
        ```
    - 确保：
        - self.skip_prefilter 为 True 时跳过调用

        - 任何异常都能被 error_handler 捕获并友好提示

3. 输出与验收

    - CLI 运行 kd_tool scan-files --dir input/（无 flag）时：

        - 首先打印预过滤统计

        - 接着继续后续分块、MD5 分析等流程

    - 单元/集成测试：

        - Mock filter_unique_files()，验证其在 run_analysis() 中被正确调用与跳过

        - 验证 engine.process_blocks()、engine.analyze_blocks() 等后续调用仍能接收 unique_files


### 原子任务 4：实现预过滤日志 & 汇总输出

**总体原则**：

* 遵循TDD原则（先红后绿）
* 完成任务后根据任务内容撰写commit，并执行github推送
* 如需要修改现有业务逻辑需取简要说明原因和预期，并得用户确认

**文件位置**

* `knowledge_distiller_kd/core/engine.py`
* `knowledge_distiller_kd/core/utils.py` 或日志模块

**目标**

* 在预过滤阶段前后埋点，记录开始和结束时间
* 打印并记录以下统计信息：

  ```
  -[Prefilter] Scanned {total_files} files, filtered {filtered_count} duplicates → {unique_count} remain. (耗时: {elapsed_ms}ms)
  ```
* 使用 `loguru` INFO 级别输出，同时保证结构化日志记录该事件

**实现细节**

1. **埋点**：在调用 `filter_unique_files()` 前记录 `start_time = time.monotonic()`，调用后记录 `end_time`
2. **计算耗时**：`elapsed_ms = int((end_time - start_time) * 1000)`
3. **日志打印**：

   ```python
   logger.info(f"[Prefilter] Scanned {total_files} files, filtered {filtered_count} duplicates → {unique_count} remain. (耗时: {elapsed_ms}ms)")
   ```
4. **结构化日志**：附加上下文字段，例如

   ```python
   logger.bind(
       total_files=total_files,
       filtered_count=filtered_count,
       unique_count=unique_count,
       elapsed_ms=elapsed_ms
   ).info("prefilter_summary")
   ```
5. **单元测试**：使用 `pytest` 和 `monkeypatch` 模拟 `time.monotonic()`，并捕获日志输出，验证格式与内容。

**验收标准**

1. CLI 执行默认流程（无 `--skip-prefilter`）时，在控制台看到格式正确的预过滤统计日志
2. 日志级别为 INFO，结构化日志包含 `total_files`、`filtered_count`、`unique_count`、`elapsed_ms`
3. 单元测试能模拟时间并断言日志输出字符串和结构化字段
4. CI 流水线中新增测试通过，且覆盖率不下降


### 原子任务 5：编写 Smoke Test 脚本

**文件位置**

* `tests/`
* 推荐新文件：`tests/test_smoke_end_to_end.py`

**目标**

* 在 CI 中对 `/input/` 目录一键跑通两种模式：

  1. **`--pre-filter` 模式**：仅执行预过滤并验证输出统计日志后退出
  2. **默认模式（无 flag）**：执行完整流程，至少跑到 MD5 分析，并输出 MD5 重复对统计

**实现细节**

1. 使用 `subprocess` 或 `click.testing.CliRunner` 模拟命令行调用：

   ```python
   result = runner.invoke(cli.main, ['scan-files', '--pre-filter', '--dir', 'input/'])
   ```
2. **断言**：

   * **`--pre-filter` 模式**：

     * `result.exit_code == 0`
     * `result.output` 包含 `[Prefilter] Scanned 4 files, filtered 2 duplicates → 2 remain.`
     * **无**后续 MD5 分析日志出现
   * **默认模式**：

     * `result.exit_code == 0`
     * `result.output` 包含 `[Prefilter] Scanned 4 files, filtered 2 duplicates → 2 remain.`
     * `result.output` 包含 `MD5 duplicates found: 1 pairs`
3. 如果使用 `CliRunner`，需在 `conftest.py` 中导入并准备 `runner` fixture；如果用 `subprocess`, 确保在 CI 环境 `PATH` 可找到 `kd_tool` 可执行脚本。
4. 将测试标记为 `@pytest.mark.smoke`，方便单独运行。

**验收标准**

1. CI 执行 `pytest -q` 时，新增 Smoke Test 文件中的两个测试全部通过
2. 测试脚本能够在无外部依赖的干净环境下运行，且不依赖硬编码路径（使用相对路径 `input/`）
3. Smoke Test 覆盖预过滤与默认主流程两种场景，并对日志与分析输出做关键断言
4. CI 中整体覆盖率不下降，测试执行时间合理（<30s）


### 原子任务 5：编写 Smoke Test 脚本

**文件位置**

* `tests/`
* 推荐新文件：`tests/test_smoke_end_to_end.py`

**目标**

* 在 CI 中对 `/input/` 目录一键跑通两种模式：

  1. **`--pre-filter` 模式**：仅执行预过滤并验证输出统计日志后退出
  2. **默认模式（无 flag）**：执行完整流程，至少跑到 MD5 分析，并输出 MD5 重复对统计

**实现细节**

1. 使用 `subprocess` 或 `click.testing.CliRunner` 模拟命令行调用：

   ```python
   result = runner.invoke(cli.main, ['scan-files', '--pre-filter', '--dir', 'input/'])
   ```
2. **断言**：

   * **`--pre-filter` 模式**：

     * `result.exit_code == 0`
     * `result.output` 包含 `[Prefilter] Scanned 4 files, filtered 2 duplicates → 2 remain.`
     * **无**后续 MD5 分析日志出现
   * **默认模式**：

     * `result.exit_code == 0`
     * `result.output` 包含 `[Prefilter] Scanned 4 files, filtered 2 duplicates → 2 remain.`
     * `result.output` 包含 `MD5 duplicates found: 1 pairs`
3. 如果使用 `CliRunner`，需在 `conftest.py` 中导入并准备 `runner` fixture；如果用 `subprocess`, 确保在 CI 环境 `PATH` 可找到 `kd_tool` 可执行脚本。
4. 将测试标记为 `@pytest.mark.smoke`，方便单独运行。

**验收标准**

1. CI 执行 `pytest -q` 时，新增 Smoke Test 文件中的两个测试全部通过
2. 测试脚本能够在无外部依赖的干净环境下运行，且不依赖硬编码路径（使用相对路径 `input/`）
3. Smoke Test 覆盖预过滤与默认主流程两种场景，并对日志与分析输出做关键断言
4. CI 中整体覆盖率不下降，测试执行时间合理（<30s）
