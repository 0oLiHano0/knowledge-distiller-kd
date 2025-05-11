**任务ID:** `KD-FEAT-007.1`

---

**一、 背景信息**

项目中同时存在 `FileStorage`（JSON 文件存储）与 `ORMStorage`（SQLite 存储）两套持久化实现。但运行时 `cli.py` 依然默认、唯一地实例化并使用 `FileStorage`，且无动态切换逻辑。本次任务旨在删除或禁用 `FileStorage`，为全项目切换到 `ORMStorage` 做准备。

---

**二、 任务目标 (Objective)**

* 从项目中移除或禁用所有与 `FileStorage` 相关的引用。
* 确保在 CLI 启动时不再实例化 `FileStorage`，同时保留代码目录整洁。

---

**三、 文件位置**

* `knowledge_distiller_kd/cli.py`
* `knowledge_distiller_kd/storage/file_storage.py`
* `knowledge_distiller_kd/storage/__init__.py`
* 可能涉及的测试文件（后续任务清理，这里只注记）

  * `tests/test_file_storage.py`
  * `tests/test_integration.py`（涉及 `FileStorage` 的场景）

---

**四、 子任务列表**

1. **注释或删除 `file_storage.py`**

   * 在 `knowledge_distiller_kd/storage/file_storage.py` 文件顶部添加注释说明“已弃用，保留仅作参考”，或直接删除该文件。
2. **移除 CLI 中的 `FileStorage` 实例化**

   * 打开 `cli.py`，删除以下代码行：

     ```python
     from .storage.file_storage import FileStorage
     …
     storage = FileStorage(base_path=storage_path)
     logger.info(f"FileStorage initialized with base path: {storage_path}")
     ```
3. **清理 `storage/__init__.py`**

   * 如果 `file_storage.py` 被删除，则从 `storage/__init__.py` 中移除对其的导入。
4. **保留但不引用 `FileStorage`**

   * 确保项目中不再有任何 `FileStorage(` 的调用点（可全文搜索并删除）。
5. **运行本地测试**

   * 执行 `pytest -q`，确认所有测试依赖 `FileStorage` 的部分暂时失败（这些测试会在后续 7.4 中重构）。
6. **提交变更**

   * 在 `feature/sqlite-persistence` 分支完成后，提交为一条清晰的 commit 信息：

     ```
     KD-FEAT-007: Disable FileStorage references in preparation for SQLite-only persistence
     ```
   * 推送到远程并更新 PR。

---

**五、 验收标准**

* **代码检查**：项目中不再存在对 `FileStorage` 的任何调用；`file_storage.py` 文件已标记为弃用或删除。
* **CLI 启动**：运行 `python -m knowledge_distiller_kd.cli -h` 不再报找不到 `FileStorage` 的错误。
* **测试反馈**：虽然部分测试会因移除 `FileStorage` 失败，但任务本身的变更不应引入新的、非预期的错误或异常。
* **Git 提交**：变更已提交至 `feature/sqlite-persistence` 分支，并附带符合规范的 commit message。

---
**六、其他**
注意：如在任务执行中发现其他问题，需在任务完成后文字说明


**任务ID:** `KD-FEAT-00`7.2

---

**一、 背景信息**

在 7.1 任务中，我们已移除所有对 `FileStorage` 的引用，项目切换到只保留 `ORMStorage`。但目前 CLI 启动仍未实例化并使用 `ORMStorage`，需要在命令行入口完成这一改造。

---

**二、任务目标 (Objective)**

* 修改 `cli.py`，在程序启动时导入并实例化 `ORMStorage`，并调用 `init_db()` 创建／校验数据库结构。
* 将该 `ORMStorage` 实例传递给 `KnowledgeDistillerEngine`，替代原先的 `FileStorage` 注入。

---

**三、文件位置**

* `knowledge_distiller_kd/cli.py`
* `knowledge_distiller_kd/storage/orm_storage.py` (确认 `ORMStorage` 可用)

---

**四、子任务列表**

1. \*\*导入 \*\***`ORMStorage`**

   ```python
   # 在 cli.py 顶部
   from .storage.orm_storage import ORMStorage
   ```
2. **实例化并初始化数据库**

   ```python
   # 替换原 FileStorage 实例化位置
   storage = ORMStorage(database_url=constants.DATABASE_URL)
   storage.init_db()
   logger.info(f"✅ ORMStorage initialized at {constants.DATABASE_URL}")
   ```
3. **传递给 Engine**

   ```python
   engine = KnowledgeDistillerEngine(
       storage=storage,
       input_dir=...,
       …
   )
   ```
4. **移除残余注释**

   * 删除所有与 `FileStorage` 相关的注释或占位代码。
5. **更新 Detect Storage Usage 测试**

   * 运行 `tests/storage/test_detect_storage_usage.py`，确认只有 `ORMStorage` 被实例化。
6. **本地测试**

   * 执行全量单元/集成测试 (`pytest -q`)，验证无误。

---

**五、验收标准**

* `cli.py` 仅导入并实例化 `ORMStorage`，无 `FileStorage` 引用。
* 启动 CLI（`python -m knowledge_distiller_kd.cli -h`）正常显示帮助信息，无错误。
* `tests/storage/test_detect_storage_usage.py` 报告中只出现 `ORMStorage`，且 `FileStorage` 实例化点已消失。
* 全量测试通过，无新增失败。

---

**六、其他**

如在改造过程中发现其他问题，请在任务完成后记录并同步，以便在后续任务中完善。




**任务ID:** `KD-FEAT-007.3`

---

### 一、背景信息

* 由于前期将 `Document.file_id` 字段改为 `nullable=False` 并为持久化存储切换到 `ORMStorage`，导致现有单元测试和集成测试在创建 `Document` 时未提供 `file_id` 参数，引发多处 `NOT NULL constraint failed: files.file_id` 错误。

---

### 二、任务目标 (Objective)

* 修复和更新所有直接或间接创建 `Document` 对象的测试用例，使其在构造时提供合法的 `file_id`，并保证所有与之关联的 `Block`、`AnalysisResult`、`UserDecision` 外键引用与模型定义保持一致，最终实现全量测试通过。

---

### 三、文件位置

以下测试文件中涉及 `Document` 实例化或插入，需要修改：

* `tests/core/test_models_registration.py`
* `tests/storage/test_persistence.py`
* `tests/storage/test_alembic_config.py` （若有直接 `Document` 构造）
* `tests/storage/test_migration_autogenerate.py`
* `tests/test_run_analysis_persistence.py`
* `tests/test_sqlite_storage.py`
* `tests/test_engine.py` 中若有使用真实 `Document`
* 以及任何其他直接调用 `Document(...)` 的测试模块

---

### 四、子任务列表

1. **统一生成 `file_id`**

   * 在测试中引入 `from uuid import uuid4`，为每个新建 `Document(...)` 提供 `file_id=str(uuid4())`。
   * 示例：

     ```python
     doc = Document(
         file_id=str(uuid4()),
         path="/tmp/foo.md",
         file_hash="abc123",
         type="md",
         size=123,
         ctime=None,
         mtime=None,
         status="processed"
     )
     ```
2. **调整外键引用**

   * 在测试中，若有直接使用 `doc.id` 插入 `Block`、`AnalysisResult` 或 `UserDecision`，改为：

     ```python
     # 先 session.add(doc); session.flush(); 
     # 再拿到 doc.id 或直接使用 doc.file_id 依据模型外键定义
     block = Block(
         block_id=str(uuid4()),
         file_id=doc.id,  # 如果外键指向 files.id
         …
     )
     ```
   * 若外键已改为指向 `files.file_id`，则确保使用 `file_id=doc.file_id`。
3. **更新测试数据构造**

   * 在所有因简化而生成测试 fixture（如字典或 DTO）中，加入 `file_id` 字段。
   * 验证 `TestPersistence`、`TestRelationships` 等测试场景都包含 `file_id`。
4. **执行并修正测试**

   * 运行 `pytest -q`，定位因 `file_id` 缺失或不匹配再次失败的用例，逐一修复。
   * 确保所有数据插入和关联都符合新的模型约束。
5. **文档与注释**

   * 在测试文件头部或共享 fixture 中，添加注释说明为何需要显式提供 `file_id`。
   * 保持一致的 UUID 生成策略或在需要时使用固定字符串以便断言。

---

### 五、验收标准

*  **前提**：所有因 `file_id` 缺失引发的 `NOT NULL constraint failed: files.file_id` 错误已被消除。
*  **测试通过**：执行 `pytest -q`，所有新旧测试均绿灯通过。
*  **关联验证**：`Block`、`AnalysisResult`、`UserDecision` 的外键关系测试均能正确读写。
*  **无回归**：确保更新后未引入其他测试失败或逻辑错误。

---

### 六、其他

* 任务过程中发现的其他问题，需在任务完成后汇总说明